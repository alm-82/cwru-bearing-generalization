"""Check public protocol fingerprints, source syntax, links, and publication content."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cwru_bearing.data import load_config, load_manifest
from cwru_bearing.metrics import classification_metrics
from prepare_publication import public_files, scan, verify_candidate


def check_links():
    count = 0
    documents = [(p, p.read_text(encoding="utf-8")) for p in public_files() if p.suffix == ".md"]
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        documents.extend((path, "".join(cell["source"])) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    for path, text in documents:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            relative = unquote(target.split("#")[0])
            if not (path.parent / relative).exists():
                raise ValueError(f"Broken local link in {path.relative_to(ROOT)}: {target}")
            count += 1
    return count


def check_classical_evidence():
    evidence = json.loads((ROOT / "results/historical/classical_svm_evidence.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "results/historical/results.json").read_text(encoding="utf-8"))["classical"]
    expected = {
        "fixed_size_cross_load_four_features": (82, "fixed_size_cross_load_accuracy"),
        "fixed_size_cross_load_rms_only": (85, "fixed_size_cross_load_rms_only_accuracy"),
        "fault_only_size_transfer_four_features": (102, "fault_only_size_transfer_accuracy"),
        "fault_only_size_transfer_rms_only": (103, "fault_only_size_transfer_rms_only_accuracy"),
        "joint_load_size_transfer_four_features": (93, "joint_load_size_transfer_four_features_accuracy"),
        "joint_load_size_transfer_rms_only": (95, "joint_load_size_transfer_rms_only_accuracy"),
    }
    observations = evidence["observations"]
    if len(observations) != 6 or {o["id"] for o in observations} != set(expected):
        raise ValueError("Classical evidence inventory differs")
    for observation in observations:
        cell, result_key = expected[observation["id"]]
        saved = observation["saved_output"]
        printed_accuracy = float(saved.splitlines()[0].rsplit(":", 1)[1])
        printed_matrix = [[int(x) for x in row.split()] for row in re.findall(r"\[([ \d]+)\]", saved)]
        matrix = observation["confusion_matrix"]
        accuracy = sum(row[i] for i, row in enumerate(matrix)) / sum(map(sum, matrix))
        if (observation["source_cell"] != cell or printed_matrix != matrix
                or len(observation["class_order"]) != len(matrix)
                or printed_accuracy != observation["accuracy"]
                or results[result_key] != printed_accuracy
                or abs(accuracy - printed_accuracy) > 1e-15):
            raise ValueError(f"Classical numerical evidence differs: {observation['id']}")
    return len(observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Require exact candidate inventory")
    args = parser.parse_args()
    paths = public_files()
    for name in ["stft_a", "stft_b", "cross_load", "cross_fault_size"]:
        load_manifest(load_config(name))
    reference = json.loads((ROOT / "data/historical_reference.json").read_text(encoding="utf-8"))
    if hashlib.sha256((ROOT / "data/recordings.csv").read_bytes()).hexdigest() != reference["recordings_sha256"]:
        raise ValueError("Recording metadata fingerprint changed")
    for path in paths:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    results = json.loads((ROOT / "results/historical/results.json").read_text(encoding="utf-8"))
    for name in ["cross_load", "cross_fault_size"]:
        saved = results[name]
        labels, predictions = [], []
        for i, row in enumerate(saved["confusion_matrix"]):
            for j, count in enumerate(row):
                labels.extend([i] * count)
                predictions.extend([j] * count)
        metrics = classification_metrics(labels, predictions, range(len(saved["confusion_matrix"])))
        if any(abs(metrics[key] - saved[key]) > 1e-12 for key in ["accuracy", "macro_f1"]):
            raise ValueError(f"Historical metric transcription differs: {name}")
    scan(paths)
    if args.strict:
        verify_candidate(ROOT)
    print(json.dumps({"public_files": len(paths), "local_links": check_links(),
                      "manifest_fingerprints": "match", "historical_metrics": "match", "classical_observations": check_classical_evidence(),
                      "syntax_and_content_scan": "passed", "strict_inventory": args.strict}, indent=2))


if __name__ == "__main__":
    main()
