"""Check Git-tracked protocol evidence, source syntax, links, and repository content."""
from pathlib import Path, PurePosixPath
import argparse
import ast
import hashlib
import json
import re
import sys
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cwru_bearing.data import load_config, load_manifest
from cwru_bearing.metrics import classification_metrics


def git_output(root, *arguments):
    """Read local Git metadata without modifying the index or working tree."""
    result = subprocess.run(["git", *arguments], cwd=root,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"Git inventory failed: {result.stderr.decode('utf-8', errors='replace').strip()}")
    return result.stdout


def tracked_files(root=None):
    """Return existing index-tracked files, rejecting missing or escaping paths."""
    root = Path(root or ROOT).resolve()
    top = git_output(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    if Path(top).resolve() != root:
        raise ValueError("Verification must run at the repository root")
    names = git_output(root, "ls-files", "--cached", "-z").decode("utf-8").split("\0")
    names = [name for name in names if name]
    if not names or len(names) != len(set(names)):
        raise ValueError("Git inventory is empty or has unresolved duplicate entries")
    paths = []
    for name in sorted(names):
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts
                or "\\" in name or ":" in name or relative.as_posix() != name):
            raise ValueError(f"Invalid tracked path: {name}")
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Tracked path escapes repository: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing tracked file: {name}; stage intended removals before verification")
        paths.append(path)
    return paths


def verify_inventory(root=None):
    """Reject non-ignored untracked files as well as invalid tracked inventory.

    Git metadata and ignored runtime artifacts are excluded by Git. Intended
    additions and removals must be staged to update the expected inventory.
    """
    root = Path(root or ROOT).resolve()
    paths = tracked_files(root)
    unexpected = git_output(root, "ls-files", "--others", "--exclude-standard", "-z").decode("utf-8").split("\0")
    unexpected = sorted(name for name in unexpected if name)
    if unexpected:
        raise ValueError(f"Untracked repository files: {unexpected}")
    return paths


def scan(files):
    """Check selected UTF-8 content, including decoded notebook JSON strings."""
    patterns = [r"(?i)[A-Z]:[\\/]+Users[\\/]+[^\\/\s]+",
                r"gh[pousr]_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}",
                r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}",
                r"AKIA[A-Z0-9]{16}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"]
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, child in value.items():
                yield str(key)
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)
    for path in files:
        if path.suffix == ".png":
            from PIL import Image
            with Image.open(path) as image:
                text = "\n".join(str(value) for value in image.info.values())
                image.verify()
            if any(re.search(pattern, text) for pattern in patterns):
                raise ValueError(f"Private metadata in {path.name}")
            continue
        if path.suffix not in {".py", ".md", ".txt", ".json", ".csv", ".toml", ".yml", ".cff", ".svg", ".ipynb", ""} and path.name not in {".gitignore", ".gitattributes"}:
            raise ValueError(f"Unsupported tracked file type: {path.name}")
        text = path.read_text(encoding="utf-8")
        if path.suffix in {".json", ".ipynb"}:
            text += "\n" + "\n".join(strings(json.loads(text)))
        if any(re.search(pattern, text) for pattern in patterns):
            raise ValueError(f"Private path or credential pattern in {path.name}")
        if any(marker in text for marker in ["\ufffd", "\u00e2\u20ac", "\u00c3\u00a2"]):
            raise ValueError(f"Malformed text encoding in {path.name}")
        if re.search(r"[\u0600-\u06ff]", text):
            raise ValueError(f"Non-English research text requires review: {path.name}")
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if any(output.get("output_type") == "error" for output in cell.get("outputs", [])):
                    raise ValueError(f"Notebook has an error output: {path.name}")


def check_links(paths=None):
    paths = tracked_files() if paths is None else paths
    count = 0
    documents = [(p, p.read_text(encoding="utf-8")) for p in paths if p.suffix == ".md"]
    for path in (p for p in paths if p.suffix == ".ipynb"):
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
    parser.add_argument("--strict", action="store_true", help="Require Git-tracked inventory with no non-ignored untracked files")
    args = parser.parse_args()
    paths = verify_inventory() if args.strict else tracked_files()
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
    print(json.dumps({"public_files": len(paths), "local_links": check_links(paths),
                      "manifest_fingerprints": "match", "historical_metrics": "match", "classical_observations": check_classical_evidence(),
                      "syntax_and_content_scan": "passed", "strict_inventory": args.strict}, indent=2))


if __name__ == "__main__":
    main()
