"""Export an explicitly selected research checkout without Git history."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (
    '.gitattributes',
    '.github/workflows/checks.yml',
    '.gitignore',
    'CITATION.cff',
    'LICENSE',
    'README.md',
    'THIRD_PARTY.md',
    'configs/cross_fault_size.json',
    'configs/cross_load.json',
    'configs/stft_a.json',
    'configs/stft_b.json',
    'data/README.md',
    'data/historical_reference.json',
    'data/manifests/manifest.csv',
    'data/manifests/manifest_crossfaultsize.csv',
    'data/manifests/manifest_crossload_10class.csv',
    'data/recordings.csv',
    'docs/protocols.md',
    'docs/exploration.md',
    'docs/reproduction.md',
    'docs/scientific_audit.md',
    'docs/traceability.md',
    'docs/validation_report.md',
    'figures/README.md',
    'figures/historical_confusion_matrices.png',
    'figures/historical_confusion_matrices.svg',
    'notebooks/01_signal_exploration.ipynb',
    'notebooks/02_classical_baselines.ipynb',
    'notebooks/03_stft_exploration.ipynb',
    'notebooks/04_cross_load.ipynb',
    'notebooks/05_cross_fault_size.ipynb',
    'pyproject.toml',
    'requirements-inspected-cpu.txt',
    'requirements-notebooks-windows.txt',
    'results/historical/D_cell_023.txt',
    'results/historical/D_cell_025.txt',
    'results/historical/D_cell_043.txt',
    'results/historical/D_cell_050.txt',
    'results/historical/D_cell_064.txt',
    'results/historical/D_cell_072.txt',
    'results/historical/D_cell_079.txt',
    'results/historical/README.md',
    'results/historical/results.json',
    'results/historical/classical_svm_evidence.json',
    'results/historical/signal_exploration_evidence.json',
    'scripts/check_notebooks.py',
    'scripts/execute_notebooks.py',
    'scripts/make_figures.py',
    'scripts/prepare_publication.py',
    'scripts/verify_repository.py',
    'src/cwru_bearing/__init__.py',
    'src/cwru_bearing/classical.py',
    'src/cwru_bearing/exploration.py',
    'src/cwru_bearing/cli.py',
    'src/cwru_bearing/data.py',
    'src/cwru_bearing/metrics.py',
    'src/cwru_bearing/models.py',
    'src/cwru_bearing/preprocessing.py',
    'src/cwru_bearing/training.py',
    'tests/test_cleanup.py',
    'tests/test_models.py',
    'tests/test_protocols.py',
    'tests/test_exploration.py',
    'tests/test_integrity.py',
)


def public_files(source=None, names=None):
    """Resolve exact approved paths; fail on duplicates, missing files, or escapes."""
    source = Path(source or ROOT).resolve()
    names = PUBLIC_FILES if names is None else tuple(names)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate publication path")
    paths = []
    for name in names:
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or ":" in name or relative.as_posix() != name
                or any(part.startswith(".") and part not in {".github", ".gitignore", ".gitattributes"}
                       for part in relative.parts)):
            raise ValueError(f"Invalid publication path: {name}")
        path = source / name
        if not path.resolve().is_relative_to(source):
            raise ValueError(f"Publication path escapes source: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing approved file: {name}")
        paths.append(path)
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
            raise ValueError(f"Unapproved file type: {path.name}")
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


def verify_candidate(candidate):
    """Require an exact public file inventory, including hidden files."""
    root = Path(candidate).resolve()
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    expected = set(PUBLIC_FILES)
    if actual != expected:
        raise ValueError(f"Candidate inventory differs: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")
    if any(p.name == ".git" for p in root.rglob("*")):
        raise ValueError("Git history is not publication content")
    paths = public_files(root)
    scan(paths)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def export_candidate(destination, source=None):
    """Create a new candidate; never overwrite a file or directory."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Candidate already exists; choose a new destination")
    source = Path(source or ROOT).resolve()
    target = destination.resolve()
    if target == source or (target.is_relative_to(source) and not target.is_relative_to(source / "work")):
        raise ValueError("Use a separate candidate directory or an ignored work directory")
    paths = public_files(source)
    scan(paths)
    target.mkdir(parents=True, exist_ok=False)
    for path in paths:
        output = target / path.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)
    return verify_candidate(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New publication candidate directory")
    args = parser.parse_args()
    hashes = export_candidate(args.output)
    print(json.dumps({"files": len(hashes), "output": str(args.output), "git_history": False}))


if __name__ == "__main__":
    main()
