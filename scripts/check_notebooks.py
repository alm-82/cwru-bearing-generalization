"""Validate the five public notebook schemas, code cells, and saved outputs."""
from pathlib import Path
import ast
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells, 1):
            if cell.cell_type == "code":
                ast.parse(cell.source, filename=f"{path.name}:cell{index}")
                if any(output.output_type == "error" for output in cell.outputs):
                    raise ValueError(f"Error output: {path.name}, cell {index}")
        print(f"Valid: {path.name} ({len(notebook.cells)} cells)")


if __name__ == "__main__":
    main()
