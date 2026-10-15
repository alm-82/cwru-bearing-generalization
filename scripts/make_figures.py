"""Render saved confusion matrices without model inference or new experiments."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    results = json.loads((ROOT / "results/historical/results.json").read_text(encoding="utf-8"))
    labels = [["H", "IR7", "B7", "OR7", "IR14", "B14", "OR14", "IR21", "B21", "OR21"],
              ["Healthy", "IR", "Ball", "OR"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    for ax, key, names, title in zip(axes, ["cross_load", "cross_fault_size"], labels,
                                    ["Cross-load", "Cross-fault-size"]):
        matrix = np.asarray(results[key]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
        for row in range(len(matrix)):
            for column in range(len(matrix)):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=9,
                        color="white" if matrix[row, column] > matrix.max() * .5 else "#202020")
        ax.set(xticks=np.arange(len(matrix)), yticks=np.arange(len(matrix)),
               xticklabels=names, yticklabels=names, xlabel="Predicted class", ylabel="True class",
               title=f"{title}: {matrix.trace()}/{matrix.sum()} correct")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Historical notebook evidence — original trained weights unavailable", fontsize=14)
    fig.savefig(ROOT / "figures/historical_confusion_matrices.png", dpi=160)
    fig.savefig(ROOT / "figures/historical_confusion_matrices.svg")
    plt.close(fig)
    print("Rendered the saved historical confusion matrices")


if __name__ == "__main__":
    main()
