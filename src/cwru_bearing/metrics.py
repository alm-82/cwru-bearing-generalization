"""Classification metrics and historical validation selection rules."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix


def classification_metrics(labels, predictions, class_order=None):
    if len(labels) == 0 or len(labels) != len(predictions):
        raise ValueError("Empty or misaligned predictions")
    if class_order is not None:
        class_order = list(class_order)
        if not class_order or len(set(class_order)) != len(class_order):
            raise ValueError("Class order must be nonempty and distinct")
        if not set(labels).union(predictions).issubset(class_order):
            raise ValueError("Observed labels are outside the declared class order")
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", labels=class_order, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=class_order).tolist()}


def is_better(f1, loss, best_f1, best_loss, tie_break):
    if not np.isfinite([f1, loss]).all():
        raise ValueError("Nonfinite validation metric")
    same = np.isclose(f1, best_f1) if tie_break == "isclose" else f1 == best_f1
    if tie_break not in {"exact", "isclose"}:
        raise ValueError("Unknown selection rule")
    return bool(f1 > best_f1 or (same and loss < best_loss))
