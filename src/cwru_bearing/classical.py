"""Exploratory time-domain baseline utilities.

No fitting occurs at import. These methods are separate from the final CNN protocols.
"""
import numpy as np
from scipy.stats import kurtosis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def windows(signal, length=2048):
    """Return complete non-overlapping sample windows, omitting the final remainder."""
    return np.asarray([signal[start:start + length]
                       for start in range(0, len(signal) - length + 1, length)])


def features(window):
    """Return RMS, absolute peak, Pearson kurtosis, and peak/RMS in that order."""
    rms = np.sqrt(np.mean(window ** 2))
    peak = np.max(np.abs(window))
    # Intentionally retain historical kurtosis definition and division behavior.
    return rms, peak, kurtosis(window, fisher=False), peak / rms


def svm_pipeline():
    return Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf"))])


def run_baselines(data_root):
    """Reconstruct the fixed-size and fault-only SVM experiments using new execution."""
    import pandas as pd
    from .data import load_config, load_manifest, load_signal, validate_recordings
    from .metrics import classification_metrics

    config = load_config("cross_load")
    records = load_manifest(config).drop_duplicates("file_id")
    records = records[(records.class_label == 0) | records.fault_size_inch.isin([.007, .014])]
    validate_recordings(data_root, config, file_ids=records.file_id.unique())
    columns = ["RMS", "Peak", "Kurtosis", "CrestFactor"]
    rows = []
    for _, record in records.iterrows():
        label = "Healthy" if record.class_label == 0 else record.fault_type
        for index, window in enumerate(windows(load_signal(data_root, record.file_id))):
            rows.append({"file_id": int(record.file_id), "window": index, "label": label,
                         "load_hp": record.load_hp, "size": record.fault_size_inch,
                         **dict(zip(columns, features(window)))})
    frame = pd.DataFrame(rows)
    fixed = frame[(frame.label == "Healthy") | (frame["size"] == .007)]
    cases = [
        ("fixed_size_cross_load", fixed[fixed.load_hp.isin([0, 1, 2])], fixed[fixed.load_hp == 3]),
        ("fault_only_size_transfer", frame[frame["size"] == .007], frame[frame["size"] == .014]),
        # Load and fault size shift together, with healthy windows retained.
        ("joint_load_size_transfer", fixed[fixed.load_hp.isin([0, 1, 2])],
         frame[(frame.load_hp == 3) & ((frame.label == "Healthy") | (frame["size"] == .014))]),
    ]
    results = {"evidence_status": "newly_executed_classical_experiments"}
    for name, train, test in cases:
        order = sorted(set(train.label) | set(test.label))
        for features_used, suffix in [(columns, "four_features"), (["RMS"], "rms_only")]:
            model = svm_pipeline().fit(train[features_used], train.label)
            predictions = model.predict(test[features_used])
            results[f"{name}_{suffix}"] = {
                **classification_metrics(test.label, predictions, order),
                "train_windows": len(train), "test_windows": len(test),
                "confusion_label_order": order,
            }
    return results
