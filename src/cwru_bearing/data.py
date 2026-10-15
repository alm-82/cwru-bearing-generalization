"""Exact signal selection and validation of the preserved manifests."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]


def validate_config(config):
    """Reject malformed settings before loading data or creating run artifacts."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    required = {"name", "protocol_version", "manifest", "classes", "segment_length",
                "processing_rate_hz", "nperseg", "noverlap", "nfft", "window",
                "boundary", "padded", "epsilon", "percentiles", "signal_dtype",
                "quantization", "flip_vertical", "image_pattern", "image_size",
                "dropout", "weight_decay", "batch_size", "num_workers", "pin_memory",
                "seed", "loader_generator_seed", "tie_break", "transform_order",
                "stage1", "stage2"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(sorted(missing))}")

    def integer(name, value, minimum=1):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    def number(name, value, minimum=0, strict=False):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or (value <= minimum if strict else value < minimum)):
            raise ValueError(f"Invalid {name}: expected a finite number {'>' if strict else '>='} {minimum}")

    if not isinstance(config["name"], str) or config["name"] not in {"stft_a", "stft_b", "cross_load", "cross_fault_size"}:
        raise ValueError("Unknown protocol name")
    for field in ["segment_length", "nperseg", "nfft", "image_size", "batch_size"]:
        integer(field, config[field])
    integer("noverlap", config["noverlap"], 0)
    integer("num_workers", config["num_workers"], 0)
    if not 0 <= config["noverlap"] < config["nperseg"] <= config["segment_length"]:
        raise ValueError("STFT requires 0 <= noverlap < nperseg <= segment_length")
    if config["nfft"] < config["nperseg"]:
        raise ValueError("nfft must be >= nperseg")
    number("processing_rate_hz", config["processing_rate_hz"], strict=True)
    number("epsilon", config["epsilon"], strict=True)
    number("weight_decay", config["weight_decay"])
    number("dropout", config["dropout"])
    if config["dropout"] >= 1:
        raise ValueError("dropout must be < 1")
    percentiles = config["percentiles"]
    if not isinstance(percentiles, list) or len(percentiles) != 2:
        raise ValueError("percentiles must contain two ordered values")
    for value in percentiles:
        number("percentile", value)
    if not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    for field, allowed in {
        "transform_order": {"grayscale_then_resize", "resize_then_grayscale"},
        "tie_break": {"exact", "isclose"}, "signal_dtype": {"float32", "float64"},
        "quantization": {"round", "truncate"}, "window": {"hann"},
    }.items():
        if not isinstance(config[field], str) or config[field] not in allowed:
            raise ValueError(f"Invalid {field}: {config[field]!r}")
    if config["boundary"] is not None or config["padded"] is not False:
        raise ValueError("Historical STFT requires boundary=None and padded=False")
    for field in ["flip_vertical", "pin_memory"]:
        if not isinstance(config[field], bool):
            raise ValueError(f"{field} must be boolean")
    classes = config["classes"]
    if (not isinstance(classes, list) or not classes
            or any(not isinstance(x, str) or not x for x in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError("classes must be a nonempty list of distinct names")
    for field in ["seed", "loader_generator_seed"]:
        if config[field] is not None:
            integer(field, config[field], 0)
    manifest = config["manifest"]
    if not isinstance(manifest, str) or Path(manifest).name != manifest or not manifest.endswith(".csv"):
        raise ValueError("manifest must be a CSV basename")
    for stage in ["stage1", "stage2"]:
        settings = config[stage]
        if stage == "stage2" and settings is None:
            continue
        if not isinstance(settings, dict):
            raise ValueError(f"{stage} must be a settings mapping")
        integer(f"{stage}.epochs", settings.get("epochs"))
        number(f"{stage}.classifier_lr", settings.get("classifier_lr"), strict=True)
        modes = {"features_eval"} if stage == "stage1" else {"last_projection_train", "all_bn_eval"}
        if not isinstance(settings.get("mode"), str) or settings["mode"] not in modes:
            raise ValueError(f"Invalid {stage} training mode")
        if stage == "stage2":
            number(f"{stage}.backbone_lr", settings.get("backbone_lr"), strict=True)
    return config


def load_config(name):
    if not isinstance(name, str) or name not in {"stft_a", "stft_b", "cross_load", "cross_fault_size"}:
        raise ValueError(f"Unknown protocol: {name}")
    path = ROOT / "configs" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown protocol: {name}")
    return validate_config(json.loads(path.read_text(encoding="utf-8")))


def historical_reference(config):
    """Independent, versioned fingerprints of the preserved scientific inputs."""
    references = json.loads((ROOT / "data/historical_reference.json").read_text(encoding="utf-8"))
    try:
        return references["profiles"][config["name"]]
    except KeyError as exc:
        raise ValueError("No historical reference for this profile") from exc


def semantic_digest(df):
    """Hash scientific fields, including missing healthy fault sizes, in row order."""
    columns = ["file_id", "class_label", "split", "segment_idx", "sample_start",
               "sample_end", "load_hp", "fault_type", "fault_size_inch"]
    records = [{k: None if pd.isna(v) else v for k, v in row.items()}
               for row in df[columns].to_dict("records")]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_manifest(config):
    validate_config(config)
    path = ROOT / "data/manifests" / config["manifest"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != historical_reference(config)["manifest_sha256"]:
        raise ValueError("Preserved manifest bytes differ from the independent reference")
    df = pd.read_csv(path)
    # The originals use both numeric IDs and '<id>.mat'. Preserve archived CSVs.
    df["file_id"] = df["file_id"].map(lambda x: int(str(x).removesuffix(".mat")))
    df["filename"] = df["file_id"].map(lambda x: f"{x}.mat")
    validate_manifest(df, config)
    return df


def load_signal(data_root, file_id, dtype="float64"):
    """Never select the first DE key: 99.mat also contains X098 arrays."""
    file_id = int(file_id)
    path = Path(data_root) / f"{file_id}.mat"
    key = f"X{file_id:03d}_DE_time"
    mat = loadmat(path)
    if key not in mat:
        raise KeyError(f"{path.name}: missing expected key {key}")
    signal = np.asarray(mat[key]).ravel().astype(dtype)
    if not np.isfinite(signal).all():
        raise ValueError(f"Nonfinite samples in {path.name}")
    return signal


def segment(signal, start, end, length=4096):
    start, end = int(start), int(end)
    if start < 0 or end > len(signal) or end - start != length:
        raise ValueError(f"Invalid segment [{start}, {end}) for {len(signal)} samples")
    return signal[start:end]


def validate_manifest(df, config):
    required = {"file_id", "class_label", "split", "segment_idx", "sample_start", "sample_end", "load_hp", "fault_type", "fault_size_inch"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    if df.empty or not set(df.split).issubset({"train", "val", "test", "gap"}):
        raise ValueError("Empty manifest or unknown split")
    for col in ["file_id", "class_label", "segment_idx", "sample_start", "sample_end", "load_hp"]:
        if df[col].isna().any() or not np.equal(df[col], np.floor(df[col])).all():
            raise ValueError(f"Noninteger values in {col}")
    if "filename" in df and not (df.filename == df.file_id.map(lambda x: f"{int(x)}.mat")).all():
        raise ValueError("Recording filename and ID disagree")
    if df.duplicated(["file_id", "segment_idx"]).any():
        raise ValueError("Repeated recording/segment pair")
    if not (df.sample_start == df.segment_idx * config["segment_length"]).all():
        raise ValueError("Unexpected segment offsets")
    if not (df.sample_end - df.sample_start == config["segment_length"]).all():
        raise ValueError("Unexpected segment lengths")
    if (df.sample_start < 0).any():
        raise ValueError("Negative sample offset")
    expected = set(range(len(config["classes"])))
    for split in ["train", "val", "test"]:
        if set(df.loc[df.split == split, "class_label"]) != expected:
            raise ValueError(f"Class mapping incomplete in {split}")
    for _, group in df.groupby("file_id"):
        g = group.sort_values("sample_start")
        if np.any(g.sample_start.to_numpy()[1:] < g.sample_end.to_numpy()[:-1]):
            raise ValueError("Overlapping sample intervals")
    if config["name"] == "cross_load":
        if (df.groupby("file_id").split.nunique() != 1).any():
            raise ValueError("Recording leakage in cross-load")
        mapping = {0: "train", 1: "train", 2: "val", 3: "test"}
        if not (df.load_hp.map(mapping) == df.split).all():
            raise ValueError("Cross-load assignment changed")
    reference = historical_reference(config)
    if config["classes"] != reference["preprocessing_identity"]["classes"]:
        raise ValueError("Historical class names or ordering changed")
    if semantic_digest(df) != reference["semantic_sha256"]:
        raise ValueError("Manifest scientific fields differ from the independent reference")



def validate_recordings(data_root, config, check_hashes=True, file_ids=None):
    """Validate selected protocol recordings against the preserved local evidence."""
    df = load_manifest(config)
    if file_ids is not None:
        selected = set(file_ids)
        if not selected or not selected.issubset(set(df.file_id)):
            raise ValueError("Requested recordings are outside the protocol")
        df = df[df.file_id.isin(selected)]
    recording_path = ROOT / "data/recordings.csv"
    references = json.loads((ROOT / "data/historical_reference.json").read_text(encoding="utf-8"))
    if hashlib.sha256(recording_path.read_bytes()).hexdigest() != references["recordings_sha256"]:
        raise ValueError("Recording metadata differs from the independent reference")
    metadata = pd.read_csv(recording_path).set_index("file_id")
    reports = []
    for fid, group in df.groupby("file_id"):
        path = Path(data_root) / f"{fid}.mat"
        x = load_signal(data_root, fid, config["signal_dtype"])
        if len(x) != int(metadata.loc[fid, "samples"]) or group.sample_end.max() > len(x):
            raise ValueError(f"Signal length or manifest bounds differ: {fid}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if check_hashes else None
        if check_hashes and digest != metadata.loc[fid, "sha256"]:
            raise ValueError(f"Local-reference checksum mismatch: {fid}; inspect provenance before use")
        reports.append({"file_id": int(fid), "samples": len(x), "sha256": digest})
    return reports
