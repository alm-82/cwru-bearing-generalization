"""Historical STFT profiles, without resampling."""
from pathlib import Path
import json
import numpy as np
from scipy.signal import stft
from PIL import Image
from .data import load_signal, load_manifest, segment, historical_reference, semantic_digest, validate_config


PREPROCESSING_FIELDS = (
    "protocol_version", "processing_rate_hz", "segment_length", "nperseg", "noverlap",
    "nfft", "window", "boundary", "padded", "epsilon", "percentiles", "flip_vertical",
    "signal_dtype", "quantization", "manifest", "classes", "image_pattern",
)


def preprocessing_identity(config):
    """Fields determining saved pixels and their scientific assignment, not training."""
    return {key: config[key] for key in PREPROCESSING_FIELDS}


def spectrogram_db(x, config):
    validate_config(config)
    if len(x) != config["segment_length"]:
        raise ValueError("Incorrect segment length")
    _, _, z = stft(x, fs=config["processing_rate_hz"], window=config["window"],
                   nperseg=config["nperseg"], noverlap=config["noverlap"],
                   nfft=config["nfft"], boundary=config["boundary"], padded=config["padded"])
    db = 20 * np.log10(np.abs(z) + config["epsilon"])
    frames = 1 + (config["segment_length"] - config["nperseg"]) // (config["nperseg"] - config["noverlap"])
    if db.shape != (config["nfft"] // 2 + 1, frames):
        raise ValueError(f"Unexpected STFT dimensions: {db.shape}")
    return db


def fit_limits(specs, manifest, config):
    values = [specs[i].ravel() for i in manifest.index[manifest.split == "train"]]
    if not values:
        raise ValueError("No training spectra")
    values = np.concatenate(values)
    # Separate scalar calls preserve historical NumPy float32 percentile precision.
    # A vector-valued percentile call promotes the cross-load result to float64.
    lo = np.percentile(values, config["percentiles"][0])
    hi = np.percentile(values, config["percentiles"][1])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Degenerate train-fitted limits")
    return lo, hi


def quantize(db, limits, config):
    lo, hi = limits
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Invalid limits")
    pixels = (np.clip(db, lo, hi) - lo) / (hi - lo) * 255
    if config["quantization"] == "round":
        pixels = pixels.round()  # NumPy ties-to-even rounding.
    elif config["quantization"] != "truncate":
        raise ValueError("Unknown quantization")
    pixels = pixels.astype(np.uint8)
    return np.flipud(pixels) if config["flip_vertical"] else pixels


def image_relative_path(row, config):
    values = {"file_id": int(row.file_id), "class_label": int(row.class_label), "segment_idx": int(row.segment_idx)}
    return Path(row.split) / str(values["class_label"]) / config["image_pattern"].format(**values)


def compute_specs(data_root, manifest, config):
    cache, specs = {}, {}
    for i, row in manifest.iterrows():
        if row.split == "gap":
            continue
        if row.file_id not in cache:
            cache[row.file_id] = load_signal(data_root, row.file_id, config["signal_dtype"])
        x = segment(cache[row.file_id], row.sample_start, row.sample_end, config["segment_length"])
        specs[i] = spectrogram_db(x, config)
    return specs


def generate(data_root, output_root, config):
    """Create a new image directory; never overwrite an existing dataset."""
    validate_config(config)
    out = Path(output_root)
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    manifest = load_manifest(config)
    if preprocessing_identity(config) != historical_reference(config)["preprocessing_identity"]:
        raise ValueError("This historical profile does not support altered preprocessing")
    specs = compute_specs(data_root, manifest, config)
    limits = fit_limits(specs, manifest, config)
    out.mkdir(parents=True)
    for i, row in manifest.iterrows():
        if row.split == "gap":
            continue
        path = out / image_relative_path(row, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(quantize(specs[i], limits, config)).save(path)
    metadata = {"evidence_status": "newly_generated_preprocessing", "config": config,
                "limits": [float(x) for x in limits], "fit_split": "train", "resampled": False,
                "schema_version": 2, "preprocessing_identity": preprocessing_identity(config),
                "manifest_semantic_sha256": semantic_digest(manifest),
                "reference_manifest_sha256": historical_reference(config)["manifest_sha256"]}
    (out / "preprocessing.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest.to_csv(out / "manifest.csv", index=False)
    return metadata


def compare_historical(data_root, historical_root, config):
    """Read-only pixel comparison; produces no images and fits train-only limits."""
    df = load_manifest(config)
    specs = compute_specs(data_root, df, config)
    limits = fit_limits(specs, df, config)
    exact = total = changed = max_difference = 0
    for i, row in df.iterrows():
        if row.split == "gap":
            continue
        new = quantize(specs[i], limits, config)
        with Image.open(Path(historical_root) / image_relative_path(row, config)) as im:
            old = np.asarray(im)
        if old.shape != new.shape:
            raise ValueError("Historical image dimensions differ")
        delta = np.abs(new.astype(np.int16) - old.astype(np.int16))
        total += 1
        exact += int(not delta.any())
        changed += int(np.count_nonzero(delta))
        max_difference = max(max_difference, int(delta.max()))
    return {"profile": config["name"], "images": total, "exact_images": exact,
            "changed_pixels": changed, "max_pixel_difference": max_difference, "limits": [float(x) for x in limits]}
