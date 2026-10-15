"""Training, validation-selected checkpoints, and evaluation for the recorded protocols."""
from pathlib import Path
import copy
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import numpy as np
import torch
from torch import nn
from .data import ROOT, load_manifest, validate_config
from .models import build_model, build_loaders, configure_stage, set_training_mode
from .metrics import classification_metrics, is_better


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ["numpy", "scipy", "pandas", "Pillow", "scikit-learn", "torch", "torchvision"]},
            "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def evaluate(model, loader, device, class_order=None):
    model.eval()
    labels, predictions, loss_sum = [], [], 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            logits = model(images)
            loss_sum += criterion(logits, target).item() * len(target)
            labels.extend(target.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    result = classification_metrics(labels, predictions, class_order)
    result["loss"] = loss_sum / len(loader.dataset)
    return result, labels, predictions


def train_stage(model, loaders, config, stage, device, out):
    validate_config(config)
    settings = config[f"stage{stage}"]
    configure_stage(model, stage)
    if stage == 1:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": settings["classifier_lr"]}]
    else:
        groups = [{"params": model.features[-1].parameters(), "lr": settings["backbone_lr"]},
                  {"params": model.classifier.parameters(), "lr": settings["classifier_lr"]}]
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    best_f1, best_loss, best_epoch = -1.0, float("inf"), None
    best_state, best_optimizer, history = None, None, []
    for epoch in range(1, settings["epochs"] + 1):
        set_training_mode(model, settings["mode"])
        train_loss, labels, predictions = 0.0, [], []
        for images, target in loaders["train"]:
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target)
            labels.extend(target.detach().cpu().tolist())
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
        train = classification_metrics(labels, predictions, range(len(config["classes"])))
        val, _, _ = evaluate(model, loaders["val"], device, range(len(config["classes"])))
        row = {"epoch": epoch, "train_loss": train_loss / len(loaders["train"].dataset),
               "train_accuracy": train["accuracy"], "train_macro_f1": train["macro_f1"],
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        # Persist partial history even if a later epoch is interrupted.
        save_json(out / f"stage{stage}_history.json", history)
        print(f"stage={stage} epoch={epoch} val_F1={val['macro_f1']:.4f} val_loss={val['loss']:.4f}")
        if is_better(val["macro_f1"], val["loss"], best_f1, best_loss, config["tie_break"]):
            best_f1, best_loss, best_epoch = val["macro_f1"], val["loss"], epoch
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            checkpoint = {"model_state": best_state, "optimizer_state": best_optimizer,
                          "config": config, "stage": stage, "best_epoch": best_epoch,
                          "best_val_macro_f1": best_f1, "best_val_loss": best_loss,
                          "history": history, "evidence_status": "newly_executed_experiment"}
            temp = out / f"stage{stage}_best.pth.tmp"
            torch.save(checkpoint, temp)
            temp.replace(out / f"stage{stage}_best.pth")
    model.load_state_dict(best_state)
    columns = [k for k in history[0] if k != "val_confusion_matrix"]
    with (out / f"stage{stage}_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(history)
    return best_epoch


def validate_image_dataset(image_root, config):
    """Decode every image and verify pixels against independent historical references.

    PNG encoding may vary; decoded grayscale pixels must be identical. Old metadata
    must be regenerated explicitly. Runtime settings do not determine image identity.
    """
    from PIL import Image
    import pandas as pd
    from .data import historical_reference, semantic_digest, validate_manifest
    from .preprocessing import image_relative_path, preprocessing_identity
    root = Path(image_root)
    df = load_manifest(config)
    reference = historical_reference(config)
    identity = preprocessing_identity(config)
    if identity != reference["preprocessing_identity"]:
        raise ValueError("Preprocessing differs from the historical profile")
    expected = {image_relative_path(row, config).as_posix()
                for _, row in df.iterrows() if row.split != "gap"}
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}
    found = {p.relative_to(root).as_posix() for split in ["train", "val", "test"]
             for p in (root / split).rglob("*") if p.is_file() and p.suffix.lower() in extensions}
    if found != expected:
        raise ValueError(f"Image inventory differs: missing={len(expected-found)}, extra={len(found-expected)}")
    try:
        metadata = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
        limits = metadata["limits"]
        if (metadata["schema_version"] != 2 or metadata["fit_split"] != "train"
                or metadata["resampled"] is not False
                or metadata["preprocessing_identity"] != identity
                or preprocessing_identity(metadata["config"]) != identity
                or metadata["manifest_semantic_sha256"] != reference["semantic_sha256"]
                or metadata["reference_manifest_sha256"] != reference["manifest_sha256"]):
            raise ValueError("Contradictory preprocessing metadata")
        if (not isinstance(limits, list) or len(limits) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in limits)
                or not np.isfinite(limits).all() or limits[0] >= limits[1]
                or limits != reference["limits"]):
            raise ValueError("Invalid or nonhistorical train-fitted limits")
        generated_manifest = pd.read_csv(root / "manifest.csv")
        generated_manifest["file_id"] = generated_manifest["file_id"].map(
            lambda x: int(str(x).removesuffix(".mat")))
        validate_manifest(generated_manifest, config)
        if semantic_digest(generated_manifest) != reference["semantic_sha256"]:
            raise ValueError("Generated manifest differs")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Missing or malformed preprocessing metadata/manifest; regenerate explicitly") from exc
    frames = 1 + (config["segment_length"] - config["nperseg"]) // (config["nperseg"] - config["noverlap"])
    for relative in sorted(expected):
        try:
            with Image.open(root / relative) as image:
                image.load()
                if image.format != "PNG" or image.mode != "L" or image.size != (frames, config["nfft"] // 2 + 1):
                    raise ValueError(f"Incorrect PNG mode or dimensions: {relative}")
                digest = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                if digest != reference["pixel_sha256"].get(relative):
                    raise ValueError(f"Pixel content differs from historical reference: {relative}")
        except OSError as exc:
            raise ValueError(f"Cannot decode image: {relative}") from exc
    return metadata


def capture_random_state(loaders=None):
    """Capture available states using JSON-safe primitives without sampling RNGs."""
    py = random.getstate()
    np_state = np.random.get_state()
    return {"schema_version": 1,
            "python": [py[0], list(py[1]), py[2]],
            "numpy": [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
            "torch_cpu": torch.get_rng_state().tolist(),
            "torch_cuda": [state.tolist() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            "loader_generators": {name: loader.generator.get_state().tolist()
                                  for name, loader in (loaders or {}).items() if loader.generator is not None}}


def restore_random_state(state, loaders=None):
    """Restore trusted JSON state in a compatible environment; not an epoch resume."""
    py, ns = state["python"], state["numpy"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG state requires the original device count")
        torch.cuda.set_rng_state_all([torch.tensor(x, dtype=torch.uint8) for x in state["torch_cuda"]])
    for name, values in state["loader_generators"].items():
        if loaders is None or loaders[name].generator is None:
            raise ValueError(f"Missing explicit loader generator: {name}")
        loaders[name].generator.set_state(torch.tensor(values, dtype=torch.uint8))


def train_stages(model, loaders, config, device, out):
    """Validate selected stage-one weights before stage two, preserving iterator RNG consumption."""
    train_stage(model, loaders, config, 1, device, out)
    if config["stage2"]:
        boundary, _, _ = evaluate(model, loaders["val"], device, range(len(config["classes"])))
        save_json(out / "stage1_boundary_validation.json", boundary)
        save_json(out / "stage2_entry_rng.json", capture_random_state(loaders))
        train_stage(model, loaders, config, 2, device, out)


def export_evaluation(model, loader, device, out, config, evidence="newly_executed_experiment"):
    metrics, labels, predictions = evaluate(model, loader, device, range(len(config["classes"])))
    metrics.update(evidence_status=evidence, protocol=config["name"])
    save_json(out / "test_metrics.json", metrics)
    if len(labels) != len(loader.dataset.samples):
        raise ValueError("Dataset/prediction alignment differs")
    with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_label", "predicted_label"])
        for (path, _), y, pred in zip(loader.dataset.samples, labels, predictions):
            w.writerow([Path(path).name, y, pred])
    return metrics


def run_training(image_root, output_root, config):
    """Explicitly called only; training is never performed at import or notebook default."""
    validate_config(config)
    out = Path(output_root)
    if out.exists():
        raise FileExistsError(f"Choose a fresh run directory: {out}")
    metadata = validate_image_dataset(image_root, config)
    initial_rng = capture_random_state()
    if config["seed"] is not None:
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
    # Cross-fault-size retains ambient RNG state; historical session state is unavailable.
    loaders = build_loaders(image_root, config)
    if config["name"] in {"stft_a", "cross_fault_size"}:
        # STFT-A and cross-fault-size consume a training batch before model initialization.
        next(iter(loaders["train"]))
    initialization_rng = capture_random_state(loaders)
    model = build_model(config).to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    out.mkdir(parents=True)
    save_json(out / "rng_before_policy.json", initial_rng)
    save_json(out / "rng_before_model.json", initialization_rng)
    save_json(out / "rng_before_training.json", capture_random_state(loaders))
    save_json(out / "config.json", config)
    save_json(out / "preprocessing.json", metadata)
    save_json(out / "class_mapping.json", {str(i): name for i, name in enumerate(config["classes"])})
    save_json(out / "environment.json", environment())
    manifest = ROOT / "data/manifests" / config["manifest"]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT / "src/cwru_bearing").glob("*.py")}
    save_json(out / "provenance.json", {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_sha256": source_hashes, "recording_metadata_sha256": hashlib.sha256((ROOT / "data/recordings.csv").read_bytes()).hexdigest(),
        "rng_note": "CFS inherits ambient RNG state; historical state unavailable" if config["seed"] is None else "Historical seed settings retained"})
    train_stages(model, loaders, config, device, out)
    # A/B original notebooks did not publish test results; final protocols evaluate here.
    if config["name"] in {"cross_load", "cross_fault_size"}:
        return export_evaluation(model, loaders["test"], device, out, config)
    return {"status": "training_complete", "test_evaluation": "not performed for historical A/B workflow"}


def evaluate_checkpoint(checkpoint_path, image_root, output_root):
    out = Path(output_root)
    if out.exists():
        raise FileExistsError("Evaluation output must be a new directory")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ckpt["config"]
    validate_image_dataset(image_root, config)
    # Inference construction should not reset the caller's RNG.
    with torch.random.fork_rng():
        model = build_model(config, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loaders = build_loaders(image_root, config)
    out.mkdir(parents=True)
    save_json(out / "environment.json", environment())
    save_json(out / "checkpoint.json", {"sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "best_epoch": ckpt["best_epoch"], "stage": ckpt["stage"]})
    return export_evaluation(model, loaders["test"], device, out, config, "evaluation_of_new_checkpoint")
