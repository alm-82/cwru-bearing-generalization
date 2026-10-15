"""Regression fixtures exercise engineering behavior, never CWRU performance."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch optional dependency unavailable")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cwru_bearing.data import load_config, load_manifest, validate_manifest, historical_reference
from cwru_bearing.preprocessing import image_relative_path, preprocessing_identity
from cwru_bearing.training import (capture_random_state, restore_random_state,
    validate_image_dataset, train_stages, run_training, evaluate_checkpoint)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))
        self.classifier = nn.Sequential(nn.Dropout(.2), nn.Linear(3, 4))

    def forward(self, x):
        return self.classifier(self.features(x))


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config("cross_fault_size")

    def make_images(self):
        """Valid synthetic PNGs, with a test-only independent content reference."""
        df = load_manifest(self.config)
        ref = copy.deepcopy(historical_reference(self.config))
        ref["pixel_sha256"] = {}
        for _, row in df.iterrows():
            path = self.root / image_relative_path(row, self.config)
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((257, 57), 20 + int(row.class_label) * 50, np.uint8)
            Image.fromarray(pixels).save(path)
            ref["pixel_sha256"][path.relative_to(self.root).as_posix()] = hashlib.sha256(pixels.tobytes()).hexdigest()
        df.to_csv(self.root / "manifest.csv", index=False)
        meta = {"schema_version": 2, "config": self.config,
                "preprocessing_identity": preprocessing_identity(self.config),
                "limits": ref["limits"], "fit_split": "train", "resampled": False,
                "manifest_semantic_sha256": ref["semantic_sha256"],
                "reference_manifest_sha256": ref["manifest_sha256"]}
        (self.root / "preprocessing.json").write_text(json.dumps(meta))
        return ref, meta

    def test_image_and_metadata_mutations(self):
        ref, meta = self.make_images()
        first = self.root / next(iter(ref["pixel_sha256"]))
        original = first.read_bytes()
        with patch("cwru_bearing.data.historical_reference", return_value=ref):
            validate_image_dataset(self.root, self.config)
            for name, mutate in [
                ("corrupt", lambda: first.write_bytes(b"not a PNG")),
                ("dimensions", lambda: Image.new("L", (10, 10)).save(first)),
                ("mode", lambda: Image.new("RGB", (57, 257)).save(first)),
                ("pixels", lambda: Image.new("L", (57, 257), 251).save(first))]:
                with self.subTest(name=name):
                    mutate()
                    with self.assertRaises(ValueError): validate_image_dataset(self.root, self.config)
                    first.write_bytes(original)
            for field, value in [("fit_split", "test"), ("resampled", True),
                    ("limits", [float("nan"), -999]), ("limits", [1, 0]),
                    ("limits", [False, 0]), ("schema_version", 1),
                    ("manifest_semantic_sha256", "tampered")]:
                with self.subTest(field=field, value=value):
                    altered = {**meta, field: value}
                    (self.root / "preprocessing.json").write_text(json.dumps(altered))
                    with self.assertRaises(ValueError): validate_image_dataset(self.root, self.config)
            (self.root / "preprocessing.json").write_text(json.dumps(meta))
            extra = first.with_suffix(".jpeg")
            extra.write_bytes(original)
            with self.assertRaises(ValueError): validate_image_dataset(self.root, self.config)
            extra.unlink()
            (self.root / "preprocessing.json").write_text("{")
            with self.assertRaises(ValueError): validate_image_dataset(self.root, self.config)

    def test_runtime_identity_and_preprocessing_rejection(self):
        ref, _ = self.make_images()
        with patch("cwru_bearing.data.historical_reference", return_value=ref):
            runtime = {**self.config, "num_workers": 2, "batch_size": 3, "seed": 12}
            validate_image_dataset(self.root, runtime)
            for field, value in [("noverlap", 256), ("quantization", "round"),
                                 ("percentiles", [1, 99]), ("processing_rate_hz", 48000)]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_image_dataset(self.root, {**self.config, field: value})

    def test_independent_manifest_mutations(self):
        for profile in ["stft_a", "cross_load", "cross_fault_size"]:
            config = load_config(profile)
            df = load_manifest(config)
            for field, value in [("file_id", 999), ("class_label", 2),
                    ("fault_type", "invented"), ("fault_size_inch", .099),
                    ("load_hp", 3), ("split", "test"), ("sample_end", 8192)]:
                changed = df.copy()
                changed.loc[0, field] = value
                with self.subTest(profile=profile, field=field), self.assertRaises(ValueError):
                    validate_manifest(changed, config)
            with self.assertRaises(ValueError):
                validate_manifest(df, {**config, "classes": list(reversed(config["classes"]))})

    def test_manifest_file_cannot_validate_against_itself(self):
        import shutil
        from cwru_bearing.data import ROOT
        local = self.root / "isolated_reference"
        (local / "data/manifests").mkdir(parents=True)
        shutil.copy2(ROOT / "data/historical_reference.json", local / "data/historical_reference.json")
        source = ROOT / "data/manifests" / self.config["manifest"]
        path = local / "data/manifests" / self.config["manifest"]
        text = source.read_text()
        path.write_text(text.replace("0.007", "0.099", 1))
        with patch("cwru_bearing.data.ROOT", local), self.assertRaises(ValueError):
            load_manifest(self.config)

    def test_rng_capture_is_passive_and_json_roundtrip(self):
        gen = torch.Generator().manual_seed(13)
        loader = DataLoader(TensorDataset(torch.zeros(2, 1)), generator=gen)
        loaders = {"train": loader}
        state = capture_random_state(loaders)
        self.assertEqual(state, capture_random_state(loaders))
        expected = [random.random(), float(np.random.random()), float(torch.rand(())),
                    float(torch.rand((), generator=gen))]
        restore_random_state(json.loads(json.dumps(state)), loaders)
        actual = [random.random(), float(np.random.random()), float(torch.rand(())),
                  float(torch.rand((), generator=gen))]
        self.assertEqual(expected, actual)

    def test_stage_boundary_restoration_and_iterator_rng(self):
        loader = DataLoader(TensorDataset(torch.zeros(4, 3, 2, 2), torch.arange(4)), batch_size=4)
        model = TinyModel()
        expected = {k: value.clone() for k, value in model.state_dict().items()}
        events = []
        # Force the selected epoch to be first, while the second epoch updates weights.
        calls = 0
        from cwru_bearing.training import evaluate as real_evaluate, train_stage as real_train_stage
        def ranked_evaluate(*args):
            nonlocal calls
            metrics, labels, preds = real_evaluate(*args)
            calls += 1
            metrics["macro_f1"] = 1.0 if calls == 1 else 0.0
            return metrics, labels, preds
        config = copy.deepcopy(self.config)
        config["stage1"]["epochs"] = 2
        config["stage2"]["epochs"] = 1
        def stage(model, loaders, config, number, device, out):
            if number == 1:
                with patch("cwru_bearing.training.evaluate", side_effect=ranked_evaluate):
                    real_train_stage(model, loaders, config, number, device, out)
                ckpt = torch.load(out / "stage1_best.pth", weights_only=True)
                self.assertEqual(ckpt["best_epoch"], 1)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, ckpt["model_state"][key]))
                events.append("restored")
                expected.update({k: v.clone() for k,v in model.state_dict().items()})
                state = torch.get_rng_state()
                next(iter(loader))
                expected["rng"] = torch.get_rng_state()
                self.assertFalse(torch.equal(state, expected["rng"]))
                torch.set_rng_state(state)
            else:
                events.append("stage2")
                self.assertTrue(torch.equal(torch.get_rng_state(), expected["rng"]))
                self.assertFalse(model.training)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, expected[key]))
                real_train_stage(model, loaders, config, number, device, out)
        with patch("cwru_bearing.training.train_stage", side_effect=stage):
            train_stages(model, {"train": loader, "val": loader}, config, "cpu", self.root)
        self.assertEqual(events, ["restored", "stage2"])
        self.assertTrue((self.root / "stage1_boundary_validation.json").exists())

    def test_offline_training_checkpoint_evaluation_roundtrip(self):
        ref, _ = self.make_images()
        config = copy.deepcopy(self.config)
        config["image_size"] = 8
        config["stage1"]["epochs"] = config["stage2"]["epochs"] = 1
        config["batch_size"] = 60
        def tiny(config, pretrained=True):
            return TinyModel()
        with patch("cwru_bearing.data.historical_reference", return_value=ref), \
             patch("cwru_bearing.training.build_model", side_effect=tiny):
            metrics = run_training(self.root, self.root / "run", config)
            second = evaluate_checkpoint(self.root / "run/stage2_best.pth", self.root, self.root / "evaluation")
        self.assertEqual(metrics["confusion_matrix"], second["confusion_matrix"])
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 60)
        self.assertEqual((self.root / "run/test_predictions.csv").read_text(),
                         (self.root / "evaluation/test_predictions.csv").read_text())
        self.assertTrue((self.root / "run/rng_before_model.json").is_file())
        self.assertTrue((self.root / "run/stage1_best.pth").is_file())
        self.assertTrue((self.root / "run/stage2_best.pth").is_file())


if __name__ == "__main__":
    unittest.main()
