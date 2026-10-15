"""Configuration, publication boundaries, and historical execution-order regressions."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from cwru_bearing.data import ROOT, load_config, validate_config
from cwru_bearing.metrics import classification_metrics

spec = importlib.util.spec_from_file_location("publication_export", ROOT / "scripts/prepare_publication.py")
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


class ConfigurationTests(unittest.TestCase):
    def test_all_historical_profiles_are_valid(self):
        for name in ["stft_a", "stft_b", "cross_load", "cross_fault_size"]:
            validate_config(load_config(name))

    def test_invalid_settings_fail_clearly(self):
        base = load_config("cross_fault_size")
        changes = [("transform_order", "typo"), ("batch_size", 0), ("batch_size", True),
                   ("nperseg", 5000), ("noverlap", 512), ("noverlap", -1),
                   ("nfft", 128), ("window", "unknown"), ("padded", True),
                   ("processing_rate_hz", 0), ("epsilon", float("nan")),
                   ("percentiles", [99, 1]), ("num_workers", -1)]
        for field, value in changes:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_config({**base, field: value})
        for stage in ["stage1", "stage2"]:
            for epochs in [0, -1, 1.5, True, None]:
                changed = copy.deepcopy(base)
                changed[stage]["epochs"] = epochs
                with self.subTest(stage=stage, epochs=epochs), self.assertRaises(ValueError):
                    validate_config(changed)
        with self.assertRaises(ValueError):
            validate_config({})
        for name in [[], {}, None, 42]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_config(name)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_config({**base, "name": name})

    def test_confusion_order_includes_absent_classes(self):
        metrics = classification_metrics([0, 0], [0, 0], range(4))
        self.assertEqual(metrics["confusion_matrix"], [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(metrics["macro_f1"], .25)
        with self.assertRaises(ValueError):
            classification_metrics([4], [4], range(4))

    def test_classical_features_and_train_only_scaler(self):
        from cwru_bearing.classical import windows, features, svm_pipeline
        signal = np.sin(np.arange(6144) * .05)
        self.assertEqual(windows(signal).shape, (3, 2048))
        values = np.array([features(w) for w in windows(signal)])
        self.assertTrue(np.isfinite(values).all())
        train = np.array([[1., 2.], [2., 3.], [10., 11.], [11., 12.]])
        model = svm_pipeline().fit(train, [0, 0, 1, 1])
        model.predict(np.array([[1000., 2000.]]))
        np.testing.assert_array_equal(model["scaler"].mean_, train.mean(axis=0))

    def test_classical_protocol_record_selection(self):
        from cwru_bearing.classical import run_baselines
        from cwru_bearing.data import load_manifest
        config = load_config("cross_load")
        records = load_manifest(config).drop_duplicates("file_id")
        expected = set(records.loc[(records.class_label == 0) | records.fault_size_inch.isin([.007, .014]), "file_id"])
        seen = []
        def signal(root, file_id):
            seen.append(int(file_id))
            return (1 + int(file_id) / 100) * np.sin(np.arange(4096) * .05)
        with patch("cwru_bearing.data.validate_recordings") as validate, \
             patch("cwru_bearing.data.load_signal", side_effect=signal):
            results = run_baselines("synthetic")
        self.assertEqual(set(seen), expected)
        self.assertEqual(len(seen), 28)
        self.assertEqual(set(validate.call_args.kwargs["file_ids"]), expected)
        self.assertEqual(results["fixed_size_cross_load_four_features"]["train_windows"], 24)
        self.assertEqual(results["fixed_size_cross_load_four_features"]["test_windows"], 8)
        self.assertEqual(results["fault_only_size_transfer_four_features"]["test_windows"], 24)
        self.assertEqual(results["joint_load_size_transfer_four_features"]["train_windows"], 24)
        self.assertEqual(results["joint_load_size_transfer_four_features"]["test_windows"], 8)
        self.assertEqual(results["joint_load_size_transfer_rms_only"]["test_windows"], 8)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "README.md").write_text("Public research description", encoding="utf-8")
        self.approved = patch.object(publication, "PUBLIC_FILES", ("README.md",))
        self.approved.start()
        self.addCleanup(self.approved.stop)

    def test_unapproved_source_files_are_never_exported(self):
        for relative in ["notebooks/new.ipynb", "docs/internal_report.md", ".git/config", "data/unreviewed.json"]:
            p = self.source / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("Internal", encoding="utf-8")
        target = self.root / "candidate"
        publication.export_candidate(target, self.source)
        self.assertEqual(set(p.name for p in target.iterdir()), {"README.md"})
        self.assertFalse((target / ".git").exists())
        (target / "unapproved.txt").write_text("Unexpected", encoding="utf-8")
        with self.assertRaises(ValueError):
            publication.verify_candidate(target)

    def test_existing_target_is_never_overwritten(self):
        target = self.root / "candidate"
        target.mkdir()
        marker = target / "keep.txt"
        marker.write_text("Preserve", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            publication.export_candidate(target, self.source)
        self.assertEqual(marker.read_text(encoding="utf-8"), "Preserve")

    def test_escaping_and_missing_paths_are_rejected(self):
        for name in ["../outside.md", "/outside.md", "C:/outside.md", ".git/config"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                publication.public_files(self.source, [name])
        with self.assertRaises(FileNotFoundError):
            publication.public_files(self.source, ["missing.md"])
        original = Path.resolve
        def escaped(path, *args, **kwargs):
            if path == self.source / "README.md":
                return self.root / "outside.md"
            return original(path, *args, **kwargs)
        with patch.object(Path, "resolve", autospec=True, side_effect=escaped):
            with self.assertRaises(ValueError):
                publication.public_files(self.source)

    def test_private_and_malformed_content_is_rejected(self):
        path = self.source / "README.md"
        for text in ["gh" + "p_" + "x" * 36, chr(0xfffd), "C:" + chr(92) + "Users" + chr(92) + "example" + chr(92) + "private"]:
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                publication.scan([path])
        notebook = self.source / "notebook.ipynb"
        notebook.write_text('{"cells":[{"outputs":[{"output_type":"error"}]}]}', encoding="utf-8")
        with self.assertRaises(ValueError):
            publication.scan([notebook])


class NotebookRunnerTests(unittest.TestCase):
    def test_timeout_retains_diagnostics_when_cleanup_also_times_out(self):
        import subprocess
        from unittest.mock import MagicMock
        spec = importlib.util.spec_from_file_location("notebook_runner", ROOT / "scripts/execute_notebooks.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        child = MagicMock()
        child.pid = 123456
        child.communicate.side_effect = [
            subprocess.TimeoutExpired("fixture", 180, output=b"kernel start"),
            subprocess.TimeoutExpired("fixture", 10, output=b"kernel start", stderr=b"fixture timeout"),
        ]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner, "WORK", Path(directory) / "execution"), \
             patch.object(runner.subprocess, "Popen", return_value=child), \
             patch.object(runner.subprocess, "run"), \
             patch.object(runner.os, "killpg", create=True):
            self.assertEqual(runner.main(), 1)
            import json
            rows = json.loads((Path(directory) / "execution/execution.json").read_text(encoding="utf-8"))
        self.assertEqual(rows[0]["stdout"], "kernel start")
        self.assertFalse(rows[0]["passed"])
        child.kill.assert_called_once()


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch optional dependency unavailable")
class ExecutionOrderTests(unittest.TestCase):
    def test_cfs_sanity_batch_precedes_model_without_reseeding(self):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from cwru_bearing.training import run_training
        config = load_config("cross_fault_size")
        loader = DataLoader(TensorDataset(torch.zeros(60, 1), torch.arange(60) % 4), batch_size=8, shuffle=True)
        loaders = {name: loader for name in ["train", "val", "test"]}
        torch.manual_seed(731)
        initial = torch.get_rng_state().clone()
        next(iter(loader))
        expected = torch.get_rng_state().clone()
        self.assertFalse(torch.equal(initial, expected))
        torch.set_rng_state(initial)
        events = []
        def construct(*args, **kwargs):
            events.append("model")
            self.assertTrue(torch.equal(torch.get_rng_state(), expected))
            return nn.Linear(1, 4)
        def stages(*args):
            events.append("stages")
        with tempfile.TemporaryDirectory() as directory, \
             patch("cwru_bearing.training.validate_image_dataset", return_value={}), \
             patch("cwru_bearing.training.build_loaders", return_value=loaders), \
             patch("cwru_bearing.training.build_model", side_effect=construct), \
             patch("cwru_bearing.training.train_stages", side_effect=stages), \
             patch("cwru_bearing.training.export_evaluation", return_value={}), \
             patch("cwru_bearing.training.environment", return_value={}):
            run_training("unused", Path(directory) / "run", config)
        self.assertEqual(events, ["model", "stages"])
        self.assertIsNone(config["seed"])


if __name__ == "__main__":
    unittest.main()
