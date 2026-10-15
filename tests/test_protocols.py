import copy
import importlib.util
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path
import numpy as np
from scipy.io import savemat
from cwru_bearing.data import ROOT, load_config, load_manifest, load_signal, segment, validate_config
from cwru_bearing.preprocessing import spectrogram_db, fit_limits, quantize
from cwru_bearing.metrics import classification_metrics, is_better


class ProtocolTests(unittest.TestCase):
    def test_exact_array_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            savemat(Path(directory) / "99.mat", {"X098_DE_time": np.array([99., 99.]),
                                                "X099_DE_time": np.array([1., 2.])})
            np.testing.assert_array_equal(load_signal(directory, 99), [1., 2.])
            savemat(Path(directory) / "99.mat", {"X098_DE_time": np.array([1.])})
            with self.assertRaises(KeyError):
                load_signal(directory, 99)

    def test_manifests(self):
        for name, counts in [("cross_load", {"train":580,"val":290,"test":290}),
                             ("cross_fault_size", {"train":60,"val":60,"test":60}),
                             ("stft_a", {"train":200,"val":40,"test":30,"gap":20})]:
            df = load_manifest(load_config(name))
            self.assertEqual(df.split.value_counts().to_dict(), counts)
        df = load_manifest(load_config("cross_fault_size"))
        self.assertEqual(df.loc[df.class_label == 0].file_id.unique().tolist(), [97])
        self.assertEqual(df.loc[(df.class_label==0)&(df.split=="val")].segment_idx.tolist(), list(range(16,31)))

    def test_segmentation(self):
        x = np.arange(8192)
        a, b = segment(x, 0, 4096), segment(x, 4096, 8192)
        self.assertEqual(len(np.intersect1d(a, b)), 0)
        for bounds in [(-1,4095),(4096,8193),(0,4000)]:
            with self.assertRaises(ValueError):
                segment(x,*bounds)

    def test_stft_dimensions_and_rate_label(self):
        x = np.sin(np.arange(4096) * .3)
        a, b = load_config("stft_a"), load_config("stft_b")
        self.assertEqual(spectrogram_db(x,a).shape,(257,15))
        self.assertEqual(spectrogram_db(x,b).shape,(257,57))
        np.testing.assert_array_equal(spectrogram_db(x,b),spectrogram_db(x,{**b,"processing_rate_hz":48000}))

    def test_train_only_statistics(self):
        import pandas as pd
        df = pd.DataFrame({"split":["train","val","test"]})
        specs = {0:np.arange(100.),1:np.full(10,1e6),2:np.full(10,-1e6)}
        np.testing.assert_allclose(fit_limits(specs,df,load_config("stft_b")),[.495,98.505])

    def test_scalar_percentile_precision(self):
        import pandas as pd
        values=np.linspace(-120,-10,1003,dtype=np.float32)
        lo,hi=fit_limits({0:values},pd.DataFrame({'split':['train']}),load_config('cross_load'))
        self.assertEqual(lo.dtype, np.percentile(values,.5).dtype)
        self.assertEqual(hi, np.percentile(values,99.5))

    def test_quantization_and_orientation(self):
        x=np.array([[-1,0.5/255],[1.5/255,2.]])
        c=load_config("cross_load")
        np.testing.assert_array_equal(quantize(x,(0,1),c),[[2,255],[0,0]])
        np.testing.assert_array_equal(quantize(x,(0,1),{**c,"quantization":"truncate"}),[[1,255],[0,0]])

    def test_checkpoint_selection(self):
        self.assertTrue(is_better(.5,2,.4,1,"exact"))
        self.assertFalse(is_better(.5,2,.5,1,"exact"))
        self.assertFalse(is_better(.5-1e-7,.5,.5,1,"exact"))
        self.assertTrue(is_better(.5-1e-7,.5,.5,1,"isclose"))

    def test_classical_saved_evidence_and_tamper_detection(self):
        import copy
        import importlib.util
        import json
        from unittest.mock import patch
        from cwru_bearing.data import ROOT
        spec = importlib.util.spec_from_file_location("repository_checks", ROOT / "scripts/verify_repository.py")
        import sys
        with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
            checks = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(checks)
        self.assertEqual(checks.check_classical_evidence(), 6)
        evidence = json.loads((ROOT / "results/historical/classical_svm_evidence.json").read_text(encoding="utf-8"))
        tampered = copy.deepcopy(evidence)
        tampered["observations"][-1]["accuracy"] = 1.0
        original_loads = json.loads
        def altered_loads(text, *args, **kwargs):
            value = original_loads(text, *args, **kwargs)
            return tampered if isinstance(value, dict) and "observations" in value else value
        with patch.object(checks.json, "loads", side_effect=altered_loads):
            with self.assertRaises(ValueError):
                checks.check_classical_evidence()

    def test_confusion_metrics(self):
        import json
        from cwru_bearing.data import ROOT
        results=json.loads((ROOT/'results/historical/results.json').read_text())
        for key in ['cross_load','cross_fault_size']:
            y,p=[],[]
            for i,row in enumerate(results[key]['confusion_matrix']):
                for j,count in enumerate(row):
                    y.extend([i]*count);p.extend([j]*count)
            metrics=classification_metrics(y,p)
            self.assertAlmostEqual(metrics['accuracy'],results[key]['accuracy'])
            self.assertAlmostEqual(metrics['macro_f1'],results[key]['macro_f1'],places=8)


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


class RepositoryVerificationTests(unittest.TestCase):
    def setUp(self):
        import json
        self.json = json
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.readme = self.root / "README.md"
        self.readme.write_text("Public research description", encoding="utf-8")
        spec = importlib.util.spec_from_file_location("repository_inventory_checks", ROOT / "scripts/verify_repository.py")
        self.checks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.checks)
        self.names = ["README.md"]
        self.untracked = []

    def git(self, root, *arguments):
        self.assertEqual(Path(root), self.root)
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(self.root).encode("utf-8") + b"\n"
        if arguments == ("ls-files", "--cached", "-z"):
            return ("\0".join(self.names) + "\0").encode("utf-8")
        if arguments == ("ls-files", "--others", "--exclude-standard", "-z"):
            return ("\0".join(self.untracked) + "\0").encode("utf-8")
        self.fail(f"Unexpected Git inventory command: {arguments}")

    def test_inventory_comes_from_git_including_spaced_names(self):
        guide = self.root / "guide with spaces.md"
        guide.write_text("Guide", encoding="utf-8")
        self.names.append(guide.name)
        (self.root / "untracked.txt").write_text("Not part of the index", encoding="utf-8")
        with patch.object(self.checks, "git_output", side_effect=self.git):
            self.assertEqual(self.checks.tracked_files(self.root), [self.readme, guide])

    def test_strict_inventory_rejects_untracked_but_excludes_ignored_metadata(self):
        (self.root / ".git").mkdir()
        (self.root / ".git/config").write_text("Git metadata", encoding="utf-8")
        (self.root / "work").mkdir()
        (self.root / "work/runtime.txt").write_text("Ignored runtime artifact", encoding="utf-8")
        with patch.object(self.checks, "git_output", side_effect=self.git):
            self.assertEqual(self.checks.verify_inventory(self.root), [self.readme])
            self.untracked = ["new.txt"]
            with self.assertRaisesRegex(ValueError, "Untracked repository files"):
                self.checks.verify_inventory(self.root)

    def test_missing_index_tracked_file_is_rejected(self):
        self.names.append("missing.md")
        with patch.object(self.checks, "git_output", side_effect=self.git):
            with self.assertRaisesRegex(FileNotFoundError, "Missing tracked file"):
                self.checks.verify_inventory(self.root)

    def test_escaping_duplicate_and_empty_inventory_is_rejected(self):
        with patch.object(self.checks, "git_output", side_effect=self.git):
            for names in [[], ["README.md", "README.md"], ["../outside.md"],
                          ["/outside.md"], ["C:/outside.md"], [".git/config"]]:
                self.names = names
                with self.subTest(names=names), self.assertRaises(ValueError):
                    self.checks.tracked_files(self.root)
            self.names = ["README.md"]
            original = Path.resolve
            def escaped(path, *args, **kwargs):
                if path == self.readme:
                    return self.root.parent / "outside.md"
                return original(path, *args, **kwargs)
            with patch.object(Path, "resolve", autospec=True, side_effect=escaped):
                with self.assertRaisesRegex(ValueError, "escapes repository"):
                    self.checks.tracked_files(self.root)

    def test_sensitive_decoded_notebook_and_error_outputs_are_rejected(self):
        for content in ["gh" + "p_" + "x" * 36, chr(0xfffd),
                        "C:" + chr(92) + "Users" + chr(92) + "example" + chr(92) + "private"]:
            self.readme.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.checks.scan([self.readme])
        notebook = self.root / "notebook.ipynb"
        notebook.write_text(self.json.dumps({"cells": [{"source": "gh" + "p_" + "x" * 36}]}), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.checks.scan([notebook])
        notebook.write_text(self.json.dumps({"cells": [{"outputs": [{"output_type": "error"}]}]}), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.checks.scan([notebook])

    def test_png_metadata_is_scanned(self):
        from PIL import Image, PngImagePlugin
        image = self.root / "figure.png"
        Image.new("L", (2, 2)).save(image)
        self.checks.scan([image])
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("comment", "gh" + "p_" + "x" * 36)
        Image.new("L", (2, 2)).save(image, pnginfo=metadata)
        with self.assertRaisesRegex(ValueError, "Private metadata"):
            self.checks.scan([image])

    def test_link_checks_use_only_tracked_documents_and_notebooks(self):
        guide = self.root / "guide with spaces.md"
        guide.write_text("Guide", encoding="utf-8")
        self.readme.write_text("[Guide](guide%20with%20spaces.md#section)", encoding="utf-8")
        (self.root / "untracked.ipynb").write_text("Not JSON", encoding="utf-8")
        notebook = self.root / "tracked.ipynb"
        notebook.write_text(self.json.dumps({"cells": [{"cell_type": "markdown", "source": ["[Home](README.md)"]}]}), encoding="utf-8")
        with patch.object(self.checks, "ROOT", self.root):
            self.assertEqual(self.checks.check_links([self.readme, guide, notebook]), 2)
            self.readme.write_text("[Missing](missing.md)", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Broken local link"):
                self.checks.check_links([self.readme])

    def test_git_failure_has_no_hardcoded_fallback(self):
        import subprocess
        failure = subprocess.CompletedProcess(["git"], 128, b"", b"fatal: not a repository")
        with patch.object(self.checks.subprocess, "run", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "Git inventory failed"):
                self.checks.tracked_files(self.root)


if __name__ == '__main__':
    unittest.main()
