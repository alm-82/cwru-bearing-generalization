import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.io import savemat
from cwru_bearing.data import load_config, load_manifest, load_signal, segment
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


if __name__ == '__main__':
    unittest.main()
