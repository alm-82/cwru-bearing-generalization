"""Synthetic and original-formula regressions for exploratory signal processing."""
import contextlib
import io
import json
import unittest

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from cwru_bearing.data import ROOT
from cwru_bearing.exploration import (
    fft_spectrum, bandpass_signal, hilbert_envelope, envelope_spectrum,
    bearing_frequencies, peak_near_frequency, harmonic_score,
    band_spectral_energy, harmonic_band_score, fault_scores,
    frequency_rule, shaft_spaced_pairs,
)


class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads((ROOT / "results/historical/signal_exploration_evidence.json").read_text(encoding="utf-8"))
        # These selected archived definitions contain only numerical functions/imports.
        # Execute them in a separate namespace to compare against original source,
        # rather than repeating the maintained implementation as the test oracle.
        cls.original = {"np": np, "butter": butter, "filtfilt": filtfilt, "hilbert": hilbert}
        for cell in [108, 112, 117, 118, 122, 123, 124]:
            exec(compile(cls.evidence["source_code"][str(cell)], f"original_E{cell}", "exec"), cls.original)

    def test_raw_and_hann_fft_original_expressions(self):
        signal = 2 + np.sin(np.arange(1200) * .19)
        ns = {"de_signal": signal, "fs": 12000, "np": np}
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in [8, 9, 13, 14]:
                exec(self.evidence["source_code"][str(cell)], ns)
        f, raw = fft_spectrum(signal, windowed=False)
        _, hann = fft_spectrum(signal)
        np.testing.assert_array_equal(f, ns["frequencies"])
        np.testing.assert_array_equal(raw, ns["magnitude"])
        np.testing.assert_array_equal(hann, ns["amplitude"])

    def test_sinusoid_frequency_and_amplitude(self):
        fs = 12000
        signal = 2.5 * np.cos(2 * np.pi * 600 * np.arange(fs) / fs)
        f, raw = fft_spectrum(signal, fs, False)
        _, windowed = fft_spectrum(signal, fs)
        self.assertEqual(f[np.argmax(raw)], 600)
        self.assertAlmostEqual(raw.max(), 1.25, places=10)
        self.assertAlmostEqual(windowed.max(), 2.5, places=7)

    def test_historical_endpoint_doubling_is_preserved(self):
        signal = (-1.0) ** np.arange(1200)
        _, amplitude = fft_spectrum(signal)
        self.assertAlmostEqual(amplitude[-1], 2.0, places=12)

    def test_filter_and_envelope_match_original_variants(self):
        signal = np.random.default_rng(14).normal(size=6000) + .3
        f, amplitude = envelope_spectrum(signal)
        old_f, old_amplitude = self.original["compute_envelope_spectrum"](signal)
        np.testing.assert_array_equal(f, old_f)
        np.testing.assert_array_equal(amplitude, old_amplitude)
        ns = {"fault_de_signal": signal, "fs": 12000, "np": np}
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in [34, 35]:
                exec(self.evidence["source_code"][str(cell)], ns)
        filtered = bandpass_signal(signal, remove_mean=False)
        np.testing.assert_array_equal(filtered, ns["fault_band"])
        np.testing.assert_array_equal(hilbert_envelope(filtered), ns["fault_envelope"])
        _, early = envelope_spectrum(signal, remove_signal_mean=False)
        np.testing.assert_array_equal(early, ns["env_amplitude"])

    def test_am_modulation_recovery(self):
        fs = 12000
        t = np.arange(2 * fs) / fs
        expected = 1 + .5 * np.cos(2 * np.pi * 120 * t)
        signal = expected * np.cos(2 * np.pi * 3000 * t)
        envelope = hilbert_envelope(bandpass_signal(signal))
        np.testing.assert_allclose(envelope[1000:-1000], expected[1000:-1000], atol=.003)
        f, amplitude = envelope_spectrum(signal)
        region = (f >= 50) & (f <= 500)
        self.assertEqual(f[region][np.argmax(amplitude[region])], 120)
        self.assertAlmostEqual(amplitude[region].max(), .5, places=3)

    def test_characteristic_frequency_original_coefficients(self):
        ns = {"np": np}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(self.evidence["source_code"]["17"], ns)
        values = bearing_frequencies(1796)
        for name in ["BPFI", "BPFO", "BSF", "FTF"]:
            self.assertEqual(values[name], ns[name.lower()])
        fundamental = 1.537 / (2 * .3126) * (1 - (.3126 / 1.537) ** 2)
        self.assertAlmostEqual(2 * fundamental, 4.7135, delta=1e-4)  # Rounded published geometry.
        self.assertEqual(values["BSF"], 4.7135 * (1796 / 60))

    def test_peak_and_band_rules_match_original_source(self):
        f = np.arange(0, 601, .25)
        amplitude = np.random.default_rng(19).uniform(0, .1, len(f))
        for rpm in [1721, 1796, 1797]:
            self.assertEqual(fault_scores(f, amplitude, rpm), self.original["calculate_fault_scores"](f, amplitude, rpm))
            self.assertEqual(fault_scores(f, amplitude, rpm, "band"), self.original["calculate_normalized_fault_scores"](f, amplitude, rpm))
        self.assertEqual(harmonic_score(f, amplitude, 120), self.original["harmonic_score"](f, amplitude, 120))
        self.assertEqual(band_spectral_energy(f, amplitude, 120), self.original["band_spectral_energy"](f, amplitude, 120))
        self.assertEqual(harmonic_band_score(f, amplitude, 120), self.original["harmonic_band_score"](f, amplitude, 120))

    def test_inclusive_bands_empty_bands_and_ties(self):
        f, amplitude = np.arange(601), np.zeros(601)
        amplitude[118] = amplitude[122] = .5
        self.assertEqual(peak_near_frequency(f, amplitude, 120), (118, .5))
        self.assertEqual(band_spectral_energy(f, amplitude, 120, 2), .5)
        self.assertEqual(band_spectral_energy(f, amplitude, 900), 0)
        with self.assertRaises(ValueError):
            peak_near_frequency(f, amplitude, 900)
        for method in ["peak", "band"]:
            self.assertEqual(frequency_rule(f, np.zeros_like(amplitude), 1796, method)[0], "InnerRace")

    def test_shaft_pair_search_matches_original_loop(self):
        peaks = np.array([50., 79.5, 109.5, 140., 200.])
        ns = {"strong_peaks_df": {"frequency": type("Column", (), {"values": peaks})()},
              "shaft_freq_197": 1796 / 60, "pd": type("Frame", (), {"DataFrame": lambda value: value})}
        exec(self.evidence["source_code"]["139"], ns)
        self.assertEqual(shaft_spaced_pairs(peaks, 1796), ns["matching_pairs"])

    def test_bad_inputs_fail_without_artifacts(self):
        for signal in [[], [1, float("nan"), 2], [[1, 2, 3]]]:
            with self.assertRaises(ValueError):
                fft_spectrum(signal)
        with self.assertRaises(ValueError):
            bandpass_signal(np.ones(100), lowcut=4000, highcut=2000)
        with self.assertRaises(ValueError):
            bearing_frequencies(0)
        with self.assertRaises(ValueError):
            harmonic_score(np.arange(601), np.ones(601), 120, harmonics=0)

    def test_saved_rule_summary_is_explicitly_exploratory(self):
        saved = self.evidence["band_rule_observation"]
        self.assertEqual(saved["accuracy"], saved["correct_recordings"] / saved["total_recordings"])
        self.assertIn(str(saved["accuracy"]), self.evidence["outputs"]["128"])
        self.assertIn("not independent validation", saved["status"])


if __name__ == "__main__":
    unittest.main()
