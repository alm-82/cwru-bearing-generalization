"""Exploratory spectra and fault-family rules.

These methods inspect signals; their scores are not calibrated probabilities or a
validated diagnostic model. Sampling-rate arguments describe processing labels.
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert


# Retain the historical names and coefficients, including BSF.
HISTORICAL_RATIOS = {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 4.7135, "FTF": 0.39828}


def _signal(signal):
    values = np.asarray(signal)
    if (values.ndim != 1 or len(values) < 3 or not np.isrealobj(values)
            or not np.isfinite(values).all()):
        raise ValueError("Expected a finite real one-dimensional signal with at least three samples")
    return values


def _positive(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def fft_spectrum(signal, fs=12000, windowed=True):
    """Return historical rFFT bins and amplitudes.

    Raw mode uses abs(rFFT(x))/N without removing the mean. Windowed mode removes
    the mean, applies np.hanning(N), and scales every bin by 2/sum(window).
    Historical doubling of DC/Nyquist bins is retained; this is not a PSD.
    """
    values = _signal(signal)
    _positive(fs, "fs")
    if windowed:
        window = np.hanning(len(values))
        values = (values - np.mean(values)) * window
        amplitude = np.abs(np.fft.rfft(values)) * 2 / np.sum(window)
    else:
        amplitude = np.abs(np.fft.rfft(values)) / len(values)
    return np.fft.rfftfreq(len(values), d=1 / fs), amplitude


def bandpass_signal(signal, fs=12000, lowcut=2000, highcut=4000, remove_mean=True):
    """Use a fourth-order Butterworth b/a bandpass design with filtfilt.

    Optional mean removal preserves both historical variants. The processing rate determines
    normalized cutoffs. Using a wrong rate label changes the physical passband.
    """
    values = _signal(signal)
    _positive(fs, "fs")
    if not (np.isfinite(lowcut) and np.isfinite(highcut) and 0 < lowcut < highcut < fs / 2):
        raise ValueError("Cutoffs must satisfy 0 < lowcut < highcut < fs/2")
    if remove_mean:
        values = values - np.mean(values)
    b, a = butter(4, [lowcut / (fs / 2), highcut / (fs / 2)], btype="band")
    return filtfilt(b, a, values)


def hilbert_envelope(signal):
    """Magnitude of the analytic signal; no implicit filtering."""
    return np.abs(hilbert(_signal(signal)))


def envelope_spectrum(signal, fs=12000, lowcut=2000, highcut=4000, remove_signal_mean=True):
    """Bandpass, Hilbert magnitude, envelope demeaning, and Hann rFFT.

    Set remove_signal_mean=False for the earlier variant without signal demeaning.
    Finite-record filter and Hilbert edge effects remain; no trimming is added.
    """
    filtered = bandpass_signal(signal, fs, lowcut, highcut, remove_signal_mean)
    return fft_spectrum(hilbert_envelope(filtered), fs, windowed=True)


def bearing_frequencies(rpm):
    """Return shaft frequency and historical bearing-family targets in Hz.

    The historical BSF label uses 4.7135 times shaft speed, the CWRU rolling-element
    defect multiplier. It is approximately twice fundamental ball spin under the
    usual zero-contact-angle geometry formula. The coefficient is not corrected.
    """
    _positive(rpm, "rpm")
    shaft = rpm / 60
    return {"shaft": shaft, **{name: ratio * shaft for name, ratio in HISTORICAL_RATIOS.items()}}


def _spectrum(frequencies, amplitude):
    frequencies, amplitude = np.asarray(frequencies), np.asarray(amplitude)
    if (frequencies.ndim != 1 or frequencies.size == 0 or frequencies.shape != amplitude.shape
            or not np.isfinite(frequencies).all() or not np.isfinite(amplitude).all()
            or np.any(np.diff(frequencies) <= 0) or np.any(amplitude < 0)):
        raise ValueError("Expected increasing frequency bins and matching finite nonnegative amplitudes")
    return frequencies, amplitude


def peak_near_frequency(frequencies, amplitude, target_frequency, tolerance=2.0):
    """Return the largest bin in the inclusive target band, without peak fitting."""
    frequencies, amplitude = _spectrum(frequencies, amplitude)
    _positive(target_frequency, "target_frequency")
    _positive(tolerance, "tolerance")
    mask = (frequencies >= target_frequency - tolerance) & (frequencies <= target_frequency + tolerance)
    if not mask.any():
        raise ValueError("No frequency bins fall within the target band")
    local_frequencies, local_amplitude = frequencies[mask], amplitude[mask]
    index = np.argmax(local_amplitude)
    return local_frequencies[index], local_amplitude[index]


def harmonic_score(frequencies, amplitude, base_frequency, harmonics=3, tolerance=2.0):
    """Sum local maxima near integer harmonics and return per-harmonic details."""
    if isinstance(harmonics, bool) or not isinstance(harmonics, int) or harmonics < 1:
        raise ValueError("harmonics must be a positive integer")
    score, details = 0, []
    for harmonic in range(1, harmonics + 1):
        target = harmonic * base_frequency
        measured, value = peak_near_frequency(frequencies, amplitude, target, tolerance)
        score += value
        details.append({"harmonic": harmonic, "target_freq": target,
                        "measured_freq": measured, "amplitude": value})
    return score, details


def band_spectral_energy(frequencies, amplitude, target_frequency, half_width=3.0):
    """Sum squared amplitude bins; not integrated PSD or physical energy.

    Bin spacing is deliberately not included. Empty bands return zero as before.
    """
    frequencies, amplitude = _spectrum(frequencies, amplitude)
    _positive(target_frequency, "target_frequency")
    _positive(half_width, "half_width")
    mask = (frequencies >= target_frequency - half_width) & (frequencies <= target_frequency + half_width)
    return np.sum(amplitude[mask] ** 2)


def harmonic_band_score(frequencies, amplitude, base_frequency, harmonics=3, half_width=3.0):
    """Sum squared-bin scores over the harmonic bands."""
    if isinstance(harmonics, bool) or not isinstance(harmonics, int) or harmonics < 1:
        raise ValueError("harmonics must be a positive integer")
    total = 0
    for harmonic in range(1, harmonics + 1):
        total += band_spectral_energy(frequencies, amplitude, harmonic * base_frequency, half_width)
    return total


def fault_scores(frequencies, amplitude, rpm, method="peak"):
    """Compute peak sums or band-squared sums and ratios with historical defaults.

    Band ratios divide by the sum across the three selected families plus 1e-12.
    They are neither whole-spectrum energy fractions nor class probabilities.
    """
    targets = bearing_frequencies(rpm)
    families = ["BPFI", "BPFO", "BSF"]
    if method == "peak":
        return {f"{name}_score": harmonic_score(frequencies, amplitude, targets[name])[0]
                for name in families}
    if method != "band":
        raise ValueError("method must be 'peak' or 'band'")
    scores = {name: harmonic_band_score(frequencies, amplitude, targets[name]) for name in families}
    total = sum(scores.values())
    return {**{f"{name}_band_score": value for name, value in scores.items()},
            **{f"{name}_ratio": value / (total + 1e-12) for name, value in scores.items()}}


def frequency_rule(frequencies, amplitude, rpm, method="peak"):
    """Return the historical three-fault argmax label and scores, without a healthy class.

    Ties retain original family order (InnerRace, OuterRace, Ball). Even all-zero
    scores select InnerRace; this historical behavior is not diagnostic evidence.
    """
    scores = fault_scores(frequencies, amplitude, rpm, method)
    suffix = "score" if method == "peak" else "ratio"
    labels = {f"BPFI_{suffix}": "InnerRace", f"BPFO_{suffix}": "OuterRace", f"BSF_{suffix}": "Ball"}
    selected = max(labels, key=lambda name: scores[name])
    return labels[selected], scores


def shaft_spaced_pairs(peak_frequencies, rpm, tolerance=1.0):
    """Return all-pairs shaft-spacing candidates; no significance test is performed."""
    peaks = np.asarray(peak_frequencies)
    if peaks.ndim != 1 or not np.isfinite(peaks).all() or np.any(np.diff(peaks) < 0):
        raise ValueError("Expected sorted finite peak frequencies")
    _positive(tolerance, "tolerance")
    shaft = bearing_frequencies(rpm)["shaft"]
    pairs = []
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            spacing = peaks[j] - peaks[i]
            if abs(spacing - shaft) <= tolerance:
                pairs.append({"peak_1": peaks[i], "peak_2": peaks[j], "spacing": spacing,
                              "difference_from_shaft": spacing - shaft})
    return pairs
