"""
Acoustic ecology indices for soundscape quality assessment.

Four established indices from the literature:
  ACI  — Acoustic Complexity Index     (Pieretti et al., 2011)
  ADI  — Acoustic Diversity Index      (Villanueva-Rivera et al., 2011)
  NDSI — Normalised Difference         (Joo et al., 2011)
         Soundscape Index
  BI   — Bioacoustic Index             (Boelman et al., 2007)

Each function accepts a raw waveform (y, sr) and returns a scalar.
compute_soundscape_report() runs all four and adds an interpretation.
"""

from __future__ import annotations
import numpy as np
import librosa
from .features import load_audio, SAMPLE_RATE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _power_spectrogram(y: np.ndarray, sr: int,
                       n_fft: int = 1024, hop_length: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Return (S_power [freq x time], freqs) for index computation."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return S, freqs


def _band_power(S: np.ndarray, freqs: np.ndarray,
                f_low: float, f_high: float) -> float:
    """Mean power in a frequency band."""
    mask = (freqs >= f_low) & (freqs <= f_high)
    return float(S[mask, :].mean()) if mask.any() else 0.0


# ── ACI ───────────────────────────────────────────────────────────────────────

def acoustic_complexity_index(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    j_seconds: float = 5.0,
    f_min: float = 200.0,
    f_max: float = 10000.0,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> float:
    """
    ACI: ratio of absolute temporal variation to total intensity per frequency bin.
    High ACI → complex, biophony-rich soundscape.

    j_seconds: length of each temporal sub-window (Pieretti default = 5 s)
    """
    S, freqs = _power_spectrogram(y, sr, n_fft, hop_length)
    band = (freqs >= f_min) & (freqs <= f_max)
    S = S[band, :]

    j_frames = max(1, int(j_seconds * sr / hop_length))
    n_windows = S.shape[1] // j_frames

    aci_total = 0.0
    for w in range(n_windows):
        window = S[:, w * j_frames : (w + 1) * j_frames]
        diffs   = np.abs(np.diff(window, axis=1)).sum(axis=1)
        totals  = window.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(totals > 0, diffs / totals, 0.0)
        aci_total += ratio.sum()

    return float(aci_total)


# ── ADI ───────────────────────────────────────────────────────────────────────

def acoustic_diversity_index(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    max_freq: float = 10000.0,
    freq_step: float = 1000.0,
    db_threshold: float = -50.0,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> float:
    """
    ADI: Shannon entropy of the proportion of above-threshold energy per frequency band.
    Higher ADI → more even distribution of acoustic energy across bands → more diversity.
    """
    S_power, freqs = _power_spectrogram(y, sr, n_fft, hop_length)
    S_db = 10 * np.log10(S_power + 1e-10)

    bands = np.arange(0, max_freq, freq_step)
    proportions = []
    for f_low in bands:
        f_high = f_low + freq_step
        mask = (freqs >= f_low) & (freqs < f_high)
        if not mask.any():
            continue
        band_slice = S_db[mask, :]
        proportion = float((band_slice > db_threshold).mean())
        proportions.append(proportion)

    proportions = np.array(proportions)
    proportions = proportions[proportions > 0]
    if len(proportions) == 0:
        return 0.0
    proportions /= proportions.sum()
    return float(-np.sum(proportions * np.log(proportions)))


# ── NDSI ──────────────────────────────────────────────────────────────────────

def normalised_difference_soundscape_index(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    anthro_band: tuple[float, float] = (1000.0, 2000.0),
    bio_band:    tuple[float, float] = (2000.0, 8000.0),
    n_fft: int = 1024,
    hop_length: int = 512,
) -> float:
    """
    NDSI: (biophony − anthropophony) / (biophony + anthropophony).
    Range [–1, +1]. Positive → biophony dominant; negative → anthropophony dominant.
    """
    S, freqs = _power_spectrogram(y, sr, n_fft, hop_length)
    bio   = _band_power(S, freqs, *bio_band)
    anthro = _band_power(S, freqs, *anthro_band)
    denom = bio + anthro
    if denom == 0:
        return 0.0
    return float((bio - anthro) / denom)


# ── BI ────────────────────────────────────────────────────────────────────────

def bioacoustic_index(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    f_min: float = 2000.0,
    f_max: float = 8000.0,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> float:
    """
    BI: mean dB amplitude integrated over the 2–8 kHz bird/insect communication band.
    Strongly correlated with bird species richness in field studies.
    """
    S_power, freqs = _power_spectrogram(y, sr, n_fft, hop_length)
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not mask.any():
        return 0.0
    mean_power = S_power[mask, :].mean()
    return float(10 * np.log10(mean_power + 1e-10))


# ── Combined report ───────────────────────────────────────────────────────────

def compute_soundscape_report(path: str) -> dict:
    """
    Load an audio file and return all four acoustic indices plus a text interpretation.

    Returns
    -------
    dict with keys: aci, adi, ndsi, bi, interpretation, health
    """
    y, sr = load_audio(path)

    aci  = acoustic_complexity_index(y, sr)
    adi  = acoustic_diversity_index(y, sr)
    ndsi = normalised_difference_soundscape_index(y, sr)
    bi   = bioacoustic_index(y, sr)

    # Heuristic health score based on NDSI (primary) and BI (secondary)
    if ndsi > 0.6 and bi > -20:
        health = "Excellent"
        interp = "Strong biophony dominance. High species richness expected. Minimal anthropogenic masking."
    elif ndsi > 0.2:
        health = "Good"
        interp = "Biophony exceeds anthropophony. Moderate biodiversity. Some noise pressure present."
    elif ndsi > -0.2:
        health = "Moderate"
        interp = "Balanced or mixed soundscape. Acoustic niche compression likely for some species."
    elif ndsi > -0.6:
        health = "Poor"
        interp = "Anthropophony dominant. Significant masking of biophony. Habitat quality degraded."
    else:
        health = "Critical"
        interp = "Severe anthropophony. Biophony near absent. Immediate acoustic management needed."

    return {
        "aci":  round(aci,  2),
        "adi":  round(adi,  4),
        "ndsi": round(ndsi, 4),
        "bi":   round(bi,   2),
        "health": health,
        "interpretation": interp,
    }
