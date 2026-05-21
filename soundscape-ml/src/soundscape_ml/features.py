"""
Audio loading and feature extraction for urban soundscape classification.

Pipeline: audio file → resampled mono signal → mel-spectrogram (model input)
                                               → MFCC (alternative features)
"""

from __future__ import annotations
import numpy as np
import librosa
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050   # Hz
DURATION      = 3.0     # seconds per segment
N_MELS        = 128
N_FFT         = 2048
HOP_LENGTH    = 512
F_MIN         = 20.0    # Hz  — captures infrasound near-boundary
F_MAX         = 11025.0 # Hz  — Nyquist at 22 050 Hz
DB_REF        = 1e-5    # reference amplitude for dB conversion


def load_audio(
    path: str | Path,
    target_sr: int = SAMPLE_RATE,
    duration: float | None = None,
    offset: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono float32 array resampled to target_sr."""
    y, sr = librosa.load(str(path), sr=target_sr, mono=True,
                         duration=duration, offset=offset)
    return y, sr


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> np.ndarray:
    """
    Compute a log-mel spectrogram normalised to the range [–80, 0] dB.
    Returns shape (n_mels, T) as float32.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=f_min, fmax=f_max, power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)


def compute_mfcc(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = 40,
    delta: bool = True,
) -> np.ndarray:
    """
    Compute MFCCs and optionally their first delta (velocity) coefficients.
    Returns shape (n_mfcc [* 2 if delta], T).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if delta:
        d1 = librosa.feature.delta(mfcc)
        return np.vstack([mfcc, d1]).astype(np.float32)
    return mfcc.astype(np.float32)


def segment_audio(
    y: np.ndarray,
    sr: int,
    segment_duration: float = DURATION,
    overlap: float = 0.5,
) -> list[np.ndarray]:
    """
    Split a signal into fixed-length overlapping segments.
    Short clips are zero-padded to exactly one segment.
    """
    n_seg  = int(segment_duration * sr)
    n_hop  = int(n_seg * (1.0 - overlap))
    starts = range(0, max(1, len(y) - n_seg + 1), n_hop)

    segments = [y[s : s + n_seg] for s in starts]

    # pad the last segment if needed
    for i, seg in enumerate(segments):
        if len(seg) < n_seg:
            pad = np.zeros(n_seg, dtype=np.float32)
            pad[: len(seg)] = seg
            segments[i] = pad

    return segments


def extract_features_for_model(
    path: str | Path,
    segment_duration: float = DURATION,
    overlap: float = 0.5,
) -> list[np.ndarray]:
    """
    End-to-end: load → segment → mel-spectrogram.
    Returns a list of (N_MELS, T) arrays, one per segment.
    The model expects each array wrapped in a channel dimension: (1, N_MELS, T).
    """
    y, sr = load_audio(path)
    segments = segment_audio(y, sr, segment_duration, overlap)
    return [compute_mel_spectrogram(seg, sr) for seg in segments]
