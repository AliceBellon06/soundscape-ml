"""
Dataset class for training SoundscapeCNN.

Expected folder layout (mirrors torchvision ImageFolder convention):
    data/processed/
        train/
            biophony_birds/       *.wav  *.mp3  *.flac
            biophony_insects/
            biophony_amphibians/
            geophony_wind/
            geophony_rain_water/
            anthropophony_traffic/
            anthropophony_construction/
            anthropophony_machinery/
        val/
            ...

Each audio file is loaded, randomly cropped to DURATION seconds, and
converted to a normalised mel-spectrogram of shape (1, N_MELS, T).

Augmentation applied during training:
  - Random gain   ± 6 dB
  - Time stretch  ×0.9 – ×1.1  (via librosa on the fly)
  - SpecAugment   (time / frequency masking on the spectrogram)
"""

from __future__ import annotations
import random
from pathlib import Path

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

from .features import (
    SAMPLE_RATE, DURATION, N_MELS, N_FFT, HOP_LENGTH, F_MIN, F_MAX,
    load_audio, compute_mel_spectrogram,
)
from .model import CLASS_TO_IDX

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff"}


class SoundscapeDataset(Dataset):
    """Audio dataset that returns (mel_spectrogram_tensor, label_int) pairs."""

    def __init__(
        self,
        root: str | Path,
        augment: bool = False,
        segment_duration: float = DURATION,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.root     = Path(root)
        self.augment  = augment
        self.duration = segment_duration
        self.sr       = sample_rate
        self.samples: list[tuple[Path, int]] = []

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label_name = class_dir.name
            if label_name not in CLASS_TO_IDX:
                continue
            label_idx = CLASS_TO_IDX[label_name]
            for f in class_dir.rglob("*"):
                if f.suffix.lower() in AUDIO_EXTENSIONS:
                    self.samples.append((f, label_idx))

        if not self.samples:
            raise RuntimeError(
                f"No audio files found under {self.root}. "
                "Check that sub-directory names match CLASSES in model.py."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        y, sr = load_audio(path, target_sr=self.sr)

        # --- Random crop / pad to fixed duration ---
        n_target = int(self.duration * sr)
        if len(y) >= n_target:
            start = random.randint(0, len(y) - n_target)
            y = y[start : start + n_target]
        else:
            pad = np.zeros(n_target, dtype=np.float32)
            pad[: len(y)] = y
            y = pad

        # --- Augmentation ---
        if self.augment:
            y = self._augment_waveform(y, sr)

        # --- Mel-spectrogram ---
        mel = compute_mel_spectrogram(y, sr)

        if self.augment:
            mel = self._spec_augment(mel)

        # Normalise to [0, 1] for stable training
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

        tensor = torch.from_numpy(mel).unsqueeze(0)  # (1, N_MELS, T)
        return tensor, label

    # ── Augmentation helpers ──────────────────────────────────────────────────

    def _augment_waveform(self, y: np.ndarray, sr: int) -> np.ndarray:
        # Random gain ± 6 dB
        gain_db = random.uniform(-6.0, 6.0)
        y = y * (10 ** (gain_db / 20.0))

        # Time stretch (10 % variation) — resample to preserve length
        if random.random() < 0.5:
            rate = random.uniform(0.9, 1.1)
            y_stretched = librosa.effects.time_stretch(y, rate=rate)
            n_target = len(y)
            if len(y_stretched) >= n_target:
                y = y_stretched[:n_target]
            else:
                pad = np.zeros(n_target, dtype=np.float32)
                pad[: len(y_stretched)] = y_stretched
                y = pad

        return y.astype(np.float32)

    def _spec_augment(self, mel: np.ndarray) -> np.ndarray:
        """SpecAugment: mask up to 20 % of time frames and 15 % of mel bins."""
        mel = mel.copy()
        n_mels, n_frames = mel.shape

        # Frequency masking
        f_mask = random.randint(0, max(1, int(n_mels * 0.15)))
        f0 = random.randint(0, n_mels - f_mask)
        mel[f0 : f0 + f_mask, :] = mel.min()

        # Time masking
        t_mask = random.randint(0, max(1, int(n_frames * 0.20)))
        t0 = random.randint(0, n_frames - t_mask)
        mel[:, t0 : t0 + t_mask] = mel.min()

        return mel
