"""
Inference CLI for SoundscapeCNN.

Usage
-----
    # Single file — print results to console
    python -m soundscape_ml.predict --audio path/to/recording.wav --checkpoint best.pt

    # JSON output (for integration with other tools / the web page)
    python -m soundscape_ml.predict --audio recording.wav --checkpoint best.pt --format json

    # Folder scan
    python -m soundscape_ml.predict --audio recordings/ --checkpoint best.pt --format json

When no checkpoint is provided, only the acoustic indices are computed (no class prediction).
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .features import extract_features_for_model, N_MELS
from .indices import compute_soundscape_report
from .model import SoundscapeCNN, CLASSES, SOUNDSCAPE_TYPE, load_checkpoint

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff"}


def predict_file(
    audio_path: str | Path,
    model: SoundscapeCNN | None = None,
    device: str = "cpu",
) -> dict:
    """
    Run the full pipeline on one audio file.

    Returns
    -------
    dict with keys:
        file, predicted_class, soundscape_type, confidence, top3,
        aci, adi, ndsi, bi, health, interpretation
    """
    audio_path = Path(audio_path)
    result: dict = {"file": str(audio_path)}

    # ── Acoustic indices (always computed) ────────────────────────────────────
    report = compute_soundscape_report(str(audio_path))
    result.update(report)

    # ── Neural classifier (only when model is available) ──────────────────────
    if model is not None:
        segments = extract_features_for_model(audio_path)

        # Normalise, add batch+channel dims
        tensors = []
        for mel in segments:
            mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
            # Resize to fixed (N_MELS, 128) so variable-length files work
            t = torch.from_numpy(mel).unsqueeze(0)            # (1, n_mels, T)
            t = F.interpolate(t.unsqueeze(0), size=(N_MELS, 128),
                              mode="bilinear", align_corners=False).squeeze(0)
            tensors.append(t)

        batch  = torch.stack(tensors).to(device)              # (S, 1, n_mels, 128)
        with torch.no_grad():
            logits = model(batch)                              # (S, C)
            probs  = F.softmax(logits, dim=-1).mean(dim=0)    # average over segments

        top3_idx  = probs.topk(3).indices.tolist()
        top3_conf = probs.topk(3).values.tolist()

        predicted_idx   = top3_idx[0]
        predicted_class = CLASSES[predicted_idx]
        confidence      = float(top3_conf[0])

        result.update({
            "predicted_class":  predicted_class,
            "soundscape_type":  SOUNDSCAPE_TYPE[predicted_class],
            "confidence":       round(confidence, 4),
            "top3": [
                {"class": CLASSES[i], "prob": round(float(p), 4)}
                for i, p in zip(top3_idx, top3_conf)
            ],
        })

    return result


def _fmt_table(result: dict) -> str:
    lines = [
        f"\n  File     : {result['file']}",
    ]
    if "predicted_class" in result:
        lines += [
            f"  Class    : {result['predicted_class']}  ({result['soundscape_type']})",
            f"  Confidence: {result['confidence']:.1%}",
            "  Top-3    :",
        ]
        for entry in result.get("top3", []):
            lines.append(f"             {entry['class']:35s} {entry['prob']:.1%}")
    lines += [
        "",
        "  ── Acoustic Indices ──────────────",
        f"  ACI      : {result['aci']}",
        f"  ADI      : {result['adi']}",
        f"  NDSI     : {result['ndsi']}  (range –1 → +1; positive = biophony dominant)",
        f"  BI       : {result['bi']} dB",
        "",
        f"  Health   : {result['health']}",
        f"  Note     : {result['interpretation']}",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SoundscapeML — predict urban soundscape class")
    p.add_argument("--audio",      required=True, help="Audio file or directory")
    p.add_argument("--checkpoint", default=None,  help="Path to .pt checkpoint (optional)")
    p.add_argument("--format",     choices=["table", "json"], default="table")
    p.add_argument("--device",     default="cpu")
    return p.parse_args()


def main():
    args  = parse_args()
    model = None

    if args.checkpoint:
        model, meta = load_checkpoint(args.checkpoint, device=args.device)
        print(f"Loaded checkpoint — epoch {meta['epoch']}, "
              f"val acc {meta['val_acc']:.2%}", file=sys.stderr)

    audio_path = Path(args.audio)
    files = (
        [audio_path]
        if audio_path.is_file()
        else [f for f in audio_path.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS]
    )

    results = []
    for f in sorted(files):
        result = predict_file(f, model=model, device=args.device)
        results.append(result)
        if args.format == "table":
            print(_fmt_table(result))

    if args.format == "json":
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
