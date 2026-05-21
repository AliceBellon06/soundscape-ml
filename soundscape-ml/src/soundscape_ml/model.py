"""
SoundscapeCNN — ResNet-18 backbone adapted for single-channel mel-spectrogram input.

Urban soundscape classes (8):
  biophony   → birds, insects, amphibians
  geophony   → wind, rain/water
  anthropophony → traffic, construction, machinery
"""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

# ── Label space ───────────────────────────────────────────────────────────────

CLASSES: dict[int, str] = {
    0: "biophony_birds",
    1: "biophony_insects",
    2: "biophony_amphibians",
    3: "geophony_wind",
    4: "geophony_rain_water",
    5: "anthropophony_traffic",
    6: "anthropophony_construction",
    7: "anthropophony_machinery",
}

# Maps each class to its Krause category for NDSI-style analysis
SOUNDSCAPE_TYPE: dict[str, str] = {
    "biophony_birds":             "biophony",
    "biophony_insects":           "biophony",
    "biophony_amphibians":        "biophony",
    "geophony_wind":              "geophony",
    "geophony_rain_water":        "geophony",
    "anthropophony_traffic":      "anthropophony",
    "anthropophony_construction": "anthropophony",
    "anthropophony_machinery":    "anthropophony",
}

CLASS_TO_IDX: dict[str, int] = {v: k for k, v in CLASSES.items()}
NUM_CLASSES = len(CLASSES)


# ── Architecture ──────────────────────────────────────────────────────────────

class SoundscapeCNN(nn.Module):
    """
    ResNet-18 adapted for (1, n_mels, T) mel-spectrogram inputs.

    Modifications from the standard ResNet-18:
      - conv1 kernel unchanged but input channels = 1 (greyscale spectrogram)
      - Final FC layer replaced with a (512 → num_classes) head
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Adapt the first conv for single-channel input.
        # Average the pretrained RGB weights across the channel dimension so
        # ImageNet-learned low-level features (edges, textures) transfer cleanly.
        orig_conv1 = backbone.conv1
        new_conv1  = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            with torch.no_grad():
                new_conv1.weight[:] = orig_conv1.weight.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv1

        # Replace the final classification head
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        return self.backbone(x)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model: SoundscapeCNN, optimizer, epoch: int,
                    val_acc: float, path: str | Path) -> None:
    torch.save({
        "epoch":     epoch,
        "val_acc":   val_acc,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def load_checkpoint(path: str | Path,
                    device: str = "cpu") -> tuple[SoundscapeCNN, dict]:
    ckpt  = torch.load(path, map_location=device)
    model = SoundscapeCNN(pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt
