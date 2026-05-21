"""
Training script for SoundscapeCNN.

Usage
-----
    python -m soundscape_ml.train \
        --data-dir  data/processed \
        --epochs    30 \
        --batch     32 \
        --lr        1e-3 \
        --output    checkpoints/best.pt

The script expects the folder layout described in dataset.py.
It saves the best checkpoint (by validation accuracy) to --output.
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .dataset import SoundscapeDataset
from .model import SoundscapeCNN, NUM_CLASSES, save_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SoundscapeCNN")
    p.add_argument("--data-dir",  default="data/processed")
    p.add_argument("--epochs",    type=int,   default=30)
    p.add_argument("--batch",     type=int,   default=32)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--workers",   type=int,   default=4)
    p.add_argument("--output",    default="checkpoints/best.pt")
    p.add_argument("--no-pretrain", action="store_true",
                   help="Train from scratch (no ImageNet weights)")
    return p.parse_args()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * len(y)
            total_acc  += accuracy(logits, y) * len(y)
            n += len(y)

    return total_loss / n, total_acc / n


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    data_root = Path(args.data_dir)
    train_ds  = SoundscapeDataset(data_root / "train", augment=True)
    val_ds    = SoundscapeDataset(data_root / "val",   augment=False)
    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = SoundscapeCNN(num_classes=NUM_CLASSES,
                               pretrained=not args.no_pretrain).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    print(f"\n{'Epoch':>5}  {'Train loss':>10}  {'Train acc':>9}  "
          f"{'Val loss':>9}  {'Val acc':>8}  {'Time':>6}")
    print("─" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                    device, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, optimizer,
                                    device, training=False)
        scheduler.step()

        elapsed = time.time() - t0
        marker  = " ✓" if vl_acc > best_val_acc else ""
        print(f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>8.2%}  "
              f"{vl_loss:>9.4f}  {vl_acc:>7.2%}  {elapsed:>5.1f}s{marker}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save_checkpoint(model, optimizer, epoch, vl_acc, output_path)

    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    print(f"Checkpoint saved to: {output_path}")


if __name__ == "__main__":
    main()
