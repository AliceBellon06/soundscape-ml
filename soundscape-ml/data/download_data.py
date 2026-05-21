"""
Download and organise free training datasets.

Datasets used:
  1. UrbanSound8K  (Salamon et al., 2014)  — 8 732 urban sound clips, 10 classes
     https://urbansounddataset.weebly.com/urbansound8k.html
     Licence: Creative Commons Attribution 4.0

  2. ESC-50  (Piczak, 2015)  — 2 000 environmental clips, 50 classes
     https://github.com/karoldvl/ESC-50
     Licence: Creative Commons Attribution Non-Commercial 3.0

The script remaps classes from both datasets to SoundscapeML's 8-class scheme,
then writes a train/val split under data/processed/.

Usage
-----
    # After manually downloading UrbanSound8K and ESC-50 (see URLs above):
    python data/download_data.py \
        --urbansound path/to/UrbanSound8K \
        --esc50       path/to/ESC-50 \
        --output      data/processed \
        --val-fold    10          # UrbanSound8K fold to use as validation

Class mapping
-------------
UrbanSound8K classes → SoundscapeML classes:
  air_conditioner    → anthropophony_machinery
  car_horn           → anthropophony_traffic
  children_playing   → (dropped — no equivalent class)
  dog_bark           → biophony_birds       (closest biophony proxy)
  drilling           → anthropophony_construction
  engine_idling      → anthropophony_machinery
  gun_shot           → anthropophony_construction
  jackhammer         → anthropophony_construction
  siren              → anthropophony_traffic
  street_music       → (dropped)

ESC-50 classes (selected) → SoundscapeML classes:
  rain               → geophony_rain_water
  sea_waves          → geophony_rain_water
  wind               → geophony_wind
  crow / chirping_birds / roosters → biophony_birds
  frogs              → biophony_amphibians
  crickets / insects → biophony_insects
"""

from __future__ import annotations
import argparse
import csv
import shutil
import random
from pathlib import Path

# ── Class mappings ────────────────────────────────────────────────────────────

US8K_MAP: dict[str, str | None] = {
    "air_conditioner":  "anthropophony_machinery",
    "car_horn":         "anthropophony_traffic",
    "children_playing": None,
    "dog_bark":         "biophony_birds",
    "drilling":         "anthropophony_construction",
    "engine_idling":    "anthropophony_machinery",
    "gun_shot":         "anthropophony_construction",
    "jackhammer":       "anthropophony_construction",
    "siren":            "anthropophony_traffic",
    "street_music":     None,
}

ESC50_MAP: dict[str, str | None] = {
    "rain":          "geophony_rain_water",
    "sea_waves":     "geophony_rain_water",
    "wind":          "geophony_wind",
    "crow":          "biophony_birds",
    "chirping_birds":"biophony_birds",
    "roosters":      "biophony_birds",
    "frogs":         "biophony_amphibians",
    "crickets":      "biophony_insects",
    "insects":       "biophony_insects",
}


def copy_file(src: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)


def process_urbansound8k(us8k_root: Path, output: Path, val_fold: int) -> int:
    meta_path = us8k_root / "metadata" / "UrbanSound8K.csv"
    if not meta_path.exists():
        print(f"  UrbanSound8K metadata not found at {meta_path}. Skipping.")
        return 0

    n = 0
    with open(meta_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_class = row["class"].strip()
            dest_class = US8K_MAP.get(src_class)
            if dest_class is None:
                continue
            fold   = int(row["fold"])
            split  = "val" if fold == val_fold else "train"
            src    = us8k_root / "audio" / f"fold{fold}" / row["slice_file_name"]
            if src.exists():
                copy_file(src, output / split / dest_class)
                n += 1
    return n


def process_esc50(esc50_root: Path, output: Path, val_fraction: float = 0.2) -> int:
    meta_path = esc50_root / "meta" / "esc50.csv"
    if not meta_path.exists():
        print(f"  ESC-50 metadata not found at {meta_path}. Skipping.")
        return 0

    rows_by_class: dict[str, list] = {}
    with open(meta_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"].strip()
            if cat in ESC50_MAP and ESC50_MAP[cat]:
                rows_by_class.setdefault(cat, []).append(row)

    n = 0
    random.seed(42)
    for src_class, rows in rows_by_class.items():
        dest_class = ESC50_MAP[src_class]
        random.shuffle(rows)
        n_val = max(1, int(len(rows) * val_fraction))
        for i, row in enumerate(rows):
            split = "val" if i < n_val else "train"
            src   = esc50_root / "audio" / row["filename"]
            if src.exists():
                copy_file(src, output / split / dest_class)
                n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--urbansound", default=None)
    p.add_argument("--esc50",      default=None)
    p.add_argument("--output",     default="data/processed")
    p.add_argument("--val-fold",   type=int, default=10)
    args = p.parse_args()

    output = Path(args.output)
    total  = 0

    if args.urbansound:
        n = process_urbansound8k(Path(args.urbansound), output, args.val_fold)
        print(f"UrbanSound8K: {n} files copied")
        total += n

    if args.esc50:
        n = process_esc50(Path(args.esc50), output)
        print(f"ESC-50: {n} files copied")
        total += n

    if total == 0:
        print("No data found. Provide --urbansound and/or --esc50 paths.")
    else:
        print(f"\nTotal: {total} files → {output}")
        for split in ["train", "val"]:
            for cls_dir in sorted((output / split).iterdir()):
                n = len(list(cls_dir.glob("*")))
                print(f"  {split}/{cls_dir.name}: {n}")


if __name__ == "__main__":
    main()
