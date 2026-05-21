# SoundscapeML

**Urban acoustic biodiversity monitoring using deep learning.**

SoundscapeML classifies urban audio recordings into 8 ecologically meaningful soundscape categories and computes 4 validated acoustic ecology indices — giving planners and researchers a real-time biodiversity health score for any city soundscape.

→ Read the full research context: [Soundscapes, Biodiversity & the Sound-Aware City](https://YOUR_USERNAME.github.io/soundscape-ml)

---

## What it does

| Input | Output |
|-------|--------|
| Any `.wav / .mp3 / .flac` recording | Predicted soundscape class + confidence |
| | ACI, ADI, NDSI, BI acoustic indices |
| | Health tier: Excellent → Critical |
| | Full JSON for GIS / dashboard integration |

### 8 soundscape classes

| Krause category | SoundscapeML class |
|---|---|
| **Biophony** | `biophony_birds`, `biophony_insects`, `biophony_amphibians` |
| **Geophony** | `geophony_wind`, `geophony_rain_water` |
| **Anthropophony** | `anthropophony_traffic`, `anthropophony_construction`, `anthropophony_machinery` |

### Acoustic indices (computed without a trained model)

| Index | What it measures | Range |
|-------|-----------------|-------|
| **ACI** | Acoustic Complexity Index — temporal amplitude variation per frequency bin | > 0, higher = richer biophony |
| **ADI** | Acoustic Diversity Index — Shannon entropy of energy across bands | 0 – ln(B), higher = more diverse |
| **NDSI** | Normalised Difference Soundscape Index — biophony vs anthropophony ratio | –1 to +1, positive = biophony dominant |
| **BI**  | Bioacoustic Index — mean dB in the 2–8 kHz bird/insect band | dB, correlated with bird species richness |

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/soundscape-ml.git
cd soundscape-ml
pip install -r requirements.txt
pip install -e .

# 2. Compute acoustic indices only (no training needed)
python -m soundscape_ml.predict --audio path/to/recording.wav

# 3. With a trained checkpoint
python -m soundscape_ml.predict \
    --audio       path/to/recording.wav \
    --checkpoint  checkpoints/best.pt \
    --format      json
```

Example output (table mode):
```
  File     : dawn_park_london.wav
  Class    : biophony_birds  (biophony)
  Confidence: 87.3%
  Top-3    :
             biophony_birds                      87.3%
             biophony_insects                     8.1%
             geophony_wind                        3.4%

  ── Acoustic Indices ──────────────
  ACI      : 4821.34
  ADI      : 1.8802
  NDSI     : 0.6240  (range –1 → +1; positive = biophony dominant)
  BI       : -18.42 dB

  Health   : Excellent
  Note     : Strong biophony dominance. High species richness expected.
```

---

## Training your own model

### 1. Get the data

Download these two free datasets:

- **UrbanSound8K** — [urbansounddataset.weebly.com](https://urbansounddataset.weebly.com/urbansound8k.html) (CC BY 4.0)
- **ESC-50** — [github.com/karoldvl/ESC-50](https://github.com/karoldvl/ESC-50) (CC BY-NC 3.0)

Then run the preparation script:
```bash
python data/download_data.py \
    --urbansound path/to/UrbanSound8K \
    --esc50       path/to/ESC-50 \
    --output      data/processed
```

### 2. Train

```bash
python -m soundscape_ml.train \
    --data-dir  data/processed \
    --epochs    30 \
    --batch     32 \
    --lr        1e-3 \
    --output    checkpoints/best.pt
```

Training takes ~15 minutes on a GPU, ~2 hours on CPU.  
Best checkpoint is saved automatically by validation accuracy.

### 3. Evaluate on a folder

```bash
python -m soundscape_ml.predict \
    --audio      data/processed/val \
    --checkpoint checkpoints/best.pt \
    --format     json > results.json
```

---

## Repository structure

```
soundscape-ml/
├── src/
│   └── soundscape_ml/
│       ├── features.py    # Audio loading, mel-spectrogram, MFCC
│       ├── indices.py     # ACI, ADI, NDSI, BI computation
│       ├── model.py       # ResNet-18 CNN + label definitions
│       ├── dataset.py     # PyTorch Dataset + augmentation
│       ├── train.py       # Training loop CLI
│       └── predict.py     # Inference CLI
├── data/
│   └── download_data.py   # Dataset remapping script
├── paper/
│   └── main.tex           # LaTeX methods paper (compile with pdflatex)
├── requirements.txt
└── README.md
```

---

## Compile the paper

```bash
cd paper
pdflatex main.tex
pdflatex main.tex   # run twice for references
```

Requires a LaTeX distribution (TeX Live, MikTeX) with TikZ.

---

## Extending SoundscapeML

The most impactful next steps are:

1. **Add a species-level head** — train a second output on Xeno-Canto bird recordings for fine-grained identification within the `biophony_birds` class.
2. **Citywide monitoring dashboard** — chain `predict.py --format json` output into a GIS layer updating NDSI/ACI scores per grid cell in real time.
3. **Regression head for indices** — instead of computing ACI/NDSI analytically, jointly learn to predict them end-to-end, enabling gradient flow from monitoring quality back into the feature extractor.

---

## Citation

If you use SoundscapeML in research, please cite the accompanying paper:

```bibtex
@misc{soundscapeml2026,
  title  = {SoundscapeML: A Deep Learning Pipeline for Urban Acoustic
            Biodiversity Monitoring},
  author = {Soundscapes \& Biodiversity Project},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/soundscape-ml}
}
```

---

## Licence

MIT — see [LICENSE](LICENSE). Training datasets have their own licences (see links above).
