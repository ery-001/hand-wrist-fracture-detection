# MURA Hand & Wrist Fracture Detection

Binary classifier for **hand and wrist X-ray abnormalities** on the [MURA](https://stanfordmlgroup.github.io/competitions/mura/) dataset. The pipeline uses **DenseNet121** (ImageNet pre-trained) with a **spatial attention block**, **CLAHE** preprocessing, class-weighted training, and a two-stage **train → fine-tune → evaluate** workflow. Inference scripts add **Grad-CAM** explainability with optional target-lock bounding boxes.

> **Note:** This project is for research and education. It is not a medical device and must not be used for clinical diagnosis without proper validation and regulatory review.

---

## Features

- Filters MURA to **hand (`XR_HAND`) and wrist (`XR_WRIST`)** studies only
- **CLAHE** enhancement on grayscale X-rays before feeding RGB tensors to DenseNet
- **Spatial attention spotlight** (1×1 conv) to amplify fracture-relevant feature maps
- **70% train / 15% validation / 15% test** split (validation carved from MURA train; MURA valid set used as held-out test)
- **Frozen backbone** initial training, then **fine-tuning** of the top **150** DenseNet layers at a low learning rate
- **Balanced class weights** to mitigate fracture vs. normal imbalance
- **Multi-threshold evaluation** at 0.3, 0.5, and 0.7 for clinical trade-offs
- **Grad-CAM** heatmaps and single-image prediction (CLI or file-upload GUI)
- **End-to-end orchestrator** to run the full pipeline sequentially
- Timestamped run folders under `runs/` with logs, saved models, and metrics

---

## Project structure

```
latest-project-trial/
├── src/
│   ├── data_loader.py      # MURA CSV parsing, CLAHE, Keras generators
│   ├── model.py              # DenseNet121 + attention block + classification head
│   ├── train.py              # Initial training (frozen backbone)
│   ├── fine_tune.py          # Unfreeze top 150 layers, low LR fine-tuning
│   ├── evaluate.py           # Multi-threshold test metrics and confusion matrices
│   ├── grad_cam.py           # Grad-CAM heatmaps + target-lock visualization
│   ├── predict.py            # CLI: predict a single X-ray with Grad-CAM overlay
│   ├── upload_predict.py     # GUI file picker for single-image prediction
│   ├── get_test_paths.py     # Export absolute test image paths to test_images_list.txt
│   └── requirements.txt      # Alternate pins (e.g. tensorflow-metal on Apple Silicon)
├── orchestration/
│   └── run_pipeline.py       # Runs train → fine_tune → evaluate in sequence
├── runs/                     # Experiment outputs (contents gitignored; folder kept)
│   ├── run_YYYYMMDD_HHMMSS/
│   │   ├── best_model.keras
│   │   ├── training_log.csv
│   │   └── training_history.png
│   └── finetune_YYYYMMDD_HHMMSS/
│       ├── best_finetuned_model.keras
│       ├── finetuning_log.csv
│       ├── evaluation_metrics_t0.3.txt
│       ├── evaluation_metrics_t0.5.txt
│       ├── evaluation_metrics_t0.7.txt
│       ├── confusion_matrix_t0.3.png
│       ├── confusion_matrix_t0.5.png
│       ├── confusion_matrix_t0.7.png
│       └── gradcam_visualizations.png   # after running grad_cam.py
├── requirements.txt            # Root Python dependencies
├── documentation.txt         # Extended project notes (plain text)
└── .gitignore                # Excludes dataset, venvs, and run artifacts
```

The MURA dataset is **not** committed to this repository. Place it at the repo root as described below.

---

## Requirements

- Python 3.10+ (3.11–3.13 tested in development)
- TensorFlow 2.x, OpenCV, pandas, scikit-learn, matplotlib, seaborn
- **tkinter** (stdlib; used by `upload_predict.py` for the file picker)

Install dependencies from the project root:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

On Apple Silicon, you may use `src/requirements.txt` instead if you need `tensorflow-metal` for GPU acceleration (that file pins TensorFlow 2.17 + Metal).

---

## Dataset setup

1. Download [MURA v1.1](https://stanfordmlgroup.github.io/competitions/mura/) from Stanford ML Group.
2. Extract so the layout matches:

```
latest-project-trial/
└── MURA-v1.1/
    ├── train/
    ├── valid/
    ├── train_image_paths.csv
    └── valid_image_paths.csv
```

CSV paths are relative to the **repository root** (e.g. `MURA-v1.1/train/...`). Scripts in `src/` use `BASE_DIR = '../'` so they resolve images from the repo root when run from `src/`.

To export absolute paths for all hand/wrist test images:

```bash
cd src
python get_test_paths.py
```

This writes `src/test_images_list.txt`.

---

## Usage

All commands below assume your virtual environment is active. Run training and evaluation scripts from `src/`:

```bash
cd src
```

### Quick start: full pipeline

From the `orchestration/` directory:

```bash
cd orchestration
python run_pipeline.py
```

Runs **train → fine_tune → evaluate** in order. The pipeline halts if any stage fails.

### 1. Train (frozen backbone)

```bash
python train.py
```

- **30 epochs** (early stopping on `val_loss`, patience 5)
- Batch size **32**, learning rate **0.001**
- Saves to `runs/run_<timestamp>/`:
  - `best_model.keras` (best `val_accuracy`)
  - `training_log.csv`
  - `training_history.png`

### 2. Fine-tune

```bash
python fine_tune.py
```

Loads the **latest** `runs/run_*` folder (by modification time), unfreezes the **top 150** DenseNet layers, and trains with learning rate **1e-5** for up to **40 epochs** (early stopping patience 8, `ReduceLROnPlateau` on `val_loss`).

Saves to `runs/finetune_<timestamp>/`:
- `best_finetuned_model.keras`
- `finetuning_log.csv`

### 3. Evaluate (multi-threshold)

```bash
python evaluate.py
```

Loads the **latest run folder that contains a `.keras` model** (base or fine-tuned), runs inference on the **test** set once, then evaluates at thresholds **0.3**, **0.5**, and **0.7**.

Writes per threshold:
- `evaluation_metrics_t<threshold>.txt` — sklearn classification report
- `confusion_matrix_t<threshold>.png`

Lower thresholds favor **fracture recall**; higher thresholds favor **specificity**.

### 4. Grad-CAM visualization

```bash
python grad_cam.py
```

Loads the latest model, runs Grad-CAM on a batch of 5 test images, and saves `gradcam_visualizations.png` to the run folder. Uses threshold **0.30** for target-lock bounding boxes on positive predictions.

### 5. Predict a single image

**CLI** (pass an image path):

```bash
python predict.py path/to/xray.png
```

**GUI file picker**:

```bash
python upload_predict.py
```

Both scripts load the latest model, apply the same preprocessing as training (resize, CLAHE, normalize), print a diagnosis, and display a Grad-CAM overlay. Default decision threshold is **0.5** in these scripts (see [Configuration](#configuration)).

### Optional: smoke-test modules

```bash
python data_loader.py   # Verifies CSV paths and generators
python model.py         # Builds and summarizes the model
```

---

## Model architecture

| Component | Detail |
|-----------|--------|
| Backbone | DenseNet121 (`imagenet` weights, `include_top=False`) |
| Attention | 1×1 Conv2D (sigmoid) × feature maps — spatial spotlight |
| Head | GlobalAveragePooling2D → Dropout(0.5) → Dense(1, sigmoid) |
| Loss | `binary_crossentropy` |
| Optimizer | Adam (0.001 train, 1e-5 fine-tune) |
| Metrics | Accuracy, precision, recall |

**Preprocessing:** resize to **224×224**, scale to `[0, 1]`, apply CLAHE on grayscale then convert back to 3-channel RGB. Training adds rotation (±15°), horizontal flip, and zoom (0.2).

---

## Data split

| Split | Source | Share |
|-------|--------|-------|
| Train | MURA `train_image_paths.csv` (hand/wrist) | ~70% |
| Validation | Same CSV, `validation_split=0.176` | ~15% |
| Test | MURA `valid_image_paths.csv` (hand/wrist) | ~15% |

Labels are parsed from paths: `positive` → 1 (abnormal), otherwise 0 (normal).

---

## Configuration

Edit constants at the top of each script in `src/`:

| Script | Key settings |
|--------|----------------|
| `train.py` | `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `BASE_DIR` |
| `fine_tune.py` | `EPOCHS` (40), `FINE_TUNE_LEARNING_RATE`, layers unfrozen (last 150) |
| `evaluate.py` | `CLINICAL_THRESHOLDS` (default `[0.3, 0.5, 0.7]`) |
| `grad_cam.py` | `CLINICAL_THRESHOLD` (default `0.30`) |
| `predict.py` / `upload_predict.py` | Decision threshold (default `0.5`) |
| `data_loader.py` | `batch_size`, `target_size`, augmentation in `ImageDataGenerator` |

`BASE_DIR` should point to the repo root (parent of `src/`), default `'../'`.

**Threshold note:** evaluation sweeps multiple cutoffs; inference scripts use a single threshold. Align `predict.py` and `upload_predict.py` with the threshold that best matches your clinical priority (recall vs. specificity).

---

## Example results

Fine-tuned runs on the held-out test set (~1,119 hand/wrist images) at threshold **0.3** have reported metrics similar to:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Normal (0) | ~0.80 | ~0.38 | ~0.51 |
| Fracture (1) | ~0.52 | ~0.88 | ~0.65 |

Run `evaluate.py` to generate fresh metrics at 0.3, 0.5, and 0.7 for your latest model.

---

## Ignored paths (`.gitignore`)

- `MURA-v1.1/` — dataset
- `mura_env/`, `myenv/`, `.venv/` — local virtual environments
- `runs/*` — experiment artifacts (the `runs/` folder itself is kept)

---

## License & attribution

- **MURA** dataset: see [Stanford MURA competition](https://stanfordmlgroup.github.io/competitions/mura/) for terms of use and citation.
- **DenseNet**: Huang et al., *Densely Connected Convolutional Networks* ([arXiv:1608.06993](https://arxiv.org/abs/1608.06993)).
- Pre-trained weights via Keras Applications (`imagenet`).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError` for CSV | Ensure `MURA-v1.1/` is at repo root and CSVs exist |
| `Could not find any base 'run_' directories` | Run `train.py` before `fine_tune.py` |
| No `.keras` in `runs/` | Complete at least one training run; check `best_model.keras` exists |
| Out of memory | Lower `BATCH_SIZE` in `train.py` / `fine_tune.py` |
| Slow training on CPU | Reduce `EPOCHS` or use a GPU-enabled TensorFlow build |
| GUI file picker does not open | Ensure tkinter is installed (usually bundled with Python on Windows/macOS) |
| Pipeline stops mid-run | Check the failing stage’s terminal output; fix the error and re-run from that stage |

For additional notes, see `documentation.txt` in the repo root.
