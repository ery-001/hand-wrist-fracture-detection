# MURA Hand & Wrist Fracture Detection

Binary classifier for **hand and wrist X-ray abnormalities** on the [MURA](https://stanfordmlgroup.github.io/competitions/mura/) dataset. The pipeline uses **DenseNet121** (ImageNet pre-trained) with transfer learning, **CLAHE** preprocessing for bone contrast, class-weighted training for label imbalance, and a two-stage **train → fine-tune → evaluate** workflow.

> **Note:** This project is for research and education. It is not a medical device and must not be used for clinical diagnosis without proper validation and regulatory review.

---

## Features

- Filters MURA to **hand (`XR_HAND`) and wrist (`XR_WRIST`)** studies only
- **CLAHE** enhancement on grayscale X-rays before feeding RGB tensors to DenseNet
- **70% train / 15% validation / 15% test** split (validation carved from MURA train; MURA valid set used as held-out test)
- **Frozen backbone** initial training, then **fine-tuning** of the top 50 DenseNet layers at a low learning rate
- **Balanced class weights** to mitigate fracture vs. normal imbalance
- Evaluation with a **0.3 decision threshold** (favoring recall on positive/fracture cases)
- Timestamped run folders under `runs/` with logs, saved models, and metrics

---

## Project structure

```
latest-project-trial/
├── src/
│   ├── data_loader.py   # MURA CSV parsing, CLAHE, Keras generators
│   ├── model.py         # DenseNet121 + binary classification head
│   ├── train.py         # Initial training (frozen backbone)
│   ├── fine_tune.py     # Unfreeze top layers, low LR fine-tuning
│   └── evaluate.py      # Test-set metrics and confusion matrix
├── runs/                # Experiment outputs (created at runtime)
│   ├── run_YYYYMMDD_HHMMSS/
│   │   ├── best_model.keras
│   │   ├── training_log.csv
│   │   └── training_history.png
│   └── finetune_YYYYMMDD_HHMMSS/
│       ├── best_finetuned_model.keras
│       ├── finetuning_log.csv
│       └── evaluation_metrics.txt   # after running evaluate.py
├── requirements.txt     # Root Python dependencies
└── .gitignore           # Excludes dataset, venvs, and local env folders
```

The MURA dataset is **not** committed to this repository. Place it at the repo root as described below.

---

## Requirements

- Python 3.10+ (3.11–3.13 tested in development)
- TensorFlow 2.x, OpenCV, pandas, scikit-learn, matplotlib, seaborn

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

CSV paths are relative to the **repository root** (e.g. `MURA-v1.1/train/...`). The data loader resolves images with `directory=base_dir`, where scripts use `BASE_DIR = '../'` when run from `src/`.

---

## Usage

All commands assume your virtual environment is active and you are in the `src/` directory:

```bash
cd src
```

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

Loads the **latest** `runs/run_*` folder (by modification time), unfreezes the **top 50** DenseNet layers, and trains with learning rate **1e-5** for up to **25 epochs**.

Saves to `runs/finetune_<timestamp>/`:
- `best_finetuned_model.keras`
- `finetuning_log.csv`

### 3. Evaluate

```bash
python evaluate.py
```

Loads the **latest run folder that contains a `.keras` model** (base or fine-tuned), runs inference on the **test** generator, and writes:

- `evaluation_metrics.txt` — sklearn classification report
- `confusion_matrix.png`

Predictions use **sigmoid > 0.3** as the positive (fracture) threshold.

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
| `fine_tune.py` | `EPOCHS`, `FINE_TUNE_LEARNING_RATE`, layers unfrozen (last 50) |
| `evaluate.py` | Prediction threshold (default `0.3`) |
| `data_loader.py` | `batch_size`, `target_size`, augmentation in `ImageDataGenerator` |

`BASE_DIR` should point to the repo root (parent of `src/`), default `'../'`.

---

## Example results

Fine-tuned runs on the held-out test set have reported metrics similar to:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Normal (0) | ~0.80 | ~0.38 | ~0.51 |
| Fracture (1) | ~0.52 | ~0.88 | ~0.65 |

The low threshold prioritizes **fracture recall** at the cost of specificity. Tune the threshold and class weights for your use case.

---

## Ignored paths (`.gitignore`)

- `MURA-v1.1/` — dataset
- `mura_env/`, `myenv/` — local virtual environments
- `__pycache__/` — Python bytecode (add explicitly if not already ignored)

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
