# CAX-Net Installation Guide

A modular framework for **pneumonia detection from chest X-rays** using **CvT**, **Autoencoder-based dimensionality reduction**, and **XGBoost**.

## 1. Clone the Repository

```bash
git clone https://github.com/nakka-thirupathi/cax-net.git
cd cax-net
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# or
source .venv/bin/activate   # Linux / Mac
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**

- `torch`, `timm` — Vision Transformers
- `numpy`, `pandas`, `scikit-learn` — Data processing
- `xgboost` — Classifier
- `matplotlib`, `seaborn` — Visualizations
- `opencv-python` — Image handling

## 4. Create Required Folders

```bash
mkdir -p models results checkpoints
```

- `models/` — Saved model weights (`CvT`, `Autoencoder`, `XGBoost`)
- `results/` — Visualizations and evaluation metrics
- `checkpoints/` — Training checkpoints

## 5. Dataset

Download the datasets:

1. [Chest X-Ray Pneumonia Dataset (Paul Mooney, Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. [Labeled Chest X-Ray Dataset (Tolgadincer, Kaggle)](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images)

- Classes: `Pneumonia` and `Normal`
- Handled automatically by `data/data_loader.py`.

## 6. Running the Project

### Training

```bash
python train.py
```

- Extract features with CvT
- Train Autoencoder for dimensionality reduction
- Train XGBoost classifier
- Save models and visualizations

### Testing / Inference

```bash
python test.py
```

- Loads models from `models/`
- Generates predictions and metrics
- Saves outputs in `results/`

## 7. Configuration

Modify `config.py` to adjust:

```python
class Config:
    IMG_SIZE = 224
    BATCH_SIZE = 32
    LATENT_DIM = 64
    AE_EPOCHS = 50
    DEVICE = 'cuda'  # or 'cpu'
```

- `XGB_PARAMS` for XGBoost can also be customized

## 8. Extending the Project

- Add custom models in `models/cax_models.py`
- Add visualizations in `utils/visualize.py`
