# Data-Driven-Polymer-Property-Prediction-with-Machine-Learning
This repository contains my solution for the [NeurIPS 2025 Open Polymer Prediction](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) Kaggle competition.   The goal is to **predict multiple polymer properties** (e.g., Tg, FFV, Density, Rg, Tc) from SMILES chemical representations.
---

## 🔹 Project Highlights

- Built an **end-to-end ML pipeline** for multi-target regression with incomplete and noisy data.
- Engineered **molecular descriptors & fingerprints** (Morgan fingerprints, RDKit descriptors).
- Designed robust **feature preprocessing & selection**:
  - Dropped NaN-heavy columns
  - Variance and correlation filtering
  - Supervised feature selection
- Developed a **target imputation module** using Neural Networks and iterative imputation to handle >50% missing labels for some targets.
- Trained and ensembled **LightGBM, XGBoost** models with K-fold cross-validation.
- Achieved a **0.083 MAE leaderboard score** (~1200th place out of thousands globally).

---

## 📂 Directory Structure

```
./
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ config.yaml                  # paths, feature flags, selection top_k, thresholds
│  ├─ tuned_xgb.json               # your per-target tuned XGB params (from Optuna)
├─ data/
│  ├─ raw/                         # train.csv, test.csv, supplement files
│  ├─ interim/                     # canonical csvs, merged extras
│  └─ processed/                   # features (parquet/npy), masks, state pickles
├─ notebooks/
│  └─ polymer-lgbm-xgb.ipynb       # your original notebook (archived)
├─ scripts/
│  ├─ make_features.py             # build RDKit descriptors (+ optional graph feats)
│  ├─ train_xgb.py                 # KFold training using tuned per-target XGB
│  ├─ predict.py                   # load fold models and create submission.csv
│  └─ tune_xgb.py                  # Optuna tuning per target (optional)
└─ src/
   ├─ __init__.py
   ├─ data/
   │  ├─ __init__.py
   │  ├─ io.py                     # load/validate CSVs, canonicalize smiles
   │  └─ augment.py                # merge extra datasets into train
   ├─ featurization/
   │  ├─ __init__.py
   │  ├─ rdkit_feats.py            # descriptors, Morgan (optional), graph features
   │  └─ graphs.py                 # graph_diameter, avg_shortest_path, num_cycles
   ├─ preprocessing/
   │  ├─ __init__.py
   │  ├─ preprocessor.py           # fit/transform (variance/corr/scale) + state
   │  ├─ selection.py              # per_target_supervised_selection
   │  └─ imputation.py             # impute_targets_adaptive (NN/simple)
   ├─ models/
   │  ├─ __init__.py
   │  ├─ tuned_params.py           # loads configs/tuned_xgb.json
   │  ├─ xgb.py                    # make_xgb_model, kfold_train_xgb
   │  └─ infer.py                  # predict_and_make_submission_xgb (and generic)
   ├─ tuning/
   │  ├─ tuning_xgb.py             # tune_xgb_mae
   ├─ viz/
   │  └─ target_dist.py                  # plot_corr, target_corr, distributions, overlays
   └─ utils/
      ├─ __init__.py
      ├─ metrics.py
      ├─ logging.py
      └─ seed.py
```

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/polymer-prediction.git
   cd polymer-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Kaggle competition data:**
   ```bash
   kaggle competitions download -c neurips-open-polymer-prediction-2025 -p data/raw
   unzip data/raw/*.zip -d data/raw
   ```

4. **Run preprocessing and training:**
   ```bash
   python src/data_preprocessing.py
   python src/models.py
   ```

5. **Generate a submission file:**
   ```bash
   python src/models.py --predict-test --output submissions/submission.csv
   ```

---

## 📦 Requirements

- Python 3.9+
- Pandas, NumPy, scikit-learn
- LightGBM, XGBoost
- RDKit (for chemical feature engineering)
- PyTorch / TensorFlow (for NN imputation)

See [requirements.txt](requirements.txt) for the full list.
## Quickstart
```bash
pip install -r requirements.txt

# edit configs/config.yaml paths (train/test/supplements)
python scripts/make_features.py --config configs/config.yaml
python scripts/train_xgb.py --config configs/config.yaml
python scripts/predict.py --config configs/config.yaml -o submission.csv
---

## 🏆 Kaggle Results

- **Leaderboard Score:** 0.083 (multi-target MAE)
- **Ranking:** ~1200 out of thousands globally

---

## 🔗 References

- [Kaggle competition page](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)
- [RDKit: Open-source cheminformatics](https://www.rdkit.org/)
