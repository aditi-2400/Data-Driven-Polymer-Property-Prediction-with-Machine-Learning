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
- Trained and ensembled **LightGBM, XGBoost, Random Forest, and Extra Trees** models with K-fold cross-validation.
- Achieved a **0.307 MAE leaderboard score** (~1400th place out of thousands globally).

---

## 📂 Directory Structure

```
polymer-prediction/
├── data/               # Train/test CSVs (Kaggle restricted, not included)
├── notebooks/          # Jupyter notebooks for EDA, feature engineering, etc.
├── src/                # Core Python modules
├── submissions/        # Submission CSVs
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

---

## 🏆 Kaggle Results

- **Leaderboard Score:** 0.083 (multi-target MAE)
- **Ranking:** ~1200 out of thousands globally

---

## 🔗 References

- [Kaggle competition page](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)
- [RDKit: Open-source cheminformatics](https://www.rdkit.org/)
