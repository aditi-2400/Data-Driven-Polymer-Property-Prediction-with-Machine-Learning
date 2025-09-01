import argparse, json
from pathlib import Path
import yaml, pandas as pd

from src.data.io import load_train_test, canonicalize_smiles
from src.data.augment import merge_supplements
from src.featurization.rdkit_feats import build_features
from src.preprocessing.preprocessor import fit_preprocessor, transform_preprocessor, save_state
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger = get_logger("make_features")
    set_global_seed(cfg.get("random_seed", 42))
    paths = cfg["paths"]
    smiles_col = cfg.get("smiles_col", "SMILES")

    train, test = load_train_test(paths["train_csv"], paths["test_csv"])
    train = canonicalize_smiles(train, smiles_col)
    test  = canonicalize_smiles(test, smiles_col)

    if paths.get("supplements"):
        train = merge_supplements(train, paths["supplements"])

    # RDKit features
    feat_train, desc_names, fp_names = build_features(train.rename(columns={smiles_col:"SMILES"}),
                                                      cfg["features"])
    feat_test,  _, _ = build_features(test.rename(columns={smiles_col:"SMILES"}), cfg["features"])

    # Align to original index shapes (invalid SMILES become zeros)
    feat_train = feat_train.reindex(train.index, fill_value=0.0)
    feat_test  = feat_test.reindex(test.index,  fill_value=0.0)

    # Preprocess
    X_train_clean, state, info = fit_preprocessor(feat_train, **cfg["preprocessing"])
    X_test_clean = transform_preprocessor(feat_test, state)

    # Save artifacts
    proc = Path(paths["processed_dir"]); proc.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_train_clean).to_parquet(paths["features_file"] + ".train.parquet", index=False)
    pd.DataFrame(X_test_clean).to_parquet(paths["features_file"] + ".test.parquet", index=False)
    save_state(state, paths["preprocessor_state"])
    meta = {"desc_names": desc_names, "fp_names": fp_names, "kept_cols": state.get("kept_cols", [])}
    with open(paths["features_meta"], "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Features built & preprocessed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)