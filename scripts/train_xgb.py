import argparse, json
from pathlib import Path
import yaml, pandas as pd, numpy as np
from joblib import dump

from src.preprocessing.selection import per_target_supervised_selection
from src.preprocessing.imputation import impute_targets_adaptive
from src.models.xgb import kfold_train_xgb_per_target
from src.models.tuned_params import load_tuned_params
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger = get_logger("train_xgb")
    set_global_seed(cfg.get("random_seed", 42))

    paths = cfg["paths"]
    targets = cfg["targets"]
    df_train = pd.read_csv(paths["train_csv"])
    y_train_raw = df_train[targets].to_numpy(dtype=float)

    X_train = pd.read_parquet(paths["features_file"] + ".train.parquet").values

    # Supervised selection (per target)
    sel_cfg = cfg.get("selection", {})
    selectors, Xsel_train = per_target_supervised_selection(
        X_train, y_train_raw, targets,
        min_labels=sel_cfg.get("min_labels", 500),
        top_k=sel_cfg.get("top_k_per_target", 60),
        cv_splits=5, verbose=True
    )

    # Adaptive imputation of targets
    imp_cfg = cfg.get("imputation", {})
    y_imputed, _ = impute_targets_adaptive(
        X=None, y=y_train_raw, target_names=targets,
        X_per_target=Xsel_train,
        min_missing_for_nn=imp_cfg.get("min_missing_for_nn", 30),
        min_known_for_nn=imp_cfg.get("min_known_for_nn", 50),
        simple_strategy=imp_cfg.get("simple_strategy", "median")
    )

    # Tuned params
    tuned = load_tuned_params("configs/tuned_xgb.json")

    # Train per target with CV
    scores, fold_models, _ = kfold_train_xgb_per_target(
        Xsel_train, y_imputed, targets, tuned,
        n_splits=cfg["training"]["n_splits"],
        random_state=cfg.get("random_seed", 42)
    )

    # Persist
    models_dir = Path(paths["models_dir"]); models_dir.mkdir(parents=True, exist_ok=True)
    for t, models in fold_models.items():
        tdir = models_dir / t; tdir.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(models):
            dump(m, tdir / f"fold{i}.joblib")

    with open(Path(paths["processed_dir"]) / "selectors.json", "w") as f:
        json.dump({k: list(map(int, v)) for k, v in selectors.items()}, f, indent=2)
    with open(Path(paths["processed_dir"]) / "cv_scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    logger.info("Training complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
