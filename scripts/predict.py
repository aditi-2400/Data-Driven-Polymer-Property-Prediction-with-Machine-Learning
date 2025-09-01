import argparse, json
from pathlib import Path
import yaml, numpy as np, pandas as pd
from joblib import load

from src.models.infer import predict_and_make_submission_xgb
from src.utils.logging import get_logger

def main(cfg_path: str, out_csv: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger = get_logger("predict")

    paths = cfg["paths"]; targets = cfg["targets"]

    X_test = pd.read_parquet(paths["features_file"] + ".test.parquet").values

    # Load selectors
    with open(Path(paths["processed_dir"]) / "selectors.json", "r") as f:
        selectors = {k: np.array(v, dtype=int) for k, v in json.load(f).items()}

    Xsel_test = {t: X_test[:, selectors[t]] for t in targets}

    # Load CV fold models per target
    models_root = Path(paths["models_dir"])
    fold_models = {}
    for t in targets:
        files = sorted((models_root / t).glob("fold*.joblib"))
        fold_models[t] = [load(p) for p in files]

    test_df = pd.read_csv(paths["test_csv"])
    sub, _ = predict_and_make_submission_xgb(fold_models, Xsel_test, targets, test_df, path=out_csv)
    logger.info(f"Wrote {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--output", default="submission.csv")
    args = ap.parse_args()
    main(args.config, args.output)
