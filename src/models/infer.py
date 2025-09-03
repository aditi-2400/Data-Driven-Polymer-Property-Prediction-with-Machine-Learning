from typing import Dict, List
import numpy as np
import pandas as pd

def predict_and_make_submission_xgb(fold_models, Xsel_test, target_names, test_df, path="submission.csv"):
    n_test = next(iter(Xsel_test.values())).shape[0]
    K = len(target_names)
    preds = np.zeros((n_test, K), dtype=float)

    for j, tname in enumerate(target_names):
        X_te = Xsel_test[tname]
        models = fold_models[tname]
        if not models:
            continue
        fold_sum = np.zeros(n_test, dtype=float)
        for mdl in models:
            fold_sum += mdl.predict(X_te)
        preds[:, j] = fold_sum / len(models)

    sub = pd.DataFrame(preds, columns=target_names)
    if "id" in test_df.columns:
        sub.insert(0, "id", test_df["id"].values)
    sub.to_csv(path, index=False)
    return sub, preds
