import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

def per_target_supervised_selection(
    X_train_clean: np.ndarray,
    y_train: np.ndarray,
    target_names,
    min_labels=500,
    top_k=60,
    cv_splits=5,
    random_state=42,
    verbose=True
):
    n, p = X_train_clean.shape
    selectors, Xsel_train = {}, {}

    for j, name in enumerate(target_names):
        yj = y_train[:, j]
        mask = np.isfinite(yj)
        n_lab = int(mask.sum())
        if verbose:
            print(f"[{name}] labeled rows = {n_lab} / {n}")

        if n_lab < 5:
            selectors[name] = np.arange(p)
            Xsel_train[name] = X_train_clean
            continue

        Xk, yk = X_train_clean[mask], yj[mask]
        kf = KFold(n_splits=min(cv_splits, n_lab), shuffle=True, random_state=random_state)

        importances = np.zeros(p, dtype=float); folds = 0
        for tr, va in kf.split(Xk):
            est = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
            est.fit(Xk[tr], yk[tr])
            importances += getattr(est, "feature_importances_", np.zeros(p))
            folds += 1
        if folds > 0: importances /= folds

        idx = np.argsort(importances)[::-1][:top_k]
        if verbose: print(f"[{name}] keep {len(idx)} / {p}")
        selectors[name] = idx
        Xsel_train[name] = X_train_clean[:, idx]

    return selectors, Xsel_train
