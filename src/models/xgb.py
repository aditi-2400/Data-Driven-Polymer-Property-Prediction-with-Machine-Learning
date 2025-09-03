from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

def make_xgb_model(params: Dict, random_state: int = 42) -> XGBRegressor:
    p = params.copy()
    if "max_depth" in p: p["max_depth"] = int(p["max_depth"])
    return XGBRegressor(
        n_estimators=4000,
        objective="reg:absoluteerror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        **p
    )

def kfold_train_xgb_per_target(
    Xsel_train: Dict[str, np.ndarray],
    y_imputed: np.ndarray,
    targets: List[str],
    tuned_params: Dict[str, Dict],
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, float], Dict[str, List], List[int]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = None
    n = y_imputed.shape[0]
    idx_all = np.arange(n)
    scores = {}
    fold_models = {t: [] for t in targets}
    oof = np.full((n, len(targets)), np.nan)

    for j, t in enumerate(targets):
        X = Xsel_train[t]; y = y_imputed[:, j]
        if folds is None: folds = list(kf.split(idx_all))
        print(f"\n==== Training XGB for {t} ====")
        for fold_id, (tr, va) in enumerate(folds):
            mdl = make_xgb_model(tuned_params.get(t, {}), random_state=random_state)
            try:
                from xgboost.callback import EarlyStopping
                mdl.fit(
                    X[tr], y[tr],
                    eval_set=[(X[va], y[va])],
                    eval_metric="mae",
                    callbacks=[EarlyStopping(rounds=200, save_best=True, maximize=False)],
                    verbose=False,
                )
            except Exception:
                mdl.fit(
                    X[tr], y[tr],
                    eval_set=[(X[va], y[va])],
                    eval_metric="mae",
                    early_stopping_rounds=200,
                    verbose=False,
                )
            oof[va, j] = mdl.predict(X[va])
            fold_models[t].append(mdl)
        mask = np.isfinite(oof[:, j])
        scores[t] = float(mean_absolute_error(y[mask], oof[mask, j]))
        print(f"{t} OOF MAE = {scores[t]:.4f}")
    print(f"Mean OOF MAE = {float(np.mean(list(scores.values()))):.4f}")
    return scores, fold_models, [len(m) for m in fold_models.values()]