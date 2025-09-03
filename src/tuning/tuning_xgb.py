import optuna
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold

def tune_xgb_mae(
    X, y,
    n_trials=40,
    n_splits=3,
    random_state=42,
    early_stopping_rounds=200,
    base_estimators=4000,  # larger; ES picks best_ntree_limit
):
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        params = {
            "n_estimators": base_estimators,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1e-1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "objective": "reg:absoluteerror",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }

        maes = []
        for fold_id, (tr, va) in enumerate(kf.split(X), 1):
            X_tr, X_va = X[tr], X[va]
            y_tr, y_va = y[tr], y[va]

            model = XGBRegressor(**params)
            try:
                from xgboost.callback import EarlyStopping
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)], eval_metric="mae",
                    callbacks=[EarlyStopping(rounds=early_stopping_rounds, save_best=True, maximize=False)],
                    verbose=False,
                )
            except Exception:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)], eval_metric="mae",
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )

            y_hat = model.predict(X_va)
            mae = mean_absolute_error(y_va, y_hat)
            maes.append(mae)
            trial.report(mae, step=fold_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params

    # Optional: fit a final ref model (captures best_ntree_limit internally)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=random_state)
    final = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                         random_state=random_state, n_jobs=-1, verbosity=0, **best)
    try:
        from xgboost.callback import EarlyStopping
        final.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)], eval_metric="mae",
            callbacks=[EarlyStopping(rounds=early_stopping_rounds, save_best=True, maximize=False)],
            verbose=False,
        )
    except Exception:
        final.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)], eval_metric="mae",
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

    return best, study.best_value, final