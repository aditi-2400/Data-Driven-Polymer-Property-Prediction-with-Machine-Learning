# src/preprocessing/imputation.py
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def impute_targets_adaptive(
    X,                      # np.ndarray (shared features) OR None if using X_per_target
    y,                      # np.ndarray shape (n_samples, n_targets) with NaNs
    target_names,           # list[str], e.g., ['Tg','FFV','Tc','Density','Rg']
    selectors=None,         # optional: dict[name] -> np.ndarray[int] (column indices into X)
    X_per_target=None,      # optional: dict[name] -> np.ndarray (feature matrix per target)
    min_missing_for_nn=30,  # use NN only if number of NaNs in that target > this
    min_known_for_nn=50,    # and at least this many known labels to train the NN
    epochs=100, batch_size=64, verbose=0,
    simple_strategy="median",   # 'median' or 'mean'
    random_state=42
):
    """
    Impute missing target values per target.
    - If missing count > min_missing_for_nn AND known count >= min_known_for_nn -> use a small MLP (Keras).
    - Else -> use SimpleImputer on the known labels (median/mean).
    - Supports per-target feature subsets via `selectors` or `X_per_target`.
    
    Returns:
        y_imp       : np.ndarray with missing targets imputed
        impute_info : dict[name] -> ('nn', model) or ('simple', value or imputer)
    """
    y_imp = y.copy().astype(float)
    impute_info = {}

    # Helper to get the design matrix for a given target
    def get_X_for(name):
        if X_per_target is not None:
            return X_per_target[name]
        if selectors is not None and X is not None:
            idx = selectors.get(name, None)
            return X[:, idx] if idx is not None else X
        # default: shared X
        return X

    # Build MLP lazily and safely
    def build_mlp(
        input_dim: int,
        hidden_units=(256, 128),
        dropout_rate: float = 0.30,     # try 0.2–0.5
        l2: float = 1e-4,               # try 1e-5–1e-3
        use_batchnorm: bool = True,
        lr: float = 1e-3,
        loss: str = "mse"               # or "huber" for robustness
    ):
        """
        MLP with L2 regularization + Dropout (and optional BatchNorm).
        """
        from tensorflow.keras import models, layers, optimizers, regularizers, initializers
    
        reg  = regularizers.l2(l2) if l2 and l2 > 0 else None
        init = initializers.HeNormal()
    
        m = models.Sequential()
        m.add(layers.Input(shape=(input_dim,)))
    
        for units in hidden_units:
            m.add(layers.Dense(units,
                               activation="relu",
                               kernel_initializer=init,
                               kernel_regularizer=reg))
            if use_batchnorm:
                m.add(layers.BatchNormalization())
            if dropout_rate and dropout_rate > 0:
                m.add(layers.Dropout(dropout_rate))
    
        # Output layer (keep small L2 to stabilize if desired)
        m.add(layers.Dense(1, kernel_regularizer=reg))
    
        # Consider Huber loss for outlier-robust imputation:
        # from tensorflow.keras.losses import Huber
        # loss = Huber(delta=1.0)
    
        m.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=loss)
        return m

    rng = np.random.RandomState(random_state)
    n_targets = y.shape[1]

    for j, name in enumerate(target_names):
        y_col = y[:, j]
        miss_mask = np.isnan(y_col)
        n_miss = int(miss_mask.sum())
        known_mask = ~miss_mask
        n_known = int(known_mask.sum())

        if n_miss == 0:
            impute_info[name] = ('none', None)
            # nothing to fill
            continue

        Xj = get_X_for(name)
        if Xj is None:
            # Safety: if we couldn't get features, fall back to simple imputer
            print(f"[Impute-{name}] No features provided; using SimpleImputer({simple_strategy}).")
            simp = SimpleImputer(strategy=simple_strategy)
            y_known_2d = y_col[known_mask].reshape(-1, 1)
            simp.fit(y_known_2d)
            y_imp[miss_mask, j] = simp.transform(np.full((n_miss, 1), np.nan))[:, 0]
            impute_info[name] = ('simple', simp)
            continue

        # Decide method
        use_nn = (n_miss > min_missing_for_nn) and (n_known >= min_known_for_nn)

        if not use_nn:
            # Simple fill from known labels statistics
            try:
                simp = SimpleImputer(strategy=simple_strategy)
                y_known_2d = y_col[known_mask].reshape(-1, 1)
                simp.fit(y_known_2d)
                y_imp[miss_mask, j] = simp.transform(np.full((n_miss, 1), np.nan))[:, 0]
                impute_info[name] = ('simple', simp)
                print(f"[Impute-{name}] SimpleImputer({simple_strategy}) used | known={n_known}, missing={n_miss}")
            except Exception as e:
                # Final fallback: manual median
                val = np.nanmedian(y_col)
                y_imp[miss_mask, j] = val
                impute_info[name] = ('simple_fallback', val)
                print(f"[Impute-{name}] SimpleImputer failed ({e}); filled with median={val:.4f}")
            continue

        # NN path
        try:
            X_known, y_known = Xj[known_mask], y_col[known_mask]
            # Small validation split
            Xtr, Xva, ytr, yva = train_test_split(
                X_known, y_known, test_size=0.2, random_state=random_state
            )

            model = build_mlp(Xj.shape[1])
            model.fit(Xtr, ytr, validation_data=(Xva, yva),
                      batch_size=batch_size, epochs=epochs, verbose=verbose)

            # Predict missing
            X_miss = Xj[miss_mask]
            y_pred = model.predict(X_miss, verbose=0).ravel()
            y_imp[miss_mask, j] = y_pred
            impute_info[name] = ('nn', model)
            print(f"[Impute-{name}] NN used | known={n_known}, missing={n_miss}")

        except Exception as e:
            # Robust fallback: SimpleImputer on labels
            try:
                simp = SimpleImputer(strategy=simple_strategy)
                y_known_2d = y_col[known_mask].reshape(-1, 1)
                simp.fit(y_known_2d)
                y_imp[miss_mask, j] = simp.transform(np.full((n_miss, 1), np.nan))[:, 0]
                impute_info[name] = ('simple_after_nn_fail', simp)
                print(f"[Impute-{name}] NN failed ({e}); used SimpleImputer({simple_strategy}).")
            except Exception as ee:
                val = np.nanmedian(y_col)
                y_imp[miss_mask, j] = val
                impute_info[name] = ('simple_fallback', val)
                print(f"[Impute-{name}] Both NN and SimpleImputer failed ({ee}); median={val:.4f}.")

    return y_imp, impute_info
