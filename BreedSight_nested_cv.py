# -*- coding: utf-8 -*-
"""
BreedSight – Nested K-Fold Cross-Validation for Hyperparameter Tuning
=====================================================================
Strictly no data leakage:
  - Outer loop: unbiased performance estimation
  - Inner loop: hyperparameter selection only (never sees outer validation fold)
  - Feature construction (GRM / PCA / scaling) fitted exclusively on the
    current training partition of each fold
  - Early stopping uses an *inner* split of the training data only
  - Final model is refit on the full training set using the best
    hyperparameters found by nested CV; the independent test set is
    touched only once, after all selection is complete

Author: Ashmitha
"""

import os
import warnings
from itertools import product
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
RANDOM_STATE = 90
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ------------------------------------------------------------------
# Default hyper-parameter search space (inner loop)
# Keep the grid modest – each combination trains a DNN + RF.
# ------------------------------------------------------------------
DEFAULT_PARAM_GRID = {
    "learning_rate": [1e-3, 5e-4],
    "l2_reg": [0.01, 0.001],
    "dropout_rate": [0.2, 0.3],
    "rf_n_estimators": [200, 400],
    "rf_max_depth": [10, None],
    "alpha": [0.4, 0.5, 0.6],          # ensemble weight: DNN * alpha + RF * (1-alpha)
}

# Fixed architectural / training choices that are *not* tuned
FIXED = {
    "epochs": 300,                     # upper bound; early stopping decides actual length
    "batch_size": 32,
    "es_patience": 15,
    "es_val_fraction": 0.15,           # fraction of *training* fold used for early stopping
    "n_components": 50,                # PCA components on [GRM | GRM²]
    "max_markers": 10000,              # optional random marker sub-sampling for GRM
}


# ==================================================================
# Utility helpers
# ==================================================================
def set_seeds(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) == 0 or len(np.unique(y_pred)) <= 1:
        return {"r2": np.nan, "pearson_r": np.nan, "rmse": np.nan, "mse": np.nan}
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    return {"r2": float(r2), "pearson_r": float(r), "rmse": float(np.sqrt(mse)), "mse": float(mse)}


def check_no_leakage(train_ids, val_ids, test_ids=None) -> bool:
    train_set, val_set = set(train_ids), set(val_ids)
    ok = True
    if train_set & val_set:
        print("CRITICAL: overlap between train and validation IDs")
        ok = False
    if test_ids is not None:
        test_set = set(test_ids)
        if train_set & test_set or val_set & test_set:
            print("CRITICAL: overlap with test IDs")
            ok = False
    if ok:
        print("  No ID overlap detected")
    return ok


# ==================================================================
# Genomic feature construction (GRM + optional PCA)
# All statistics are computed on the *training* partition only.
# ==================================================================
def compute_grm_features(
    X_train: np.ndarray,
    X_other: Optional[np.ndarray] = None,
    ref: Optional[Dict] = None,
    is_train: bool = True,
    n_components: int = 50,
    max_markers: int = 10000,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Build [G | G²] features (optionally reduced by PCA).

    Parameters
    ----------
    X_train : array, shape (n_train, n_markers)
        Genotypes of the current training partition.
    X_other : array or None
        Genotypes of validation / test samples.
    ref : dict or None
        Reference statistics produced when is_train=True.
    is_train : bool
        If True, fit means / PCA on X_train and return the ref dict.
    """
    if np.isnan(X_train).any():
        raise ValueError("Training genotypes contain NaNs")
    if X_other is not None and np.isnan(X_other).any():
        raise ValueError("Other genotypes contain NaNs")

    # Optional random marker sub-sampling (reproducible)
    if is_train and X_train.shape[1] > max_markers:
        rng = np.random.RandomState(RANDOM_STATE)
        sel = rng.choice(X_train.shape[1], max_markers, replace=False)
        X_train = X_train[:, sel]
        if X_other is not None:
            X_other = X_other[:, sel]
        selected_markers = sel
    elif not is_train and ref is not None and ref.get("selected_markers") is not None:
        sel = ref["selected_markers"]
        X_train = X_train[:, sel]
        if X_other is not None:
            X_other = X_other[:, sel]
        selected_markers = sel
    else:
        selected_markers = None

    if is_train:
        marker_means = X_train.mean(axis=0)
        Xc = X_train - marker_means
        n_markers = Xc.shape[1]
        G = (Xc @ Xc.T) / n_markers
        I = G * G
        mean_diag = np.mean(np.diag(I))
        I_norm = I / mean_diag if mean_diag != 0 else I
        X_final = np.concatenate([G, I_norm], axis=1)

        X_other_final = None
        if X_other is not None:
            Xo = X_other - marker_means
            Go = (Xo @ Xc.T) / n_markers
            Io = Go * Go
            Io_norm = Io / mean_diag if mean_diag != 0 else Io
            X_other_final = np.concatenate([Go, Io_norm], axis=1)

        pca = PCA(n_components=min(n_components, X_final.shape[1]), random_state=RANDOM_STATE)
        X_final = pca.fit_transform(X_final)
        if X_other_final is not None:
            X_other_final = pca.transform(X_other_final)

        ref_out = {
            "marker_means": marker_means,
            "mean_diag": mean_diag,
            "n_markers": n_markers,
            "selected_markers": selected_markers,
            "X_train_centered": Xc,          # needed for correct test projection
            "pca": pca,
        }
        return X_final, X_other_final, ref_out

    # ----- projection of new samples onto training GRM space -----
    if ref is None or "X_train_centered" not in ref:
        raise ValueError("ref must contain training-centred genotypes")
    Xc_new = X_train - ref["marker_means"]          # note: X_train here is the “other” matrix
    G_new = (Xc_new @ ref["X_train_centered"].T) / ref["n_markers"]
    I_new = G_new * G_new
    I_norm = I_new / ref["mean_diag"] if ref["mean_diag"] != 0 else I_new
    X_final = np.concatenate([G_new, I_norm], axis=1)
    if "pca" in ref:
        X_final = ref["pca"].transform(X_final)
    return X_final, None, ref


# ==================================================================
# BreedSight model (DNN + RF ensemble)
# ==================================================================
def build_dnn(input_dim: int, learning_rate: float, l2_reg: float, dropout_rate: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = Dense(256, kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Dense(128, kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Dense(64, kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = LeakyReLU(alpha=0.1)(x)

    outputs = Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss=tf.keras.losses.Huber(delta=1.0),
        optimizer=Adam(learning_rate=learning_rate, clipvalue=0.5),
        metrics=["mse"],
    )
    return model


def fit_breedsight(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    verbose: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit BreedSight on a pure training matrix X, y.
    Early-stopping validation is carved *from X* (never from an outer fold).

    Returns
    -------
    train_pred : predictions on the whole of X
    artifacts  : dict containing scaler, dnn, rf, history
    """
    set_seeds()
    n = X.shape[0]
    es_frac = params.get("es_val_fraction", FIXED["es_val_fraction"])
    use_es = params.get("use_early_stopping", True)

    if use_es and n > 20:
        fit_idx, es_idx = train_test_split(
            np.arange(n), test_size=es_frac, random_state=RANDOM_STATE, shuffle=True
        )
    else:
        fit_idx, es_idx = np.arange(n), None

    # Scaler fitted only on the inner-training rows
    scaler = StandardScaler()
    scaler.fit(X[fit_idx])
    X_scaled = scaler.transform(X)

    dnn = build_dnn(
        input_dim=X.shape[1],
        learning_rate=params["learning_rate"],
        l2_reg=params["l2_reg"],
        dropout_rate=params["dropout_rate"],
    )

    callbacks = []
    val_data = None
    if es_idx is not None:
        val_data = (X_scaled[es_idx], y[es_idx])
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=params.get("es_patience", FIXED["es_patience"]),
                restore_best_weights=True,
                verbose=verbose,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=verbose,
            ),
        ]

    history = dnn.fit(
        X_scaled[fit_idx],
        y[fit_idx],
        epochs=params.get("epochs", FIXED["epochs"]),
        batch_size=params.get("batch_size", FIXED["batch_size"]),
        validation_data=val_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True,
    )

    rf = RandomForestRegressor(
        n_estimators=params["rf_n_estimators"],
        max_depth=params["rf_max_depth"],
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X, y.ravel())

    # Predictions on the whole training matrix (for diagnostics)
    pred_dnn = dnn.predict(X_scaled, verbose=0).ravel()
    pred_rf = rf.predict(X)
    alpha = params["alpha"]
    train_pred = alpha * pred_dnn + (1.0 - alpha) * pred_rf

    artifacts = {
        "scaler": scaler,
        "dnn": dnn,
        "rf": rf,
        "history": history,
        "params": params,
    }
    return train_pred, artifacts


def predict_breedsight(X: np.ndarray, artifacts: Dict[str, Any]) -> np.ndarray:
    scaler = artifacts["scaler"]
    dnn = artifacts["dnn"]
    rf = artifacts["rf"]
    alpha = artifacts["params"]["alpha"]
    X_scaled = scaler.transform(X)
    pred_dnn = dnn.predict(X_scaled, verbose=0).ravel()
    pred_rf = rf.predict(X)
    return alpha * pred_dnn + (1.0 - alpha) * pred_rf


# ==================================================================
# Nested cross-validation
# ==================================================================
def expand_grid(param_grid: Dict[str, List]) -> List[Dict]:
    keys = list(param_grid.keys())
    combos = []
    for values in product(*[param_grid[k] for k in keys]):
        combos.append(dict(zip(keys, values)))
    return combos


def nested_kfold_breedsight(
    Z: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    outer_n_splits: int = 5,
    inner_n_splits: int = 3,
    param_grid: Optional[Dict] = None,
    n_components: int = FIXED["n_components"],
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Nested K-fold CV for BreedSight.

    Outer loop  → unbiased performance estimate
    Inner loop  → hyper-parameter selection (never sees the outer validation fold)

    All genomic feature construction is performed inside each training partition.
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID
    candidates = expand_grid(param_grid)
    if verbose:
        print(f"Hyper-parameter grid size: {len(candidates)} combinations")
        print(f"Outer folds: {outer_n_splits}  |  Inner folds: {inner_n_splits}")

    outer_kf = KFold(n_splits=outer_n_splits, shuffle=True, random_state=RANDOM_STATE)
    outer_results = []
    all_val_preds = []
    best_params_per_outer = []

    for outer_fold, (tr_idx, va_idx) in enumerate(outer_kf.split(Z), 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"OUTER FOLD {outer_fold}/{outer_n_splits}")
            print(f"{'='*60}")

        Z_tr, Z_va = Z[tr_idx], Z[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        ids_tr, ids_va = sample_ids[tr_idx], sample_ids[va_idx]
        check_no_leakage(ids_tr, ids_va)

        # ----------------------------------------------------------
        # Inner loop – hyper-parameter search on the outer training set
        # ----------------------------------------------------------
        inner_kf = KFold(n_splits=inner_n_splits, shuffle=True, random_state=RANDOM_STATE)
        candidate_scores = {i: [] for i in range(len(candidates))}

        for inner_fold, (in_tr, in_va) in enumerate(inner_kf.split(Z_tr), 1):
            if verbose > 1:
                print(f"  Inner fold {inner_fold}/{inner_n_splits}")

            Z_in_tr, Z_in_va = Z_tr[in_tr], Z_tr[in_va]
            y_in_tr, y_in_va = y_tr[in_tr], y_tr[in_va]

            # Feature construction *only* on the inner-training partition
            X_in_tr, X_in_va, _ = compute_grm_features(
                Z_in_tr, Z_in_va, is_train=True, n_components=n_components
            )

            for c_idx, params in enumerate(candidates):
                # Merge fixed settings
                full_params = {**FIXED, **params, "use_early_stopping": True}
                try:
                    _, art = fit_breedsight(X_in_tr, y_in_tr, full_params, verbose=0)
                    pred = predict_breedsight(X_in_va, art)
                    m = calculate_metrics(y_in_va, pred)
                    candidate_scores[c_idx].append(m["r2"])
                except Exception as e:
                    if verbose > 1:
                        print(f"    Candidate {c_idx} failed: {e}")
                    candidate_scores[c_idx].append(np.nan)

        # Select best hyper-parameters by mean inner validation R²
        mean_scores = [
            np.nanmean(candidate_scores[i]) if candidate_scores[i] else -np.inf
            for i in range(len(candidates))
        ]
        best_idx = int(np.nanargmax(mean_scores))
        best_params = candidates[best_idx]
        best_params_per_outer.append(best_params)

        if verbose:
            print(f"  Best params (outer fold {outer_fold}): {best_params}")
            print(f"  Mean inner Val R²: {mean_scores[best_idx]:.4f}")

        # ----------------------------------------------------------
        # Refit on the *entire* outer training set with the selected params
        # ----------------------------------------------------------
        X_tr, X_va, ref = compute_grm_features(
            Z_tr, Z_va, is_train=True, n_components=n_components
        )
        full_params = {**FIXED, **best_params, "use_early_stopping": True}
        train_pred, artifacts = fit_breedsight(X_tr, y_tr, full_params, verbose=0)
        val_pred = predict_breedsight(X_va, artifacts)

        train_m = calculate_metrics(y_tr, train_pred)
        val_m = calculate_metrics(y_va, val_pred)

        outer_results.append({
            "outer_fold": outer_fold,
            "best_params": best_params,
            "train_r2": train_m["r2"],
            "val_r2": val_m["r2"],
            "train_pearson": train_m["pearson_r"],
            "val_pearson": val_m["pearson_r"],
            "train_rmse": train_m["rmse"],
            "val_rmse": val_m["rmse"],
            "n_train": len(y_tr),
            "n_val": len(y_va),
        })

        all_val_preds.append(pd.DataFrame({
            "Sample_ID": ids_va,
            "True_Phenotype": y_va,
            "Predicted": val_pred,
            "Outer_Fold": outer_fold,
        }))

        if verbose:
            print(f"  Outer fold {outer_fold}  Train R²={train_m['r2']:.4f}  "
                  f"Val R²={val_m['r2']:.4f}  Val r={val_m['pearson_r']:.4f}")

    results_df = pd.DataFrame(outer_results)
    val_pred_df = pd.concat(all_val_preds, ignore_index=True)

    summary = {
        "mean_val_r2": results_df["val_r2"].mean(),
        "std_val_r2": results_df["val_r2"].std(),
        "mean_val_pearson": results_df["val_pearson"].mean(),
        "std_val_pearson": results_df["val_pearson"].std(),
        "mean_train_r2": results_df["train_r2"].mean(),
        "n_outer_folds": outer_n_splits,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("NESTED CV SUMMARY (unbiased performance estimate)")
        print("=" * 60)
        print(f"  Mean Val R²     : {summary['mean_val_r2']:.4f} ± {summary['std_val_r2']:.4f}")
        print(f"  Mean Val Pearson: {summary['mean_val_pearson']:.4f} ± {summary['std_val_pearson']:.4f}")
        print(f"  Mean Train R²   : {summary['mean_train_r2']:.4f}")

    return {
        "outer_results": results_df,
        "val_predictions": val_pred_df,
        "best_params_per_outer": best_params_per_outer,
        "summary": summary,
        "candidates": candidates,
    }


# ==================================================================
# Final model training (after nested CV)
# ==================================================================
def train_final_breedsight(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test: np.ndarray,
    best_params: Dict[str, Any],
    n_components: int = FIXED["n_components"],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Train the final BreedSight model on the *full* training set
    using the hyper-parameters selected by nested CV.
    The test set is used only for prediction (never for fitting or selection).
    """
    print("\nTraining final BreedSight model on full training data …")
    X_train, X_test, ref = compute_grm_features(
        Z_train, Z_test, is_train=True, n_components=n_components
    )
    full_params = {**FIXED, **best_params, "use_early_stopping": True}
    _, artifacts = fit_breedsight(X_train, y_train, full_params, verbose=1)
    test_pred = predict_breedsight(X_test, artifacts)
    artifacts["ref"] = ref          # keep for possible future projections
    return test_pred, artifacts


def choose_final_params(nested_result: Dict) -> Dict[str, Any]:
    """
    Simple consensus: pick the parameter set that appeared most often
    as the best in the outer folds.  Ties are broken by the highest
    mean outer-fold validation R² of that parameter set.
    """
    from collections import Counter
    # Convert dicts to hashable tuples
    tuples = [tuple(sorted(p.items())) for p in nested_result["best_params_per_outer"]]
    counts = Counter(tuples)
    most_common = counts.most_common()
    # Among the most frequent, prefer the one with highest average outer val R²
    best_tuple = None
    best_score = -np.inf
    for t, cnt in most_common:
        if cnt < most_common[0][1]:
            break
        # average val R² of the outer folds that selected this set
        mask = [tuple(sorted(p.items())) == t for p in nested_result["best_params_per_outer"]]
        avg_r2 = nested_result["outer_results"].loc[mask, "val_r2"].mean()
        if avg_r2 > best_score:
            best_score = avg_r2
            best_tuple = t
    return dict(best_tuple)


# ==================================================================
# Data loading & validation
# ==================================================================
def load_and_validate(
    train_pheno_path: str,
    train_geno_path: str,
    test_pheno_path: str,
    test_geno_path: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    train_pheno = pd.read_csv(train_pheno_path)
    train_geno = pd.read_csv(train_geno_path)
    test_pheno = pd.read_csv(test_pheno_path)
    test_geno = pd.read_csv(test_geno_path)

    print("Data shapes:")
    print(f"  Train pheno : {train_pheno.shape}")
    print(f"  Train geno  : {train_geno.shape}")
    print(f"  Test  pheno : {test_pheno.shape}")
    print(f"  Test  geno  : {test_geno.shape}")

    if "phenotypes" not in train_pheno.columns:
        raise ValueError("Training phenotype file must contain a 'phenotypes' column")

    # Basic missing-value checks
    if train_pheno["phenotypes"].isnull().any():
        raise ValueError("Missing phenotypes in training data")
    if train_geno.isnull().any().any() or test_geno.isnull().any().any():
        raise ValueError("Missing genotypes detected – impute or remove first")

    # Align markers
    train_markers = set(train_geno.columns[1:])
    test_markers = set(test_geno.columns[1:])
    common = sorted(train_markers & test_markers)
    if not common:
        raise ValueError("No common markers between train and test")
    if len(common) < len(train_markers) or len(common) < len(test_markers):
        print(f"  Aligning to {len(common)} common markers")
    train_geno = train_geno[["sample_id"] + common] if "sample_id" in train_geno.columns else train_geno.iloc[:, [0] + [train_geno.columns.get_loc(m) for m in common]]
    test_geno = test_geno[["sample_id"] + common] if "sample_id" in test_geno.columns else test_geno.iloc[:, [0] + [test_geno.columns.get_loc(m) for m in common]]

    # Remove zero-variance markers (computed on training only)
    geno_mat = train_geno.iloc[:, 1:].values
    var = np.var(geno_mat, axis=0)
    keep = var > 0
    if not keep.all():
        print(f"  Removing {(~keep).sum()} zero-variance markers")
        cols = [train_geno.columns[0]] + list(np.array(train_geno.columns[1:])[keep])
        train_geno = train_geno[cols]
        test_geno = test_geno[cols]

    Z_train = train_geno.iloc[:, 1:].values.astype(float)
    y_train = train_pheno["phenotypes"].values.astype(float)
    ids_train = train_pheno.iloc[:, 0].values

    Z_test = test_geno.iloc[:, 1:].values.astype(float)
    ids_test = test_pheno.iloc[:, 0].values
    y_test = test_pheno["phenotypes"].values.astype(float) if "phenotypes" in test_pheno.columns else None

    return train_pheno, Z_train, y_train, ids_train, test_pheno, Z_test, y_test, ids_test


# ==================================================================
# Plotting helpers
# ==================================================================
def plot_true_vs_pred(y_true, y_pred, title, path):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", lw=2, label="Perfect")
    m = calculate_metrics(y_true, y_pred)
    plt.title(f"{title}\nR²={m['r2']:.3f}  r={m['pearson_r']:.3f}")
    plt.xlabel("True phenotype")
    plt.ylabel("Predicted phenotype")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ==================================================================
# Main pipeline
# ==================================================================
def run_breedsight_nested(
    train_pheno_path: str,
    train_geno_path: str,
    test_pheno_path: str,
    test_geno_path: str,
    output_dir: str = "output_breedsight_nested",
    outer_n_splits: int = 5,
    inner_n_splits: int = 3,
    param_grid: Optional[Dict] = None,
    n_components: int = 50,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """
    Complete nested-CV pipeline for BreedSight only.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

    print("=" * 60)
    print("BreedSight – Nested K-Fold Hyper-parameter Tuning")
    print("=" * 60)

    # 1. Load & validate
    (train_pheno, Z_train, y_train, ids_train,
     test_pheno, Z_test, y_test, ids_test) = load_and_validate(
        train_pheno_path, train_geno_path, test_pheno_path, test_geno_path
    )

    # 2. Nested CV
    nested = nested_kfold_breedsight(
        Z=Z_train,
        y=y_train,
        sample_ids=ids_train,
        outer_n_splits=outer_n_splits,
        inner_n_splits=inner_n_splits,
        param_grid=param_grid,
        n_components=n_components,
        verbose=1,
    )

    # 3. Choose final hyper-parameters (consensus across outer folds)
    final_params = choose_final_params(nested)
    print(f"\nSelected final hyper-parameters: {final_params}")

    # 4. Train final model on full training data
    test_pred, final_artifacts = train_final_breedsight(
        Z_train, y_train, Z_test, final_params, n_components=n_components
    )

    # 5. Independent test metrics (if phenotypes available)
    test_metrics = None
    if y_test is not None:
        test_metrics = calculate_metrics(y_test, test_pred)
        print("\n=== INDEPENDENT TEST SET ===")
        print(f"  R²      = {test_metrics['r2']:.4f}")
        print(f"  Pearson = {test_metrics['pearson_r']:.4f}")
        print(f"  RMSE    = {test_metrics['rmse']:.4f}")

    # 6. Save everything
    nested["outer_results"].to_csv(os.path.join(output_dir, "outer_fold_results.csv"), index=False)
    nested["val_predictions"].to_csv(os.path.join(output_dir, "nested_val_predictions.csv"), index=False)

    test_df = pd.DataFrame({"Sample_ID": ids_test, "Predicted_Phenotype": test_pred})
    if y_test is not None:
        test_df["True_Phenotype"] = y_test
    test_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    # Save model (RF + scaler + params; DNN saved separately via Keras)
    light = {
        "rf": final_artifacts["rf"],
        "scaler": final_artifacts["scaler"],
        "params": final_artifacts["params"],
        "ref": final_artifacts.get("ref"),
    }
    joblib.dump(light, os.path.join(output_dir, "models", "final_breedsight_rf_scaler.pkl"))
    final_artifacts["dnn"].save(os.path.join(output_dir, "models", "final_breedsight_dnn.keras"))

    # Summary text
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("BreedSight Nested Cross-Validation Summary\n")
        f.write("==========================================\n\n")
        f.write("PROTOCOL\n")
        f.write("- Outer K-fold: unbiased performance estimation\n")
        f.write("- Inner K-fold: hyper-parameter selection only\n")
        f.write("- GRM / PCA / scaling fitted exclusively inside each training partition\n")
        f.write("- Early stopping uses an internal split of the training data\n")
        f.write("- Test set touched only once, after all selection\n\n")
        f.write(f"Outer folds : {outer_n_splits}\n")
        f.write(f"Inner folds : {inner_n_splits}\n")
        f.write(f"Grid size   : {len(nested['candidates'])}\n\n")
        f.write("NESTED-CV PERFORMANCE (mean ± sd over outer folds)\n")
        f.write(f"  Val R²     : {nested['summary']['mean_val_r2']:.4f} ± {nested['summary']['std_val_r2']:.4f}\n")
        f.write(f"  Val Pearson: {nested['summary']['mean_val_pearson']:.4f} ± {nested['summary']['std_val_pearson']:.4f}\n\n")
        f.write(f"FINAL HYPER-PARAMETERS\n{final_params}\n\n")
        if test_metrics is not None:
            f.write("INDEPENDENT TEST SET\n")
            f.write(f"  R²      : {test_metrics['r2']:.4f}\n")
            f.write(f"  Pearson : {test_metrics['pearson_r']:.4f}\n")
            f.write(f"  RMSE    : {test_metrics['rmse']:.4f}\n")

    if generate_plots and y_test is not None:
        plot_true_vs_pred(
            y_test, test_pred,
            "Test set – final BreedSight",
            os.path.join(output_dir, "plots", "test_true_vs_pred.png"),
        )

    print(f"\nAll results written to: {output_dir}")
    return {
        "nested": nested,
        "final_params": final_params,
        "test_predictions": test_pred,
        "test_metrics": test_metrics,
        "artifacts": final_artifacts,
    }


# ==================================================================
# Example entry point
# ==================================================================
if __name__ == "__main__":
    # Edit these paths for your data
    TRAIN_PHENO = "training_phenotypic_data.csv"
    TRAIN_GENO = "training_additive.csv"
    TEST_PHENO = "testing_data.csv"
    TEST_GENO = "testing_additive.csv"

    # Modest grid – expand only if you have the compute budget
    GRID = {
        "learning_rate": [1e-3, 5e-4],
        "l2_reg": [0.01, 0.001],
        "dropout_rate": [0.2, 0.3],
        "rf_n_estimators": [200, 400],
        "rf_max_depth": [10, None],
        "alpha": [0.4, 0.5, 0.6],
    }

    try:
        results = run_breedsight_nested(
            train_pheno_path=TRAIN_PHENO,
            train_geno_path=TRAIN_GENO,
            test_pheno_path=TEST_PHENO,
            test_geno_path=TEST_GENO,
            output_dir="output_breedsight_nested",
            outer_n_splits=5,
            inner_n_splits=3,
            param_grid=GRID,
            n_components=50,
            generate_plots=True,
        )
        print("\nAnalysis finished successfully.")
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
