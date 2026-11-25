# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 09:48:00 2025

@author: Ashmitha
"""


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, t, bootstrap
from sklearn.decomposition import PCA
from sklearn.model_selection import (KFold, train_test_split, RepeatedKFold,
                                     GroupKFold, GridSearchCV)
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.kernel_ridge import KernelRidge
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 90
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# IMPROVED PARAMETER SETTINGS
IMPROVED_PARAMS = {
    'epochs': 300,           # Increased for better convergence
    'batch_size': 32,        # Smaller batches for genomic data
    'learning_rate': 0.001,  # More appropriate learning rate
    'l2_reg': 0.01,          # Reduced regularization
    'dropout_rate': 0.3,     # More reasonable dropout
    'rf_n_estimators': 500,  # Increased for stability
    'rf_max_depth': None,    # Let trees grow fully
    'alpha': 0.5,            # More balanced ensemble weighting
    'fixed_epochs_final': 150,  # Fixed epochs for final training (no early stopping)
}

TRULY_FIXED_PARAMETERS={
    'epochs':1000,
    'batch_size':32,
    'learning_rate':0.001,
    'l2_reg':0.01,
    'dropout_rate':0.3,
    'rf_n_estimators':200,
    'rf_max_depth':15,
    'alpha':0.5,
    'fixed_epochs_final':1000,
    #cv specific parameters
    'outer_n_splits':5,
    'feature_selection':True,
    'heritability':0.82,
    'use_raw_genotypes':False,
    'use_pca':True,
    'n_components':50,
    'rfe_n_features':100,
    'generate_plots':True
    
}
def get_cv_parameters():
    return{
        'epochs':1000,
        'batch_size': 32,
        'learning_rate':0.001,
        'l2_reg':0.01,
        'dropout_rate':0.3,
        'rf_n_estimators':100,
        'rf_max_depth':10,
        'alpha':0.5,
        'verbose':1,
        'use_early_stopping':True
        }
    
    
def get_final_training_parameters():
    """
    TRULY FIXED parameters for final model training
    NO influence from cross-validation results
    """
    return TRULY_FIXED_PARAMETERS.copy()


def _reml_solve(K, y, log_delta_bounds=(-8.0, 8.0), n_grid=65):
    
    n = len(y)
    d, U = np.linalg.eigh(K)
    d = np.clip(d, 0.0, None)
    ys = U.T @ y
    xs = U.T @ np.ones(n)

    def _stats(log_delta):
        delta = np.exp(log_delta)
        w = 1.0 / (d + delta)
        xwx = np.sum(xs * xs * w)
        beta = np.sum(xs * ys * w) / xwx
        r = ys - xs * beta
        s2g = np.sum(r * r * w) / (n - 1)
        return delta, w, xwx, beta, s2g

    def _neg_reml(log_delta):
        delta, w, xwx, beta, s2g = _stats(log_delta)
        return 0.5 * ((n - 1) * np.log(s2g) + np.sum(np.log(d + delta)) + np.log(xwx))

    grid = np.linspace(log_delta_bounds[0], log_delta_bounds[1], n_grid)
    vals = np.array([_neg_reml(g) for g in grid])
    k = int(np.argmin(vals))
    lo = grid[max(k - 1, 0)]
    hi = grid[min(k + 1, n_grid - 1)]
    opt = minimize_scalar(_neg_reml, bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-6})
    log_delta = opt.x if opt.fun <= vals[k] else grid[k]

    delta, w, xwx, beta, s2g = _stats(log_delta)
    alpha = U @ ((ys - xs * beta) * w)          # (K + delta I)^-1 (y - mu)
    return beta, alpha, delta, s2g, delta * s2g


class GBLUP_REML:
    
    def fit(self, Z, y):
        y = np.asarray(y, dtype=float).ravel()
        self.mean_ = Z.mean(axis=0)
        self.Zc_ = Z - self.mean_
        self.c_ = np.sum(np.var(Z, axis=0))
        G = (self.Zc_ @ self.Zc_.T) / self.c_
        self.mu_, self.alpha_, self.delta_, self.s2g_, self.s2e_ = _reml_solve(G, y)
        self.h2_ = self.s2g_ / (self.s2g_ + self.s2e_)
        return self

    def predict(self, Z):
        G_new = ((Z - self.mean_) @ self.Zc_.T) / self.c_
        return self.mu_ + G_new @ self.alpha_


def _reml_solve_fixed(K, y, X, eig=None, log_delta_bounds=(-8.0, 8.0), n_grid=65):
    
    n, p = X.shape
    d, U = eig if eig is not None else np.linalg.eigh(K)
    d = np.clip(d, 0.0, None)
    ys = U.T @ y
    Xs = U.T @ X

    def _stats(log_delta):
        delta = np.exp(log_delta)
        w = 1.0 / (d + delta)
        xwx = Xs.T @ (Xs * w[:, None])
        xwy = Xs.T @ (w * ys)
        try:
            beta = np.linalg.solve(xwx, xwy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(xwx, xwy, rcond=None)[0]
        r = ys - Xs @ beta
        s2g = np.sum(r * r * w) / (n - p)
        logdet_xwx = np.linalg.slogdet(xwx)[1]
        return delta, w, beta, r, s2g, logdet_xwx

    def _neg_reml(log_delta):
        delta, w, beta, r, s2g, logdet_xwx = _stats(log_delta)
        return 0.5 * ((n - p) * np.log(s2g) + np.sum(np.log(d + delta)) + logdet_xwx)

    grid = np.linspace(log_delta_bounds[0], log_delta_bounds[1], n_grid)
    vals = np.array([_neg_reml(g) for g in grid])
    k = int(np.argmin(vals))
    lo = grid[max(k - 1, 0)]
    hi = grid[min(k + 1, n_grid - 1)]
    opt = minimize_scalar(_neg_reml, bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-6})
    log_delta = opt.x if opt.fun <= vals[k] else grid[k]

    delta, w, beta, r, s2g, _ = _stats(log_delta)
    alpha = U @ (r * w)                          # (K + delta I)^-1 (y - X b)
    return beta, alpha, delta, s2g, delta * s2g


class GBLUP_PC3_REML:
    
    def __init__(self, n_pcs=3):
        self.n_pcs = n_pcs

    def fit(self, Z, y):
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)
        self.mean_ = Z.mean(axis=0)
        self.Zc_ = Z - self.mean_
        self.c_ = np.sum(np.var(Z, axis=0))
        G = (self.Zc_ @ self.Zc_.T) / self.c_

        d, U = np.linalg.eigh(G)
        d = np.clip(d, 0.0, None)
        top = np.argsort(d)[::-1][:self.n_pcs]
        self.pc_d_ = d[top]
        self.pc_U_ = U[:, top]
        self.pc_var_explained_ = self.pc_d_ / d.sum()
        scores = self.pc_U_ * np.sqrt(self.pc_d_)            # training PC scores
        self.pc_mean_ = scores.mean(axis=0)
        self.pc_sd_ = scores.std(axis=0)
        S = (scores - self.pc_mean_) / self.pc_sd_
        X = np.column_stack([np.ones(n), S])

        beta, self.alpha_, self.delta_, self.s2g_, self.s2e_ = _reml_solve_fixed(G, y, X, eig=(d, U))
        self.mu_ = beta[0]
        self.pc_effects_ = beta[1:]
        self.h2_ = self.s2g_ / (self.s2g_ + self.s2e_)
        return self

    def predict(self, Z):
        G_new = ((Z - self.mean_) @ self.Zc_.T) / self.c_
        S_new = (G_new @ self.pc_U_) / np.sqrt(self.pc_d_)   # projection onto training axes
        S_new = (S_new - self.pc_mean_) / self.pc_sd_
        return self.mu_ + S_new @ self.pc_effects_ + G_new @ self.alpha_


class RRBLUP_REML:
   
    def fit(self, Z, y):
        y = np.asarray(y, dtype=float).ravel()
        self.mean_ = Z.mean(axis=0)
        Zc = Z - self.mean_
        K = Zc @ Zc.T
        self.c_ = np.mean(np.diag(K))
        mu, alpha, self.delta_, s2g, s2e = _reml_solve(K / self.c_, y)
        self.mu_ = mu
        self.u_ = (Zc.T @ alpha) / self.c_           # marker effects
        self.s2u_, self.s2e_ = s2g / self.c_, s2e
        self.lambda_ = self.s2e_ / self.s2u_
        return self

    def predict(self, Z):
        return self.mu_ + (Z - self.mean_) @ self.u_


class LassoInnerCV:
  
    def __init__(self, inner_splits=5, n_alphas=100, max_iter=10000,
                 random_state=RANDOM_STATE):
        self.inner_splits = inner_splits
        self.n_alphas = n_alphas
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, Z, y):
        y = np.asarray(y, dtype=float).ravel()
        self.scaler_ = StandardScaler().fit(Z)
        inner_cv = KFold(n_splits=self.inner_splits, shuffle=True,
                         random_state=self.random_state)
        self.model_ = LassoCV(cv=inner_cv, n_alphas=self.n_alphas,
                              max_iter=self.max_iter, n_jobs=-1,
                              random_state=self.random_state)
        self.model_.fit(self.scaler_.transform(Z), y)
        self.alpha_ = self.model_.alpha_
        return self

    def predict(self, Z):
        return self.model_.predict(self.scaler_.transform(Z))



LEGACY_BASELINES = {
    'Lasso': lambda: Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=10000),
    'RBLUP': lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE),
    'GBLUP': lambda: KernelRidge(alpha=1.0, kernel='rbf', gamma=None, degree=3),
}


REFERENCE_BASELINES = {
    'Lasso_1': LassoInnerCV,
    'RRBLUP_1': RRBLUP_REML,
    'GBLUP_1': GBLUP_REML,
    'GBLUP_PC3': GBLUP_PC3_REML,
}

BASELINE_MODELS = list(LEGACY_BASELINES) + list(REFERENCE_BASELINES)
ALL_MODEL_NAMES = ['BreedSight'] + BASELINE_MODELS


FOLD_PARAM_LOG = []


def _collect_fold_params(model_name, model, fold):
    
    row = {'Fold': fold, 'Model': model_name}
    if model_name in ('GBLUP_1', 'GBLUP_PC3', 'RRBLUP_1'):
        delta = float(model.delta_)                      # s2e / s2g (GRM scaled to mean diag 1)
        s2e = float(model.s2e_)
        s2g = s2e / delta
        row.update({'s2_genetic_or_marker': float(model.s2u_) if model_name == 'RRBLUP_1' else s2g,
                    's2e': s2e,
                    'lambda': float(model.lambda_) if model_name == 'RRBLUP_1' else delta,
                    'h2': 1.0 / (1.0 + delta)})
    if model_name == 'GBLUP_PC3':
        for i in range(len(model.pc_effects_)):
            row[f'PC{i+1}_effect_per_SD'] = float(model.pc_effects_[i])
        row['PC1_3_var_explained_pct'] = 100.0 * float(np.sum(model.pc_var_explained_))
    if model_name == 'Lasso_1':
        coef = model.model_.coef_
        row.update({'alpha': float(model.alpha_), 'n_markers': int(coef.size),
                    'n_nonzero': int(np.sum(coef != 0)),
                    'inner_cv_mse': float(np.min(np.mean(model.model_.mse_path_, axis=1)))})
    return row


def baseline_feature_type(name):
    
    if name in LEGACY_BASELINES:
        return 'grm'
    if name in REFERENCE_BASELINES:
        return 'raw'
    raise ValueError(f"Unknown baseline model: {name}")


def build_baseline_model(name):
    if name in LEGACY_BASELINES:
        return LEGACY_BASELINES[name]()
    if name in REFERENCE_BASELINES:
        return REFERENCE_BASELINES[name]()
    raise ValueError(f"Unknown baseline model: {name}")


################################
def train_final_models_truly_independent(X_train, y_train, X_test, X_train_raw, X_test_raw,
                                         breedsight_params=None):
    
    print("  Training final models with TRULY INDEPENDENT parameters...")

    final_predictions = {}
    final_models = {}

    print("  Training final BreedSight model...")
    if breedsight_params is not None:
       
        breedSight_params = dict(breedsight_params)
        breedSight_params['verbose'] = 0
        breedSight_params['use_early_stopping'] = True
    else:
        
        final_params = get_final_training_parameters()
        breedSight_params = {
            'epochs': final_params['fixed_epochs_final'],
            'batch_size': final_params['batch_size'],
            'learning_rate': final_params['learning_rate'],
            'l2_reg': final_params['l2_reg'],
            'dropout_rate': final_params['dropout_rate'],
            'rf_n_estimators': final_params['rf_n_estimators'],
            'rf_max_depth': final_params['rf_max_depth'],
            'alpha': final_params['alpha'],
            'verbose': 0,
            'use_early_stopping': True,
        }

    pred_train, _, pred_test, history, rf_model, artifacts = BreedSight(
        trainX=X_train, trainy=y_train,
        valX=None, valy=None,
        testX=X_test, testy=None,
        **breedSight_params
    )
    final_predictions['BreedSight'] = pred_test
    final_models['BreedSight'] = {
        'rf_model': rf_model,
        'artifacts': artifacts,
        'history': history,
    }

    for model_name in BASELINE_MODELS:
        print(f"  Training final {model_name} model...")
        model = build_baseline_model(model_name)
        if baseline_feature_type(model_name) == 'raw':
            model.fit(X_train_raw, y_train.ravel())
            final_predictions[model_name] = model.predict(X_test_raw)
        else:
            model.fit(X_train, y_train.ravel())
            final_predictions[model_name] = model.predict(X_test)
        final_models[model_name] = {'model': model}

    print("  ✅ Final models trained; test data never used for fitting, tuning or stopping")
    return final_predictions, final_models
###################################
def validate_genomic_data_comprehensive(training_data, training_additive, testing_data, testing_additive):
    
    print("=== Comprehensive Data Validation ===")
    
    # Check for missing values (keep this strict)
    train_missing_geno = training_additive.isnull().sum().sum()
    test_missing_geno = testing_additive.isnull().sum().sum()
    train_missing_pheno = training_data['phenotypes'].isnull().sum()
    
    if train_missing_geno > 0 or test_missing_geno > 0 or train_missing_pheno > 0:
        raise ValueError(f"❌ Missing values detected: "
                        f"Training genotypes: {train_missing_geno}, "
                        f"Testing genotypes: {test_missing_geno}, "
                        f"Training phenotypes: {train_missing_pheno}. "
                        f"Please preprocess data to remove missing values.")
    
    # Check for duplicate samples (keep this strict)
    train_duplicates = training_data.duplicated().sum()
    test_duplicates = testing_data.duplicated().sum()
    
    if train_duplicates > 0 or test_duplicates > 0:
        raise ValueError(f"❌ Duplicate samples found: "
                        f"Training: {train_duplicates}, Testing: {test_duplicates}")
    
   
    training_variance = np.var(training_additive.iloc[:, 1:], axis=0)
    constant_features = np.sum(training_variance == 0)
    
    if constant_features > 0:
        print(f"⚠️ Removing {constant_features} constant features (zero variance)...")
        
        non_constant_mask = training_variance > 0
        training_additive = training_additive.iloc[:, np.concatenate([[True], non_constant_mask])]
        testing_additive = testing_additive.iloc[:, np.concatenate([[True], non_constant_mask])]
    
   
    if training_data.shape[0] != training_additive.shape[0]:
        raise ValueError("❌ Training data and additive matrices have different sample counts")
    
    if testing_data.shape[0] != testing_additive.shape[0]:
        raise ValueError("❌ Testing data and additive matrices have different sample counts")
    
   
    train_markers = set(training_additive.columns[1:])
    test_markers = set(testing_additive.columns[1:])
    
    if train_markers != test_markers:
        common_markers = train_markers.intersection(test_markers)
        train_only_markers = train_markers - test_markers
        test_only_markers = test_markers - train_markers
        
        print(f"⚠️ Marker sets don't match exactly:")
        print(f"  Training markers: {len(train_markers)}")
        print(f"  Testing markers: {len(test_markers)}")
        print(f"  Common markers: {len(common_markers)}")
        print(f"  Training-only markers: {len(train_only_markers)}")
        print(f"  Testing-only markers: {len(test_only_markers)}")
        
        if len(common_markers) == 0:
            raise ValueError("❌ No common markers between training and testing data")
        
       
        common_markers_list = sorted(list(common_markers))
        training_additive = training_additive[['sample_id'] + common_markers_list]
        testing_additive = testing_additive[['sample_id'] + common_markers_list]
        print(f"  ✅ Using {len(common_markers_list)} common markers for analysis")
    
    
    
    print("✅ Comprehensive data validation passed")
    return training_additive, testing_additive

def safe_data_preprocessing(X_train, X_val=None, X_test=None, y_train=None):
   
    # Input validation
    if X_train is None or len(X_train) == 0:
        raise ValueError("Training data cannot be empty")
    
    # Initialize transformers
    feature_scaler = StandardScaler()
    feature_selector = None
    
   
    print(f"  Scaling: Fitting on {X_train.shape[0]} training samples only")
    X_train_scaled = feature_scaler.fit_transform(X_train)
    
   
    X_val_scaled = feature_scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = feature_scaler.transform(X_test) if X_test is not None else None
    
   
    if y_train is not None and X_train.shape[1] > TRULY_FIXED_PARAMETERS['rfe_n_features']:
        print(f"  Feature selection: Selecting {TRULY_FIXED_PARAMETERS['rfe_n_features']} features from training data only")
        rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        feature_selector = RFE(estimator=rf, n_features_to_select=TRULY_FIXED_PARAMETERS['rfe_n_features'])
        feature_selector.fit(X_train_scaled, y_train.ravel())
        
        
        X_train_scaled = feature_selector.transform(X_train_scaled)
        X_val_scaled = feature_selector.transform(X_val_scaled) if X_val_scaled is not None else None
        X_test_scaled = feature_selector.transform(X_test_scaled) if X_test_scaled is not None else None
        
        print(f"  ✅ Selected {np.sum(feature_selector.support_)} features from training data")
    
    artifacts = {
        'feature_scaler': feature_scaler,
        'feature_selector': feature_selector
    }
    
    return X_train_scaled, X_val_scaled, X_test_scaled, artifacts



def calculate_metrics_with_ci(true_vals, pred_vals, n_bootstraps=1000):
    
    if len(true_vals) == 0 or len(pred_vals) == 0:
        return {
            'mse': np.nan, 'rmse': np.nan, 'pearson_r': np.nan, 
            'pearson_p': np.nan, 'r2': np.nan,
            'r2_ci': [np.nan, np.nan], 'pearson_ci': [np.nan, np.nan]
        }
    
    if len(true_vals) != len(pred_vals):
        raise ValueError(f"True values ({len(true_vals)}) and predictions ({len(pred_vals)}) have different lengths")
    
  
    mse = mean_squared_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    
   
    if len(np.unique(pred_vals)) == 1:
        corr = 0.0
        p_value = 1.0
        r2 = 0.0
    else:
        corr, p_value = pearsonr(true_vals, pred_vals)
        r2 = r2_score(true_vals, pred_vals)
    
   
    n_samples = len(true_vals)
    r2_bootstrap = []
    corr_bootstrap = []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        try:
            r2_boot = r2_score(true_vals[indices], pred_vals[indices])
            corr_boot, _ = pearsonr(true_vals[indices], pred_vals[indices])
            r2_bootstrap.append(r2_boot)
            corr_bootstrap.append(corr_boot)
        except:
            continue
    
    r2_ci = np.percentile(r2_bootstrap, [2.5, 97.5]) if r2_bootstrap else [np.nan, np.nan]
    corr_ci = np.percentile(corr_bootstrap, [2.5, 97.5]) if corr_bootstrap else [np.nan, np.nan]
    
    return {
        'mse': mse, 'rmse': rmse, 'pearson_r': corr, 'pearson_p': p_value, 'r2': r2,
        'r2_ci': r2_ci, 'pearson_ci': corr_ci
    }

def compute_grm_efficient(X_train, X_val=None, ref_features=None, is_train=True,
                               max_markers=10000, use_raw_genotypes=False,
                               use_pca=False, n_components=50):
    
   
    if np.isnan(X_train).any():
        raise ValueError("❌ Training genotype matrix contains missing values")
    
    if X_val is not None and np.isnan(X_val).any():
        raise ValueError("❌ Validation genotype matrix contains missing values")
    
    
    if is_train and X_train.shape[1] > max_markers:
        np.random.seed(RANDOM_STATE)
        selected_markers = np.random.choice(X_train.shape[1], max_markers, replace=False)
        X_train = X_train[:, selected_markers]
        if X_val is not None:
            X_val = X_val[:, selected_markers]
    elif not is_train and ref_features is not None and ref_features.get('selected_markers') is not None:
        selected_markers = ref_features['selected_markers']
        X_train = X_train[:, selected_markers]
        if X_val is not None:
            X_val = X_val[:, selected_markers]
    else:
        selected_markers = None
    
    if use_raw_genotypes:
        
        if is_train:
            marker_means = np.mean(X_train, axis=0)
            X_train_centered = X_train - marker_means
            X_val_centered = (X_val - marker_means) if X_val is not None else None
            ref_features = {
                'marker_means': marker_means, 
                'selected_markers': selected_markers,
                'centering_only': True
            }
            return X_train_centered, X_val_centered, ref_features
        else:
            X_train_centered = X_train - ref_features['marker_means']
            X_val_centered = (X_val - ref_features['marker_means']) if X_val is not None else None
            return X_train_centered, X_val_centered, ref_features
    else:
       
        if is_train:
           
            marker_means = np.mean(X_train, axis=0)
            X_train_centered = X_train - marker_means
            n_markers = X_train_centered.shape[1]
            
            
            X_train_centered_original = X_train_centered.copy()  # Store for test phase
            
            
            G_train = np.dot(X_train_centered, X_train_centered.T) / n_markers
            
          
            I_train = G_train * G_train
            mean_diag = np.mean(np.diag(I_train))
            I_train_norm = I_train / mean_diag if mean_diag != 0 else I_train
            
            
            X_train_final = np.concatenate([G_train, I_train_norm], axis=1)
            
           
            X_val_final = None
            if X_val is not None:
                X_val_centered = X_val - marker_means
             
                G_val = np.dot(X_val_centered, X_train_centered.T) / n_markers  # val × train
                I_val = G_val * G_val
                I_val_norm = I_val / mean_diag if mean_diag != 0 else I_val
                X_val_final = np.concatenate([G_val, I_val_norm], axis=1)
            
            
            ref_features = {
                'marker_means': marker_means,
                'mean_diag': mean_diag,
                'n_markers': n_markers,
                'selected_markers': selected_markers,
                'centering_only': True,
                'X_train_centered_original': X_train_centered_original  # CRITICAL FIX
            }
            
            
            if use_pca:
                pca = PCA(n_components=min(n_components, X_train_final.shape[1]))
                X_train_final = pca.fit_transform(X_train_final)
                if X_val_final is not None:
                    X_val_final = pca.transform(X_val_final)
                ref_features['pca'] = pca
            
            return X_train_final, X_val_final, ref_features
            
        else:
           
            if ref_features is None:
                raise ValueError("ref_features must be provided for validation/testing")
            
            
            if 'X_train_centered_original' not in ref_features:
                raise ValueError("Original training data not stored for proper test GRM computation")
            
            X_test_centered = X_train - ref_features['marker_means']
            X_train_centered_original = ref_features['X_train_centered_original']
            n_markers = ref_features['n_markers']
            
            
            G_test = np.dot(X_test_centered, X_train_centered_original.T) / n_markers  # test × train
            
            I_test = G_test * G_test
            I_test_norm = I_test / ref_features['mean_diag'] if ref_features['mean_diag'] != 0 else I_test
            
            X_test_final = np.concatenate([G_test, I_test_norm], axis=1)
            
            
            if use_pca and 'pca' in ref_features:
                X_test_final = ref_features['pca'].transform(X_test_final)
            
            return X_test_final, None, ref_features

def safe_feature_selection(X_train, y_train, X_val, n_features=100):
    """
    Perform feature selection using ONLY training data to prevent leakage.
    """
    if X_train.shape[1] <= n_features:
        
        print(f"  RFE skipped: {X_train.shape[1]} features <= rfe_n_features={n_features}")
        return X_train, X_val, None
    
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    selector = RFE(estimator=rf, n_features_to_select=min(n_features, X_train.shape[1]))
    
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val) if X_val is not None else None
    
    print(f"  Selected {np.sum(selector.support_)} features with RFE")
    
    return X_train_selected, X_val_selected, selector

def BreedSight(trainX, trainy, valX=None, valy=None, testX=None, testy=None,
               epochs=300, batch_size=32, learning_rate=0.001,
               l2_reg=0.01, dropout_rate=0.3,
               rf_n_estimators=500, rf_max_depth=None,
               alpha=0.5, verbose=1, use_early_stopping=True,
               es_val_fraction=0.15):
    
    # Input validation - check for missing values
    if np.isnan(trainX).any() or np.isnan(trainy).any():
        raise ValueError("❌ Training data contains missing values")
    if valX is not None and (np.isnan(valX).any() or np.isnan(valy).any()):
        raise ValueError("❌ Validation data contains missing values")
    if testX is not None and np.isnan(testX).any():
        raise ValueError("❌ Test data contains missing values")
    
    if not isinstance(trainX, np.ndarray) or not isinstance(trainy, np.ndarray):
        raise ValueError("trainX and trainy must be numpy arrays")
    if trainX.shape[0] != trainy.shape[0]:
        raise ValueError("trainX and trainy must have the same number of samples")
    
    # ---- Inner split of the TRAINING data for early stopping ----
    n_train = trainX.shape[0]
    if use_early_stopping:
        es_fit_idx, es_val_idx = train_test_split(
            np.arange(n_train), test_size=es_val_fraction,
            random_state=RANDOM_STATE, shuffle=True
        )
    else:
        es_fit_idx, es_val_idx = np.arange(n_train), None

    # Scaler fitted on the (inner-)training rows only
    feature_scaler = StandardScaler()
    feature_scaler.fit(trainX[es_fit_idx])
    trainX_scaled = feature_scaler.transform(trainX)

    # Use raw phenotypes for consistency and interpretability
    trainy_final = trainy

    # Outer hold-out fold: scaled for PREDICTION ONLY (never given to model.fit)
    valX_scaled = feature_scaler.transform(valX) if valX is not None else None

    # Test data: scaled with training parameters
    if testX is not None:
        testX_scaled = feature_scaler.transform(testX)
    else:
        testX_scaled = None

    def build_dnn_model(input_shape):
        """Build improved Deep Neural Network architecture"""
        inputs = tf.keras.Input(shape=(input_shape,))
        
        # Enhanced architecture with better capacity
        x = Dense(256, kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(l2_reg))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 2
        x = Dense(128, kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 3
        x = Dense(64, kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Output layer
        outputs = Dense(1, activation="linear")(x)
        model = tf.keras.Model(inputs, outputs)
        
        # Huber loss works directly with raw phenotypes
        model.compile(loss=tf.keras.losses.Huber(delta=1.0),
                     optimizer=Adam(learning_rate=learning_rate, clipvalue=0.5),
                     metrics=['mse'])
        return model
    
    # Build and train DNN
    dnn_model = build_dnn_model(trainX.shape[1])
    
    # Callbacks act on the INNER validation split (taken from the training data)
    callbacks = []
    fit_validation_data = None
    if use_early_stopping:
        fit_validation_data = (trainX_scaled[es_val_idx], trainy_final[es_val_idx])
        callbacks = [
            EarlyStopping(monitor='val_loss',
                          verbose=verbose,
                          restore_best_weights=True, patience=15),
            ReduceLROnPlateau(monitor='val_loss',
                             factor=0.5, patience=10,
                             min_lr=1e-7, verbose=verbose)
        ]

    # Train DNN (outer hold-out fold is NOT used here)
    history = dnn_model.fit(
        trainX_scaled[es_fit_idx],
        trainy_final[es_fit_idx],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=fit_validation_data,
        verbose=verbose,
        callbacks=callbacks,
        shuffle=True
    )

    # Train Random Forest with improved parameters
    rf_model = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_split=5,
        min_samples_leaf=2
    )
    rf_model.fit(trainX, trainy.ravel())
    
    # Generate predictions
    predicted_train_dnn = dnn_model.predict(trainX_scaled).flatten()
    predicted_train_rf = rf_model.predict(trainX)
    
    # Validation predictions
    predicted_val_dnn = None
    predicted_val_rf = None
    if valX is not None:
        predicted_val_dnn = dnn_model.predict(valX_scaled).flatten()
        predicted_val_rf = rf_model.predict(valX)
    
    # Test predictions
    predicted_test_dnn = None
    predicted_test_rf = None
    if testX is not None:
        predicted_test_dnn = dnn_model.predict(testX_scaled).flatten()
        predicted_test_rf = rf_model.predict(testX)
    
    # Ensemble predictions (weighted combination)
    predicted_train = alpha * predicted_train_dnn + (1 - alpha) * predicted_train_rf
    predicted_val = (alpha * predicted_val_dnn + (1 - alpha) * predicted_val_rf
                    if valX is not None else None)
    predicted_test = (alpha * predicted_test_dnn + (1 - alpha) * predicted_test_rf
                     if testX is not None else None)
    
    # Store feature scaler for future use
    model_artifacts = {
        'feature_scaler': feature_scaler,
        'dnn_model': dnn_model,
        'rf_model': rf_model
    }
    
    # Performance reporting
    if verbose > 0:
        print("\n=== Training Summary ===")
        print(f"Train samples: {len(trainX)}, Validation samples: {len(valX) if valX is not None else 'N/A'}")
        
        train_r2 = r2_score(trainy, predicted_train) if predicted_train is not None else np.nan
        val_r2 = r2_score(valy, predicted_val) if valX is not None and valy is not None and predicted_val is not None else np.nan
        
        print(f"Training R²: {train_r2:.4f}, Validation R²: {val_r2:.4f}")
        
        # Overfitting/underfitting detection
        if val_r2 is not np.nan and train_r2 - val_r2 > 0.2:
            print("⚠️ Warning: Potential overfitting detected (large gap between train and validation R²)")
        if val_r2 is not np.nan and val_r2 < 0.6:
            print("⚠️ Warning: Potential underfitting detected (low validation R²)")
    
    return predicted_train, predicted_val, predicted_test, history, rf_model, model_artifacts

def check_data_leakage(train_ids, val_ids, test_ids=None):
    """
    Check for data leakage between datasets.
    """
    train_set = set(train_ids)
    val_set = set(val_ids)
    
    leakage_found = False
    
    if train_set.intersection(val_set):
        print("🚨 CRITICAL: Data leakage between train and validation sets!")
        print(f"  Leaking samples: {train_set.intersection(val_set)}")
        leakage_found = True
    
    if test_ids is not None:
        test_set = set(test_ids)
        if train_set.intersection(test_set):
            print("🚨 CRITICAL: Data leakage between train and test sets!")
            print(f"  Leaking samples: {train_set.intersection(test_set)}")
            leakage_found = True
        if val_set.intersection(test_set):
            print("🚨 CRITICAL: Data leakage between validation and test sets!")
            print(f"  Leaking samples: {val_set.intersection(test_set)}")
            leakage_found = True
    
    if not leakage_found:
        print("✅ No data leakage detected between datasets")
    
    return not leakage_found

def generate_regression_plot(true_vals, pred_vals, dataset_name, fold, model_name, output_dir="output1/diagnostic_plots"):
    """
    Generate comprehensive regression plot with confidence intervals.
    """
    if len(true_vals) == 0 or len(pred_vals) == 0:
        print(f"⚠️ No data to plot for {model_name}, {dataset_name}, Fold {fold}")
        return None, None
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics with CI
    metrics = calculate_metrics_with_ci(true_vals, pred_vals)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(true_vals, pred_vals, alpha=0.6, s=50, color='blue', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line (y=x)
    min_val = min(min(true_vals), min(pred_vals))
    max_val = max(max(true_vals), max(pred_vals))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Regression line
    z = np.polyfit(true_vals, pred_vals, 1)
    p = np.poly1d(z)
    plt.plot(true_vals, p(true_vals), "b-", linewidth=2, label='Regression Line')
    
    # Customize plot
    plt.xlabel('True Phenotype', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Phenotype', fontsize=12, fontweight='bold')
    
    # Add metrics as text box with CI
    textstr = '\n'.join((
        f'R² = {metrics["r2"]:.4f}',
        f'R² CI = [{metrics["r2_ci"][0]:.4f}, {metrics["r2_ci"][1]:.4f}]',
        f'Pearson r = {metrics["pearson_r"]:.4f}',
        f'Pearson CI = [{metrics["pearson_ci"][0]:.4f}, {metrics["pearson_ci"][1]:.4f}]',
        f'p-value = {metrics["pearson_p"]:.2e}',
        f'RMSE = {metrics["rmse"]:.4f}',
        f'N = {len(true_vals)}'
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    
    plt.title(f'{dataset_name} - {model_name} (Fold {fold})\nTrue vs Predicted Phenotypes', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    filename = f"{dataset_name.lower()}_{model_name}_fold_{fold}_regression.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved regression plot: {filename}")
    
    return plot_path, metrics

def KFoldCrossValidation(training_data, training_additive, testing_data, testing_additive,
                         epochs=300, batch_size=32, learning_rate=0.001,
                         l2_reg=0.01, dropout_rate=0.3, rf_n_estimators=500,
                         rf_max_depth=None, alpha=0.5, outer_n_splits=10,
                         feature_selection=True, heritability=0.82,
                         use_raw_genotypes=False, use_pca=False, n_components=100,
                         fixed_epochs_final=None,rfe_n_features=200, generate_plots=True):
    """
    Perform K-fold cross-validation with COMPLETE data leakage prevention.
    """
    # Input validation
    assert isinstance(training_data, pd.DataFrame), "Training data must be DataFrame"
    assert isinstance(testing_data, pd.DataFrame), "Testing data must be DataFrame"
    
    if 'phenotypes' not in training_data.columns:
        raise ValueError("Training data must contain 'phenotypes' column")
    
    # Validate data quality
    training_additive, testing_additive = validate_genomic_data_comprehensive(
        training_data, training_additive, testing_data, testing_additive
    )
    
    # Extract clean data
    training_additive_raw = training_additive.iloc[:, 1:].values
    phenotypic_info_raw = training_data['phenotypes'].values
    
    # Check for missing values
    if np.isnan(phenotypic_info_raw).any():
        raise ValueError("❌ Training phenotypes contain missing values. Preprocess data first.")
    
    has_test_phenotypes = 'phenotypes' in testing_data.columns
    if has_test_phenotypes:
        phenotypic_test_info_raw = testing_data['phenotypes'].values
        if np.isnan(phenotypic_test_info_raw).any():
            raise ValueError("❌ Testing phenotypes contain missing values. Preprocess data first.")
    else:
        phenotypic_test_info_raw = None
        
    test_sample_ids = testing_data.iloc[:, 0].values
    testing_additive_raw = testing_additive.iloc[:, 1:].values
    
    # Check for missing values in genomic data
    if np.isnan(training_additive_raw).any():
        raise ValueError("❌ Training genomic data contains missing values. Preprocess data first.")
    if np.isnan(testing_additive_raw).any():
        raise ValueError("❌ Testing genomic data contains missing values. Preprocess data first.")
    
    # Check for data leakage before starting
    print("=== Data Leakage Check ===")
    outer_kf = KFold(n_splits=outer_n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_idx, val_idx) in enumerate(outer_kf.split(training_data), 1):
        train_ids = training_data.iloc[train_idx, 0].values
        val_ids = training_data.iloc[val_idx, 0].values
        check_data_leakage(train_ids, val_ids)
    
    # Initialize results storage
    results_dict = {name: [] for name in ALL_MODEL_NAMES}
    FOLD_PARAM_LOG.clear()
    train_pred_list = {model: [] for model in results_dict}
    val_pred_list = {model: [] for model in results_dict}
    plot_metrics = {model: {'train': [], 'val': []} for model in results_dict}
    
   
    model_configs = {
        'BreedSight': {
            'function': BreedSight,
            'params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'l2_reg': l2_reg,
                'dropout_rate': dropout_rate,
                'rf_n_estimators': rf_n_estimators,
                'rf_max_depth': rf_max_depth,
                'alpha': alpha,
                'verbose': 1,
                'use_early_stopping': True   # inner split; outer fold untouched
            },
            'feature_type': 'grm',
            'is_tree': True
        },
    }
    for _name in BASELINE_MODELS:
        model_configs[_name] = {'feature_type': baseline_feature_type(_name), 'is_tree': False}

    print(f"\n=== Starting {outer_n_splits}-Fold Cross Validation ===")
    print("REVISION FIXES APPLIED:")
    print("  ✓ Early stopping uses an inner split of the training fold (outer fold untouched)")
    print("  ✓ Legacy Lasso / RBLUP / GBLUP preserved (fixed penalties, BreedSight feature space)")
    print("  ✓ Added RRBLUP_1 / GBLUP_1 / GBLUP_PC3 (REML) and Lasso_1 (inner-CV path) on all raw markers")
    print("  ✓ All models use identical outer folds\n")
    
    # Cross-validation loop
    for outer_fold, (outer_train_index, outer_val_index) in enumerate(
        outer_kf.split(training_additive_raw), 1
    ):
        print(f"\n=== Outer Fold {outer_fold}/{outer_n_splits} ===")
        
        # Split data for current fold
        fold_train_additive_raw = training_additive_raw[outer_train_index]
        fold_val_additive_raw = training_additive_raw[outer_val_index]
        fold_train_phenotypes = phenotypic_info_raw[outer_train_index]
        fold_val_phenotypes = phenotypic_info_raw[outer_val_index]
        
        print(f"Fold {outer_fold}: Processing {len(fold_train_phenotypes)} train and {len(fold_val_phenotypes)} val samples")
        
        # Compute genomic features
        X_train_genomic, X_val_genomic, ref_features = compute_grm_efficient(
            fold_train_additive_raw, fold_val_additive_raw, ref_features=None, is_train=True,
            use_raw_genotypes=use_raw_genotypes, use_pca=use_pca, n_components=n_components
        )
        
        # Feature selection
        
        if feature_selection:
            X_train_final, X_val_final, selector = safe_feature_selection(
        X_train_genomic, fold_train_phenotypes, X_val_genomic, rfe_n_features
    )
            
        else:
            X_train_final = X_train_genomic
            X_val_final = X_val_genomic
            selector = None
        
        # Train and evaluate each model
        for model_name, config in model_configs.items():
            print(f"  Training {model_name}...")
            
            if model_name == 'BreedSight':
                # valX/valy: outer hold-out fold, used ONLY to generate predictions.
                pred_train, pred_val, _, history, model, artifacts = config['function'](
                    trainX=X_train_final,
                    trainy=fold_train_phenotypes,
                    valX=X_val_final,
                    valy=fold_val_phenotypes,
                    testX=None,
                    testy=None,
                    **config['params']
                )
            else:
                # Baselines: same outer folds for every model.
                #   legacy  (Lasso/RBLUP/GBLUP)      -> BreedSight feature space, fixed penalties
                #   *_1 (Lasso_1/RRBLUP_1/GBLUP_1/GBLUP_PC3) -> all QC-passed raw markers,
                #                                       penalty estimated from the training fold only
                model = build_baseline_model(model_name)
                if config['feature_type'] == 'raw':
                    model.fit(fold_train_additive_raw, fold_train_phenotypes.ravel())
                    pred_train = model.predict(fold_train_additive_raw)
                    pred_val = model.predict(fold_val_additive_raw)
                else:
                    model.fit(X_train_final, fold_train_phenotypes.ravel())
                    pred_train = model.predict(X_train_final)
                    pred_val = model.predict(X_val_final)
                history = None
                artifacts = None
                if model_name in ('GBLUP_1', 'RRBLUP_1', 'GBLUP_PC3'):
                    lam = getattr(model, 'delta_', np.nan)
                    print(f"    {model_name}: REML lambda (s2e/s2g, scaled) = {lam:.4g}")
                    if model_name == 'GBLUP_PC3':
                        print("    GBLUP_PC3: PC1-3 fixed effects (per SD) = "
                              + ", ".join(f"{b:.3g}" for b in model.pc_effects_))
                elif model_name == 'Lasso_1':
                    print(f"    Lasso_1: inner-CV alpha = {model.alpha_:.4g}")
                if model_name in ('GBLUP_1', 'RRBLUP_1', 'GBLUP_PC3', 'Lasso_1'):
                    FOLD_PARAM_LOG.append(_collect_fold_params(model_name, model, outer_fold))
            
            # Store predictions
            train_pred_list[model_name].append(pd.DataFrame({
                'Sample_ID': training_data.iloc[outer_train_index, 0].values,
                f'Predicted_Phenotype_{model_name}': pred_train,
                'True_Phenotype': fold_train_phenotypes,
                'Fold': outer_fold
            }))
            
            val_pred_list[model_name].append(pd.DataFrame({
                'Sample_ID': training_data.iloc[outer_val_index, 0].values,
                f'Predicted_Phenotype_{model_name}': pred_val,
                'True_Phenotype': fold_val_phenotypes,
                'Fold': outer_fold
            }))
            
            # Calculate metrics with CI
            train_metrics = calculate_metrics_with_ci(fold_train_phenotypes, pred_train)
            val_metrics = calculate_metrics_with_ci(fold_val_phenotypes, pred_val)
            
            results_dict[model_name].append({
                'Fold': outer_fold,
                'Train_R2': train_metrics['r2'],
                'Train_R2_CI_lower': train_metrics['r2_ci'][0],
                'Train_R2_CI_upper': train_metrics['r2_ci'][1],
                'Val_R2': val_metrics['r2'],
                'Val_R2_CI_lower': val_metrics['r2_ci'][0],
                'Val_R2_CI_upper': val_metrics['r2_ci'][1],
                'Train_MSE': train_metrics['mse'],
                'Val_MSE': val_metrics['mse'],
                'Train_RMSE': train_metrics['rmse'],
                'Val_RMSE': val_metrics['rmse'],
                'Train_Pearson_r': train_metrics['pearson_r'],
                'Train_Pearson_CI_lower': train_metrics['pearson_ci'][0],
                'Train_Pearson_CI_upper': train_metrics['pearson_ci'][1],
                'Val_Pearson_r': val_metrics['pearson_r'],
                'Val_Pearson_CI_lower': val_metrics['pearson_ci'][0],
                'Val_Pearson_CI_upper': val_metrics['pearson_ci'][1]
            })
            
            # Generate regression plots
            if generate_plots:
                # Training set plot
                train_plot_path, train_plot_metrics = generate_regression_plot(
                    fold_train_phenotypes, pred_train, 
                    "Training", outer_fold, model_name
                )
                if train_plot_metrics:
                    plot_metrics[model_name]['train'].append(train_plot_metrics)
                
                # Validation set plot  
                val_plot_path, val_plot_metrics = generate_regression_plot(
                    fold_val_phenotypes, pred_val,
                    "Validation", outer_fold, model_name
                )
                if val_plot_metrics:
                    plot_metrics[model_name]['val'].append(val_plot_metrics)
            
            print(f"    {model_name}: Train R² = {train_metrics['r2']:.4f}, Val R² = {val_metrics['r2']:.4f}")
            print(f"    {model_name}: Train Pearson r = {train_metrics['pearson_r']:.4f}, Val Pearson r = {val_metrics['pearson_r']:.4f}")
    
    print("\n" + "="*60)
    print("TRAINING FINAL MODELS WITH COMPLETE LEAKAGE PREVENTION")
    print("="*60)
    
    # CRITICAL FIX: Completely independent final model training
    X_train_final_raw = training_additive_raw
    y_train_final = phenotypic_info_raw

    # Compute genomic features for final training
    X_train_genomic_final, _, ref_features_final = compute_grm_efficient(
        X_train_final_raw, ref_features=None, is_train=True,
        use_raw_genotypes=use_raw_genotypes, use_pca=use_pca, n_components=n_components
    )
    
    # Compute test features using training parameters
    X_test_genomic_final, _, _ = compute_grm_efficient(
        testing_additive_raw, ref_features=ref_features_final, is_train=False,
        use_raw_genotypes=use_raw_genotypes, use_pca=use_pca, n_components=n_components
    )
    
    # Final feature selection
    if feature_selection:
        X_train_final_selected, X_test_final_selected, selector_final = safe_feature_selection(
            X_train_genomic_final, y_train_final, X_test_genomic_final, rfe_n_features
        )
    else:
        X_train_final_selected = X_train_genomic_final
        X_test_final_selected = X_test_genomic_final
        selector_final=None
    
    # Final BreedSight uses EXACTLY the same hyperparameters as the CV runs
    final_breedsight_params = dict(model_configs['BreedSight']['params'])
    
    # Train final models COMPLETELY INDEPENDENTLY
    final_test_predictions, final_models = train_final_models_truly_independent(
        X_train_final_selected,
        y_train_final,
        X_test_final_selected,
        X_train_raw=training_additive_raw,
        X_test_raw=testing_additive_raw,
        breedsight_params=final_breedsight_params
    )

    # ---- Independent test-set metrics (only if the test file has phenotypes) ----
    test_metrics_df = None
    if has_test_phenotypes:
        test_rows = []
        for _name in model_configs:
            _pred = np.asarray(final_test_predictions[_name]).ravel()
            _m = calculate_metrics_with_ci(phenotypic_test_info_raw, _pred)
            test_rows.append({
                'Model': _name,
                'Test_R2': _m['r2'],
                'Test_R2_CI_lower': _m['r2_ci'][0], 'Test_R2_CI_upper': _m['r2_ci'][1],
                'Test_Pearson_r': _m['pearson_r'],
                'Test_Pearson_CI_lower': _m['pearson_ci'][0], 'Test_Pearson_CI_upper': _m['pearson_ci'][1],
                'Test_RMSE': _m['rmse'], 'N_test': len(_pred)
            })
        test_metrics_df = pd.DataFrame(test_rows)
        print("\n=== INDEPENDENT TEST-SET RESULTS ===")
        print(test_metrics_df[['Model', 'Test_R2', 'Test_Pearson_r']].to_string(index=False))
    
    # Compile results
    results_df_dict = {model_name: pd.DataFrame(results) for model_name, results in results_dict.items()}
    
    # Compile predictions
    train_pred_df = pd.concat([
        pd.concat(train_pred_list[model_name], ignore_index=True)
        for model_name in model_configs.keys()
    ], axis=0)
    
    val_pred_df = pd.concat([
        pd.concat(val_pred_list[model_name], ignore_index=True)
        for model_name in model_configs.keys()
    ], axis=0)
    
    # Create final test predictions dataframe
    test_pred_final_df = pd.DataFrame({'Sample_ID': test_sample_ids})
    for model_name in model_configs:
        test_pred_final_df[f'Predicted_Phenotype_{model_name}'] = final_test_predictions[model_name]
    
    if has_test_phenotypes:
        test_pred_final_df['True_Phenotype'] = phenotypic_test_info_raw
    
    # Calculate comprehensive average metrics
    metrics_summary = {}
    for model_name in model_configs:
        df = results_df_dict[model_name]
        metrics_summary[model_name] = {
            'Avg_Train_R2': df['Train_R2'].mean(),
            'Avg_Val_R2': df['Val_R2'].mean(),
            'Std_Train_R2': df['Train_R2'].std(),
            'Std_Val_R2': df['Val_R2'].std(),
            'Avg_Train_MSE': df['Train_MSE'].mean(),
            'Avg_Val_MSE': df['Val_MSE'].mean(),
            'Avg_Train_RMSE': df['Train_RMSE'].mean(),
            'Avg_Val_RMSE': df['Val_RMSE'].mean(),
            'Avg_Train_Pearson_r': df['Train_Pearson_r'].mean(),
            'Avg_Val_Pearson_r': df['Val_Pearson_r'].mean(),
            'Std_Train_Pearson_r': df['Train_Pearson_r'].std(),
            'Std_Val_Pearson_r': df['Val_Pearson_r'].std(),
            'N_Folds': len(df)
        }
    
    metrics_df = pd.DataFrame(metrics_summary).T.reset_index().rename(columns={'index': 'Model'})
    
    # Generate summary plots
    if generate_plots:
        generate_summary_plots(plot_metrics, output_dir="output1/diagnostic_plots")
    
    # Performance analysis and recommendations
    print("\n" + "="*60)
    print("CROSS-VALIDATION COMPLETE - PERFORMANCE SUMMARY")
    print("="*60)
    
    # Find best performing model based on validation R²
    best_model = metrics_df.loc[metrics_df['Avg_Val_R2'].idxmax(), 'Model']
    best_val_r2 = metrics_df.loc[metrics_df['Avg_Val_R2'].idxmax(), 'Avg_Val_R2']
    
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
    print(f"   Validation R²: {best_val_r2:.4f}")
    
    # Model comparison
    print(f"\n📊 MODEL COMPARISON (Validation R²):")
    for _, row in metrics_df.iterrows():
        model = row['Model']
        train_r2 = row['Avg_Train_R2']
        val_r2 = row['Avg_Val_R2']
        gap = train_r2 - val_r2
        
        overfitting_warning = " ⚠️" if gap > 0.15 else ""
        underfitting_warning = " ⚠️" if val_r2 < 0.6 else ""
        print(f"   {model:12} | Train: {train_r2:.4f} | Val: {val_r2:.4f} | Gap: {gap:.4f}{overfitting_warning}{underfitting_warning}")
    
    # Data quality assessment
    print(f"\n🔍 DATA QUALITY ASSESSMENT:")
    print(f"   Training samples: {training_data.shape[0]}")
    print(f"   Testing samples: {testing_data.shape[0]}")
    print(f"   Markers: {training_additive.shape[1] - 1}")
    print(f"   Heritability threshold: {heritability}")
    
    # Check if any model exceeds heritability (potential overfitting)
    exceeding_models = metrics_df[metrics_df['Avg_Val_R2'] > heritability]['Model'].tolist()
    if exceeding_models:
        print(f"   ⚠️ WARNING: These models exceed heritability threshold: {exceeding_models}")
    
    print("\n✅ REVISION CHANGES APPLIED:")
    print("   1. ✅ Early stopping / LR reduction use an inner split of the training fold")
    print("   2. ✅ Legacy Lasso / RBLUP / GBLUP preserved as submitted")
    print("   3. ✅ RRBLUP_1 / GBLUP_1 / GBLUP_PC3 (REML, linear kernel) and Lasso_1 (inner-CV) added on raw markers")
    print("   4. ✅ Identical outer folds for all models")
    print("   5. ✅ Bootstrap confidence intervals for all metrics")
    
    return (results_df_dict, train_pred_df, val_pred_df, test_pred_final_df,
            metrics_df, final_test_predictions, final_models, test_metrics_df)
#######################################################
def generate_summary_plots(plot_metrics, output_dir="output1/diagnostic_plots"):
    """
    Generate summary plots across all folds.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # R² comparison across models and folds
    plt.figure(figsize=(12, 8))
    
    models = list(plot_metrics.keys())
    val_r2_means = []
    val_r2_stds = []
    
    for model in models:
        val_r2_values = [fold['r2'] for fold in plot_metrics[model]['val'] if 'r2' in fold]
        val_r2_means.append(np.mean(val_r2_values))
        val_r2_stds.append(np.std(val_r2_values))
    
    x_pos = np.arange(len(models))
    
    plt.bar(x_pos, val_r2_means, yerr=val_r2_stds, capsize=5, alpha=0.7)
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel('Validation R²', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison (Validation R²)', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, models, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "model_comparison_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Generated summary comparison plot")

# =============================================================================
# ROBUSTNESS / DIAGNOSTIC ANALYSES (added in revision)
#   1. near_duplicate_check : are validation lines near-clones of training lines?
#   2. repeated K-fold CV   : mean +/- SD over repeats (tests seed/split dependence)
#   3. grouped (cluster) CV : whole genetic clusters held out (tests population-
#                             structure / relatedness driven accuracy)
#   4. ablations            : DNN only, RF only, RF on top-3 / top-5 PCs
#   5. tuned strong baselines (RF, XGBoost) on the SAME features as BreedSight
# Clustering for grouped CV uses genotypes only (no phenotypes), so it cannot leak
# the response.  All feature construction / tuning is done inside each training split.
# =============================================================================
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False


class TunedRF:
    """Random forest with max_features / min_samples_leaf tuned by inner 3-fold CV."""
    def __init__(self, n_estimators=300, random_state=RANDOM_STATE):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        grid = {'max_features': [0.2, 0.5, 1.0], 'min_samples_leaf': [1, 3, 5, 10]}
        gs = GridSearchCV(
            RandomForestRegressor(n_estimators=self.n_estimators,
                                  random_state=self.random_state, n_jobs=-1),
            grid, scoring='neg_mean_squared_error',
            cv=KFold(3, shuffle=True, random_state=self.random_state))
        gs.fit(X, np.ravel(y))
        self.model_, self.best_params_ = gs.best_estimator_, gs.best_params_
        return self

    def predict(self, X):
        return self.model_.predict(X)


class TunedXGB:
    """XGBoost with depth / learning rate / n_estimators tuned by inner 3-fold CV."""
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state

    def fit(self, X, y):
        grid = {'max_depth': [2, 4], 'learning_rate': [0.03, 0.1],
                'n_estimators': [200, 500]}
        gs = GridSearchCV(
            XGBRegressor(subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                         random_state=self.random_state, verbosity=0),
            grid, scoring='neg_mean_squared_error',
            cv=KFold(3, shuffle=True, random_state=self.random_state))
        gs.fit(X, np.ravel(y))
        self.model_, self.best_params_ = gs.best_estimator_, gs.best_params_
        return self

    def predict(self, X):
        return self.model_.predict(X)


def near_duplicate_check(Z, splits, output_dir=None, tag="kfold"):
    """
    For every held-out individual: highest correlation (on marker-centred genotypes)
    with any individual in the training part of the split.  Values close to 1 mean
    near-clones / replicates sit on both sides of the split.
    """
    Zc = Z - Z.mean(axis=0)
    C = np.corrcoef(Zc)
    np.fill_diagonal(C, -1.0)
    rows = []
    for i, (tr, va) in enumerate(splits, 1):
        mx = C[np.ix_(va, tr)].max(axis=1)
        rows.append({'Split': i, 'N_val': len(va),
                     'Median_max_corr': np.median(mx),
                     'P90_max_corr': np.percentile(mx, 90),
                     'P99_max_corr': np.percentile(mx, 99),
                     'Max_corr': mx.max(),
                     'Frac_gt_0.90': float(np.mean(mx > 0.90)),
                     'Frac_gt_0.95': float(np.mean(mx > 0.95)),
                     'Frac_gt_0.99': float(np.mean(mx > 0.99))})
    df = pd.DataFrame(rows)
    print(f"\n=== Near-duplicate check ({tag}) ===")
    print(df.round(3).to_string(index=False))
    if output_dir is not None:
        df.to_csv(os.path.join(output_dir, f"near_duplicate_check_{tag}.csv"), index=False)
    return df


def make_grouped_splits(Z, n_splits=5, n_clusters=10, n_pcs=10, random_state=RANDOM_STATE):
    """Cluster individuals on the top genotype PCs (no phenotypes) and hold out whole clusters."""
    Zc = Z - Z.mean(axis=0)
    d, U = np.linalg.eigh(Zc @ Zc.T)
    top = np.argsort(d)[::-1][:n_pcs]
    pcs = U[:, top] * np.sqrt(np.clip(d[top], 0.0, None))
    groups = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(pcs)
    sizes = np.bincount(groups)
    print(f"  Grouped CV: {n_clusters} genetic clusters, sizes = {sizes.tolist()}")
    return list(GroupKFold(n_splits=n_splits).split(Z, groups=groups)), groups


def _split_metrics(y_true, y_pred):
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    r2 = r2_score(y_true, y_pred)
    r = 0.0 if np.std(y_pred) == 0 else pearsonr(y_true, y_pred)[0]
    return r2, r


def evaluate_on_splits(Z, y, splits, bs_params, scheme, output_dir,
                       n_components=50, include_tuned=True, n_splits_per_repeat=None):
    """
    Fit BreedSight (+ its ablations) and the reference baselines on each (train, val) split
    and return per-split and summary tables.  BreedSight uses exactly the same code and
    hyperparameters as the main CV.
    """
    y = np.asarray(y, dtype=float).ravel()
    rows = []
    for i, (tr, va) in enumerate(splits, 1):
        print(f"\n--- [{scheme}] split {i}/{len(splits)} ---")
        Ztr, Zva, ytr, yva = Z[tr], Z[va], y[tr], y[va]

        # BreedSight feature space, built from the training part only
        Xtr, Xva, _ = compute_grm_efficient(
            Ztr, Zva, ref_features=None, is_train=True,
            use_raw_genotypes=False, use_pca=True, n_components=n_components)

        tf.random.set_seed(RANDOM_STATE + i)
        params = dict(bs_params)
        params['verbose'] = 0
        _, p_ens, _, _, rf_bs, art = BreedSight(
            trainX=Xtr, trainy=ytr, valX=Xva, valy=yva, **params)
        p_dnn = art['dnn_model'].predict(art['feature_scaler'].transform(Xva),
                                         verbose=0).ravel()
        p_rf = rf_bs.predict(Xva)

        preds = {
            'BreedSight': p_ens,
            'BreedSight_DNN_only': p_dnn,
            'BreedSight_RF_only': p_rf,
        }
        for k in (3, 5):                       # PCA columns are ordered by variance
            rf_k = RandomForestRegressor(
                n_estimators=params['rf_n_estimators'], max_depth=params['rf_max_depth'],
                min_samples_split=5, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr[:, :k], ytr)
            preds[f'RF_top{k}PCs'] = rf_k.predict(Xva[:, :k])

        if include_tuned:
            preds['RF_tuned'] = TunedRF().fit(Xtr, ytr).predict(Xva)
            if HAVE_XGB:
                preds['XGB_tuned'] = TunedXGB().fit(Xtr, ytr).predict(Xva)

        # Reference baselines (all QC-passed markers, penalties from the training part)
        for name, cls in (('GBLUP_1', GBLUP_REML), ('GBLUP_PC3', GBLUP_PC3_REML),
                          ('Lasso_1', LassoInnerCV)):
            preds[name] = cls().fit(Ztr, ytr).predict(Zva)

        for name, p in preds.items():
            r2, r = _split_metrics(yva, p)
            rows.append({'Scheme': scheme, 'Split': i,
                         'Repeat': (i - 1) // n_splits_per_repeat + 1 if n_splits_per_repeat else 1,
                         'Model': name, 'Val_R2': r2, 'Val_r': r})
            print(f"    {name:22s} R2 = {r2:7.3f}   r = {r:6.3f}")

    per_split = pd.DataFrame(rows)
    order = list(dict.fromkeys(per_split['Model']))
    summary = (per_split.groupby('Model')
               .agg(Val_R2_mean=('Val_R2', 'mean'), Val_R2_sd=('Val_R2', 'std'),
                    Val_r_mean=('Val_r', 'mean'), Val_r_sd=('Val_r', 'std'),
                    N_splits=('Split', 'nunique'))
               .reindex(order).reset_index())
    summary.insert(0, 'Scheme', scheme)

    per_split.to_csv(os.path.join(output_dir, f"robustness_{scheme}_per_split.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, f"robustness_{scheme}_summary.csv"), index=False)
    print(f"\n=== {scheme}: summary (mean +/- SD over splits) ===")
    print(summary.round(3).to_string(index=False))
    return per_split, summary


def run_robustness_analysis(Z, y, bs_params, output_dir, n_repeats=5, n_splits=5,
                            n_clusters=10, include_tuned=True, run_repeated=True,
                            run_grouped=True):
    """Near-duplicate check + repeated K-fold + grouped CV (+ ablations / tuned baselines)."""
    out = {}
    if run_repeated:
        rkf = list(RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=RANDOM_STATE).split(Z))
        near_duplicate_check(Z, rkf[:n_splits], output_dir, tag="kfold")
        out['repeated'] = evaluate_on_splits(
            Z, y, rkf, bs_params, f"repeatedCV_{n_repeats}x{n_splits}", output_dir,
            include_tuned=include_tuned, n_splits_per_repeat=n_splits)
    if run_grouped:
        gsplits, groups = make_grouped_splits(Z, n_splits=n_splits, n_clusters=n_clusters)
        near_duplicate_check(Z, gsplits, output_dir, tag="grouped")
        out['grouped'] = evaluate_on_splits(
            Z, y, gsplits, bs_params, "groupedCV", output_dir,
            include_tuned=include_tuned)
    return out


def run_complete_analysis(training_file, training_additive_file, testing_file, testing_additive_file,
                         output_dir="output1", run_robustness=False, n_repeats=5,
                         n_clusters=10, include_tuned=True, **kwargs):
    """
    Complete genomic prediction pipeline with COMPLETE data leakage prevention.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "diagnostic_plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    
    print("=== Loading Data ===")
    # Load data
    training_data = pd.read_csv(training_file)
    training_additive = pd.read_csv(training_additive_file)
    testing_data = pd.read_csv(testing_file)
    testing_additive = pd.read_csv(testing_additive_file)
    
    print(f"Training data: {training_data.shape}")
    print(f"Training additive: {training_additive.shape}")
    print(f"Testing data: {testing_data.shape}")
    print(f"Testing additive: {testing_additive.shape}")
    
    # Run cross-validation
    print("\n=== Starting Cross-Validation ===")
    print("REVISION FIXES APPLIED:")
    print("  ✓ Outer hold-out fold never used for early stopping / LR scheduling")
    print("  ✓ Legacy baselines preserved; reviewer-recommended *_1 baselines added\n")
    
    # Merge improved parameters with user-provided kwargs
    improved_kwargs = IMPROVED_PARAMS.copy()
    improved_kwargs.update(kwargs)
    
    results = KFoldCrossValidation(
        training_data=training_data,
        training_additive=training_additive,
        testing_data=testing_data,
        testing_additive=testing_additive,
        **improved_kwargs
    )
    
    (results_df_dict, train_pred_df, val_pred_df, test_pred_final_df, metrics_df,
     final_predictions, final_models, test_metrics_df) = results
    
    # Save results
    print("\n=== Saving Results ===")
    
    # Save predictions
    train_pred_df.to_csv(os.path.join(output_dir, "training_predictions.csv"), index=False)
    val_pred_df.to_csv(os.path.join(output_dir, "validation_predictions.csv"), index=False)
    test_pred_final_df.to_csv(os.path.join(output_dir, "testing_predictions.csv"), index=False)
    
    # Save metrics
    metrics_df.to_csv(os.path.join(output_dir, "model_metrics.csv"), index=False)
    if test_metrics_df is not None:
        test_metrics_df.to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)
    
    # Save detailed results per model
    for model_name, df in results_df_dict.items():
        df.to_csv(os.path.join(output_dir, f"{model_name}_detailed_results.csv"), index=False)
    
    # Save per-fold REML / Lasso parameters (Tables S10-S12)
    if FOLD_PARAM_LOG:
        pd.DataFrame(FOLD_PARAM_LOG).to_csv(os.path.join(output_dir, "baseline_fold_parameters.csv"), index=False)

    # Save final models
    import joblib
    for model_name, model_data in final_models.items():
        if model_name == 'BreedSight':
            # Save BreedSight components
            breedSight_light = {
                'rf_model': model_data.get('rf_model'),
                'artifacts': model_data.get('artifacts'),
                'history': model_data.get('history')
            }
            joblib.dump(breedSight_light, os.path.join(output_dir, "models", "final_BreedSight.pkl"))
        else:
            # Save traditional models
            joblib.dump(model_data, os.path.join(output_dir, "models", f"final_{model_name}.pkl"))
    
    # Generate comprehensive summary report
    with open(os.path.join(output_dir, "analysis_summary.txt"), "w") as f:
        f.write("Genomic Prediction Analysis Summary\n")
        f.write("===================================\n\n")
        f.write("EVALUATION PROTOCOL:\n")
        f.write("- Early stopping / LR reduction use an inner split (15%) of each training fold;\n")
        f.write("  the outer hold-out fold is only used for prediction and metrics\n")
        f.write("- Legacy (as submitted; BreedSight feature space, fixed penalties):\n")
        f.write("    Lasso = Lasso(alpha=0.1); RBLUP = Ridge(alpha=1); GBLUP = KernelRidge(rbf, alpha=1)\n")
        f.write("- Reviewer-recommended (all QC-passed raw markers, data-driven penalties):\n")
        f.write("    GBLUP_1 = linear-kernel GRM + REML; RRBLUP_1 = ridge on markers + REML lambda;\n")
        f.write("    GBLUP_PC3 = GBLUP_1 + first 3 genomic PCs (training-fold GRM) as fixed effects\n")
        f.write("    Lasso_1 = Lasso, penalty from inner 5-fold CV over a 100-value alpha path\n")
        f.write("- BreedSight uses GRM-derived features ([GRM, GRM^2] -> 50 PCs). RFE is NOT applied\n")
        f.write("  (50 PCs <= rfe_n_features), so none is described as part of the method\n")
        f.write("- Final (test) BreedSight uses the same hyperparameters as the CV runs\n")
        f.write("- Identical outer folds for all models\n\n")
        
        f.write("IMPROVED MODEL PARAMETERS:\n")
        for key, value in IMPROVED_PARAMS.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")
        
        f.write(f"DATA SUMMARY:\n")
        f.write(f"Training samples: {training_data.shape[0]}\n")
        f.write(f"Testing samples: {testing_data.shape[0]}\n")
        f.write(f"Markers: {training_additive.shape[1] - 1}\n\n")
        
        f.write("MODEL PERFORMANCE SUMMARY:\n")
        f.write(metrics_df.to_string())
        
        # Add best model recommendation
        best_model = metrics_df.loc[metrics_df['Avg_Val_R2'].idxmax(), 'Model']
        best_val_r2 = metrics_df.loc[metrics_df['Avg_Val_R2'].idxmax(), 'Avg_Val_R2']
        f.write(f"\n\nRECOMMENDATION:\n")
        f.write(f"Best performing model: {best_model} (Validation R²: {best_val_r2:.4f})\n")
    
    # ---- Robustness analyses (repeated CV, grouped CV, ablations, tuned baselines) ----
    if run_robustness:
        v_train, _ = validate_genomic_data_comprehensive(
            training_data, training_additive, testing_data, testing_additive)
        Z_all = v_train.iloc[:, 1:].values
        y_all = training_data['phenotypes'].values
        bs_keys = ('epochs', 'batch_size', 'learning_rate', 'l2_reg', 'dropout_rate',
                   'rf_n_estimators', 'rf_max_depth', 'alpha')
        bs_params = {k: improved_kwargs[k] for k in bs_keys}
        bs_params['use_early_stopping'] = True
        run_robustness_analysis(Z_all, y_all, bs_params, output_dir, n_repeats=n_repeats,
                                n_splits=improved_kwargs.get('outer_n_splits', 5),
                                n_clusters=n_clusters, include_tuned=include_tuned)

    print("✅ Analysis complete! Results saved to:", output_dir)
    print("\n🔧 REVISION CHANGES APPLIED:")
    print("  1. ✅ Inner-split early stopping (no outer-fold model selection)")
    print("  2. ✅ Legacy baselines kept; RRBLUP_1 / GBLUP_1 / GBLUP_PC3 / Lasso_1 added")
    print("  3. ✅ Bootstrap confidence intervals for all metrics")
    
    return results

# Example usage
if __name__ == "__main__":
    # ---- choose the trial / location (edit the file paths below accordingly) ----
    LOCATION = "Hyderabad"                                   # or "Varanasi"
    LOCATION_H2 = {"Hyderabad": 0.83, "Varanasi": 0.81}     # per-location heritability

    training_file_path = "training_phenotypic_data.csv"
    training_additive_file_path = "training_additive.csv"
    testing_file_path = "testing_data.csv"
    testing_additive_file_path = "testing_additive.csv"

    RUN_ROBUSTNESS = True    # repeated CV + grouped CV + ablations + tuned baselines (slow)
    N_REPEATS = 5            # 5 x 5-fold; lower (e.g. 2) for a quick first pass

    try:
        results = run_complete_analysis(
            training_file=training_file_path,
            training_additive_file=training_additive_file_path,
            testing_file=testing_file_path,
            testing_additive_file=testing_additive_file_path,
            output_dir=f"output_{LOCATION}",
            # Cross-validation parameters
            outer_n_splits=5,
            feature_selection=False,   # RFE was never active: 50 PCs <= rfe_n_features
            heritability=LOCATION_H2[LOCATION],
            use_raw_genotypes=False,
            use_pca=True,
            n_components=50,
            rfe_n_features=100,
            generate_plots=True,
            # Robustness analyses
            run_robustness=RUN_ROBUSTNESS,
            n_repeats=N_REPEATS,
            n_clusters=10,
            include_tuned=True
        )
        print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")

    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your input data for missing values or formatting issues.")
