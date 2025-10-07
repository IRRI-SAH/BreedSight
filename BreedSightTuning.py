# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 11:20:01 2025

@author: Ashmi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 10:40:28 2025

@author: Ashmi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 12 15:06:39 2025

@author: Ashmitha
"""


############################################### Libraries ##############################################################
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import tempfile
import os
import joblib
from sklearn.utils import shuffle
import warnings
import copy
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

####################################### Set all random seeds for reproducibility ########################################
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

set_seeds()

RANDOM_STATE = 42

########################################################### Model ###################################################################
def BreedSightTuning(trainX, trainy, valX=None, valy=None, testX=None, testy=None, 
            epochs=1, batch_size=64, learning_rate=0.0001, 
            l2_reg=0.0001, dropout_rate=0.5, 
            rf_n_estimators=200, rf_max_depth=42, 
            alpha=0.6, verbose=1, model_save_path=None, rf_save_path=None):
    
    # Initialize results
    predicted_test = None
    
    # ==================== Data Validation Checks ====================
    # Check for NaN values
    if np.isnan(trainX).any() or np.isnan(trainy).any():
        raise ValueError("Training data contains NaN values")
    
    if valX is not None and (np.isnan(valX).any() or (valy is not None and np.isnan(valy).any())):
        raise ValueError("Validation data contains NaN values")
    
    if testX is not None and (np.isnan(testX).any() or (testy is not None and np.isnan(testy).any())):
        raise ValueError("Test data contains NaN values")
    
    # Note: Scaling is now handled outside this function to avoid repetition
    
    # ==================== Model Architecture ====================
    def build_fnn_model(input_shape):
        inputs = tf.keras.Input(shape=(input_shape,))   
        x = Dense(512, kernel_initializer='he_normal', 
                 kernel_regularizer=regularizers.l2(l2_reg))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # First residual block
        res = x
        x = Dense(256, kernel_initializer='he_normal', 
                 kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)   
        
        if res.shape[-1] != x.shape[-1]:
            res = Dense(256, kernel_initializer='he_normal', 
                       kernel_regularizer=regularizers.l2(l2_reg))(res)
        x = Add()([x, res])
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Second residual block
        res = x
        x = Dense(128, kernel_initializer='he_normal', 
                 kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)   
        
        if res.shape[-1] != x.shape[-1]:
            res = Dense(128, kernel_initializer='he_normal', 
                       kernel_regularizer=regularizers.l2(l2_reg))(res)
        x = Add()([x, res])
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Final layers
        x = Dense(64, kernel_initializer='he_normal', 
                 kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Dense(32, kernel_initializer='he_normal', 
                 kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        outputs = Dense(1, activation="relu")(x)
        model = tf.keras.Model(inputs, outputs)
        
        model.compile(loss=tf.keras.losses.Huber(delta=0.1), 
                     optimizer=Adam(learning_rate=learning_rate, clipvalue=0.1), 
                     metrics=['mse'])
        return model
    
    fnn_model = build_fnn_model(trainX.shape[1])
    
    # ==================== Model Training ====================
    callbacks = [
        EarlyStopping(monitor='val_loss', verbose=verbose, 
                     restore_best_weights=True, patience=15, mode='min')
    ]
    
    if valX is not None and valy is not None:
        validation_data = (valX, valy)
    else:
        validation_data = None
    
    # Always set validation_split=0.0 to disable internal splitting
    history = fnn_model.fit(
        trainX, 
        trainy, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=validation_data,
        validation_split=0.0,
        verbose=verbose, 
        callbacks=callbacks,
        shuffle=True
    )
    
    # ==================== Random Forest Training ====================
    rf_model = RandomForestRegressor(
        n_estimators=rf_n_estimators, 
        max_depth=rf_max_depth, 
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(trainX, trainy.ravel())
    
    # Save Models if paths provided
    if model_save_path:
        tf.keras.models.save_model(
            fnn_model,
            model_save_path,
            overwrite=True,
            include_optimizer=True,
            save_format='tf'
        )
    
    if rf_save_path:
        joblib.dump(rf_model, rf_save_path)
    
    # ==================== Making Predictions ====================
    # FNN predictions
    predicted_train_fnn = fnn_model.predict(trainX).flatten()
    predicted_val_fnn = fnn_model.predict(valX).flatten() if valX is not None else None
    predicted_test_fnn = fnn_model.predict(testX).flatten() if testX is not None else None
    
    # RF predictions
    predicted_train_rf = rf_model.predict(trainX)
    predicted_val_rf = rf_model.predict(valX) if valX is not None else None
    predicted_test_rf = rf_model.predict(testX) if testX is not None else None
    
    # ==================== Ensemble Predictions ====================
    predicted_train = alpha * predicted_train_fnn + (1 - alpha) * predicted_train_rf
    predicted_val = alpha * predicted_val_fnn + (1 - alpha) * predicted_val_rf if valX is not None else None
    predicted_test = alpha * predicted_test_fnn + (1 - alpha) * predicted_test_rf if testX is not None else None
    
    if verbose > 0:
        print("\n=== Training Summary ===")
        print(f"Train samples: {len(trainX)}, Validation samples: {len(valX) if valX is not None else 'N/A'}")
        if valX is not None and valy is not None:
            mse, rmse, corr, r2 = calculate_metrics(valy, predicted_val)
            print(f"Validation Metrics - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Corr: {corr:.4f}")
        
    return predicted_train, predicted_val, predicted_test, history, rf_model
##################################################### Genomic features #############################################################
def compute_genomic_features(X, sample_ids=None, ref_features=None, 
                            is_train=False, train_ids=None, train_X=None):
    """
    Compute genomic relationship features with strict data leakage prevention
    
    For training data: Compute standard G matrix as X*X.T
    For validation/test data: Compute relationship with training data only
    
    Args:
        X: Input data (numpy array of shape [n_samples, n_markers])
        sample_ids: List of sample IDs for X
        ref_features: Reference features from training data (dict)
        is_train: Whether X is training data (bool)
        train_ids: List of training sample IDs (for validation/test checks)
        train_X: Training data matrix (for validation/test relationships)
    
    Returns:
        X_final: Computed genomic features
        ref_features: Updated reference features (for training) or None (for validation/test)
    """
    
    # Validate inputs to prevent data leakage
    if not is_train:
        if sample_ids is not None and train_ids is not None:
            # Check for overlapping samples between training and validation/test sets
            overlap = set(sample_ids) & set(train_ids)
            if overlap:
                raise ValueError(f"Data leakage detected: {len(overlap)} samples appear in both training and validation/test sets")
        
        if ref_features is None:
            raise ValueError("Reference features must be provided for validation/test data")
        
        # Deep copy ref_features to prevent any modification
        ref_features = copy.deepcopy(ref_features)

    if is_train:
        if ref_features is not None:
            raise ValueError("ref_features must be None for training data")
            
        # TRAINING DATA PROCESSING
        # Scale the markers using training data only
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_markers = X_scaled.shape[1]
        
        # Compute genomic relationship matrix for training data
        G_train = np.dot(X_scaled, X_scaled.T) / n_markers
        
        # Compute quadratic form of G (interaction effects)
        I_train = G_train * G_train
        
        # Normalize I matrix by mean diagonal value
        mean_diag = np.mean(np.diag(I_train))
        I_train_norm = I_train / mean_diag if mean_diag > 0 else I_train
        
        # Combine features
        X_final = np.concatenate([G_train, I_train_norm], axis=1)
        
        # Store reference features for validation/test
        ref_features = {
            'scaler': scaler,  # Store the fitted scaler
            'mean_diag': mean_diag,
            'n_markers': n_markers,
            'X_train_scaled': X_scaled  # Store scaled training data
        }
    
    else:
        # VALIDATION/TEST DATA PROCESSING
        # Check marker dimension consistency
        if X.shape[1] != ref_features['n_markers']:
            raise ValueError(f"Validation data has {X.shape[1]} markers, expected {ref_features['n_markers']}")
        
        # Scale validation/test data using training data scaler
        X_scaled = ref_features['scaler'].transform(X)
        n_markers = ref_features['n_markers']
        
        # Compute genomic relationship between validation and training samples only
        G_val_train = np.dot(X_scaled, ref_features['X_train_scaled'].T) / n_markers
        
        # Compute quadratic form using training parameters
        I_val_train = G_val_train * G_val_train
        I_val_train_norm = I_val_train / ref_features['mean_diag'] if ref_features['mean_diag'] > 0 else I_val_train
        
        # Combine features - each validation sample characterized by relationship to training samples
        X_final = np.concatenate([G_val_train, I_val_train_norm], axis=1)
    
    return X_final, ref_features
################################################### Calculating metrics ##########################################################
def calculate_metrics(true_values, predicted_values):
    """Compute performance metrics between true and predicted values"""
    mask = ~np.isnan(predicted_values)
    if np.sum(mask) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    true_values = true_values[mask]
    predicted_values = predicted_values[mask]
    
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    corr = pearsonr(true_values, predicted_values)[0]
    return mse, rmse, corr, r2

###################################################### Validate Sample IDs ##################################################
def validate_sample_ids(df, dataset_name, id_column=0):
    """Validate sample IDs for uniqueness and no missing values"""
    sample_ids = df.iloc[:, id_column]
    if sample_ids.duplicated().any():
        raise ValueError(f"Duplicate sample IDs found in {dataset_name}")
    if sample_ids.isna().any():
        raise ValueError(f"Missing sample IDs found in {dataset_name}")

###################################################### Cross validation ##################################################

def KFoldCrossValidation(training_data, training_additive, val_data=None, val_additive=None,
                        epochs=1, learning_rate=0.0001, batch_size=2,
                        l2_reg=0.0001, dropout_rate=0.5, rf_n_estimators=200,
                        rf_max_depth=42, alpha=0.6, outer_n_splits=2, 
                        feature_selection=True, save_models=True, model_dir='saved_models',
                        verbose=1):# 	Increase outer_n_splits=10
    """
    
    Parameters:
    -----------
    training_data : pd.DataFrame
        DataFrame containing sample IDs and phenotypes (columns: 'sample_id', 'phenotypes')
    training_additive : pd.DataFrame
        DataFrame containing additive genomic data (columns: 'sample_id', marker1, marker2...)
    val_data : pd.DataFrame, optional
        Optional validation data with same structure as training_data
    val_additive : pd.DataFrame, optional
        Optional validation additive data with same structure as training_additive
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for Adam optimizer
    batch_size : int
        Batch size for training
    l2_reg : float
        L2 regularization strength
    dropout_rate : float
        Dropout rate
    rf_n_estimators : int
        Number of trees in Random Forest
    rf_max_depth : int
        Max depth for Random Forest
    alpha : float
        Weight for FNN in ensemble (1-alpha for RF)
    outer_n_splits : int
        Number of outer CV folds
    feature_selection : bool
        Whether to perform feature selection
    save_models : bool
        Whether to save models for each fold
    model_dir : str
        Directory to save models
    verbose : int
        Verbosity level (0: silent, 1: progress, 2: detailed)
        
    Returns:
    --------
    dict
        Dictionary containing all results, predictions, and plots
    """
    
    # ==================== Initial Validation Checks ====================
    if not isinstance(training_data, pd.DataFrame) or not isinstance(training_additive, pd.DataFrame):
        raise ValueError("Training data must be pandas DataFrames")
        
    if 'phenotypes' not in training_data.columns:
        raise ValueError("Training data must contain 'phenotypes' column")
        
    if training_data.shape[0] != training_additive.shape[0]:
        raise ValueError("Mismatch in number of samples between training data and additive data")
        
    # Validate sample IDs
    validate_sample_ids(training_data, "training data")
    
    # Check for validation data consistency if provided
    if val_data is not None or val_additive is not None:
        if val_data is None or val_additive is None:
            raise ValueError("Must provide both val_data and val_additive or neither")
            
        if not isinstance(val_data, pd.DataFrame) or not isinstance(val_additive, pd.DataFrame):
            raise ValueError("Validation data must be pandas DataFrames")
            
        if val_data.shape[0] != val_additive.shape[0]:
            raise ValueError("Mismatch in number of samples between validation data and additive data")
            
        if training_additive.shape[1] != val_additive.shape[1]:
            raise ValueError("Mismatch in number of features between training and validation additive data")
        
        validate_sample_ids(val_data, "validation data")
            
        # Check for overlapping samples
        train_ids = set(training_data.iloc[:, 0].values)
        val_ids = set(val_data.iloc[:, 0].values)
        overlap = train_ids & val_ids
        if overlap:
            raise ValueError(f"Data leakage: {len(overlap)} samples appear in both training and validation sets")

    # ==================== Data Preparation ====================
    # Extract numpy arrays
    training_additive_raw = training_additive.iloc[:, 1:].values
    phenotypic_info = training_data['phenotypes'].values
    train_sample_ids = training_data.iloc[:, 0].values
    
    # Prepare validation data if provided
    val_additive_raw = None
    phenotypic_val_info = None
    val_sample_ids = None
    has_val_phenotypes = False
    
    if val_data is not None and val_additive is not None:
        val_additive_raw = val_additive.iloc[:, 1:].values
        has_val_phenotypes = 'phenotypes' in val_data.columns
        phenotypic_val_info = val_data['phenotypes'].values if has_val_phenotypes else None
        val_sample_ids = val_data.iloc[:, 0].values

    # ==================== Outer CV Loop ====================
    outer_kf = KFold(n_splits=outer_n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = []
    train_predictions = []
    val_predictions = []
    feature_importances = []
    
    for outer_fold, (outer_train_index, outer_val_index) in enumerate(outer_kf.split(training_additive_raw), 1):
        if verbose >= 1:
            print(f"\n=== Outer Fold {outer_fold}/{outer_n_splits} ===")
        
        # Split data for current fold
        outer_trainX = training_additive_raw[outer_train_index]
        outer_valX = training_additive_raw[outer_val_index]
        outer_trainy = phenotypic_info[outer_train_index]
        outer_valy = phenotypic_info[outer_val_index]
        
        # Sample IDs for current fold
        fold_train_ids = train_sample_ids[outer_train_index]
        fold_val_ids = train_sample_ids[outer_val_index]
        
        # ==================== Feature Engineering ====================
        # Compute genomic features for current fold training data
        X_train_genomic, ref_features = compute_genomic_features(
            outer_trainX, 
            sample_ids=fold_train_ids,
            ref_features=None, 
            is_train=True
        )
        
        # Compute genomic features for current fold validation data
        X_val_genomic, _ = compute_genomic_features(
            outer_valX,
            sample_ids=fold_val_ids,
            ref_features=ref_features,
            is_train=False,
            train_ids=fold_train_ids
        )
        
        # ==================== Feature Selection ====================
        if feature_selection:
            if verbose >= 2:
                print("Performing feature selection...")
                
            selector = SelectFromModel(
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=RANDOM_STATE + outer_fold,
                    n_jobs=-1
                ), 
                threshold="1.25*median"
            )
            
            # Only fit on training portion of the fold
           
            selector.fit(X_train_genomic, outer_trainy)
            selected_features = selector.get_support()
            
            if np.sum(selected_features) == 0:
                if verbose >= 1:
                    print("Warning: No features selected in this fold, using all features")
                X_train_final = X_train_genomic
                X_val_final = X_val_genomic
            else:
                X_train_final = selector.transform(X_train_genomic)
                X_val_final = selector.transform(X_val_genomic)
                
            # Store feature importance information
            if hasattr(selector.estimator_, 'feature_importances_'):
                feature_importances.append({
                    'fold': outer_fold,
                    'importances': selector.estimator_.feature_importances_,
                    'selected_features': selected_features
                })
        else:
            X_train_final = X_train_genomic
            X_val_final = X_val_genomic
        
        # ==================== Target Scaling (Once per Fold) ====================
        target_scaler = StandardScaler()
        outer_trainy_scaled = target_scaler.fit_transform(outer_trainy.reshape(-1, 1)).flatten()
        outer_valy_scaled = target_scaler.transform(outer_valy.reshape(-1, 1)).flatten()
        
        # ==================== Model Training ====================
        # Create model paths if saving
        model_path = None
        rf_path = None
        if save_models:
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'BreedSight_fold_{outer_fold}.h5')
            rf_path = os.path.join(model_dir, f'rf_fold_{outer_fold}.joblib')
        
        # Train model (pass scaled targets directly, no internal scaling)
        pred_train_scaled, pred_val_scaled, _, history, rf_model = BreedSightTuning(
            trainX=X_train_final, 
            trainy=outer_trainy_scaled,
            valX=X_val_final,
            valy=outer_valy_scaled,
            testX=None,
            testy=None,
            epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            alpha=alpha,
            verbose=verbose,
            model_save_path=model_path,
            rf_save_path=rf_path
        )
        
        # Inverse transform predictions
        pred_train = target_scaler.inverse_transform(pred_train_scaled.reshape(-1, 1)).flatten()
        pred_val = target_scaler.inverse_transform(pred_val_scaled.reshape(-1, 1)).flatten()
        
        # Store predictions
        train_predictions.append(pd.DataFrame({
            'Sample_ID': fold_train_ids,
            'True_Phenotype': outer_trainy,
            'Predicted_Phenotype': pred_train,
            'Fold': outer_fold
        }))
        
        val_predictions.append(pd.DataFrame({
            'Sample_ID': fold_val_ids,
            'True_Phenotype': outer_valy,
            'Predicted_Phenotype': pred_val,
            'Fold': outer_fold
        }))
        
        # Calculate metrics
        mse_train, rmse_train, corr_train, r2_train = calculate_metrics(outer_trainy, pred_train)
        mse_val, rmse_val, corr_val, r2_val = calculate_metrics(outer_valy, pred_val)
        
        results.append({
            'Fold': outer_fold,
            'Train_MSE': mse_train, 'Train_RMSE': rmse_train,
            'Train_R2': r2_train, 'Train_Corr': corr_train,
            'Val_MSE': mse_val, 'Val_RMSE': rmse_val,
            'Val_R2': r2_val, 'Val_Corr': corr_val,
            'N_Features': X_train_final.shape[1],
            'Epochs': len(history.history['loss'])
        })
        
        if verbose >= 1:
            print(f"Fold {outer_fold} Results:")
            print(f"  Training:   R² = {r2_train:.4f}, Corr = {corr_train:.4f}, RMSE = {rmse_train:.4f}")
            print(f"  Validation: R² = {r2_val:.4f}, Corr = {corr_val:.4f}, RMSE = {rmse_val:.4f}")
    
    # ==================== Final Model Training (if validation data provided) ====================
    test_pred_final_df = pd.DataFrame()
    final_model_metrics = {}
    
    if val_data is not None and val_additive is not None:
        if verbose >= 1:
            print("\n=== Training Final model on ALL training data ===")
        
        # Compute genomic features for full training data
        X_train_genomic, ref_features = compute_genomic_features(
            training_additive_raw, 
            sample_ids=train_sample_ids,
            ref_features=None, 
            is_train=True
        )
        y_train_raw = phenotypic_info
        
        # Compute features for test data
        X_test_genomic, _ = compute_genomic_features(
            val_additive_raw,
            sample_ids=val_sample_ids,
            ref_features=ref_features,
            is_train=False,
            train_ids=train_sample_ids
        )

        # Feature selection for final model
        if feature_selection:
            if verbose >= 2:
                print("Performing final feature selection...")
                
            final_selector = SelectFromModel(
                RandomForestRegressor(
                    n_estimators=100, 
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ), 
                max_features=100
            )
            y_final_selector = StandardScaler().fit_transform(y_train_raw.reshape(-1, 1)).flatten()
            final_selector.fit(X_train_genomic, y_final_selector)
            selected_features = final_selector.get_support()
            
            if np.sum(selected_features) == 0:
                raise ValueError("Feature selector selected zero features")
            
            X_train_final = final_selector.transform(X_train_genomic)
            X_test_final = final_selector.transform(X_test_genomic)
        else:
            X_train_final = X_train_genomic
            X_test_final = X_test_genomic

        # Target scaling for final model (once)
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        
        # Shuffle data
        X_train_final, y_train_scaled = shuffle(X_train_final, y_train_scaled, random_state=RANDOM_STATE)

        # Train final model (no val, no testy during training)
        final_model_path = None
        final_rf_path = None
        if save_models:
            final_model_path = os.path.join(model_dir, 'BreedSight_final.h5')
            final_rf_path = os.path.join(model_dir, 'rf_final.joblib')

        _, _, pred_test_scaled, final_history, _ = BreedSightTuning(
            trainX=X_train_final, 
            trainy=y_train_scaled,
            valX=None,
            valy=None,
            testX=X_test_final,
            testy=None,  # Explicitly None to avoid misuse
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            alpha=alpha,
            verbose=verbose,
            model_save_path=final_model_path,
            rf_save_path=final_rf_path
        )
        
        # Inverse transform test predictions
        pred_test = target_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).flatten()
        
        test_pred_final_df = pd.DataFrame({
            'Sample_ID': val_sample_ids,
            'Predicted_Phenotype': pred_test,
            'Model': 'Final'
        })
        
        if has_val_phenotypes:
            test_pred_final_df['True_Phenotype'] = phenotypic_val_info
        
        if has_val_phenotypes:
            mse_test_final, rmse_test_final, corr_test_final, r2_test_final = calculate_metrics(
                phenotypic_val_info, pred_test
            )
            final_model_metrics = {
                'Test_MSE': mse_test_final,
                'Test_RMSE': rmse_test_final,
                'Test_R2': r2_test_final,
                'Test_Corr': corr_test_final,
                'N_Features': X_train_final.shape[1],
                'Epochs': len(final_history.history['loss'])
            }
            
            if verbose >= 1:
                print(f"\n=== Final Test Results ===")
                print(f"MSE: {mse_test_final:.4f}, RMSE: {rmse_test_final:.4f}")
                print(f"R²: {r2_test_final:.4f}, Correlation: {corr_test_final:.4f}")
    
    # ==================== Results Compilation ====================
    # Combine results
    train_pred_df = pd.concat(train_predictions, ignore_index=True)
    val_pred_df = pd.concat(val_predictions, ignore_index=True)
    results_df = pd.DataFrame(results)
    
    if final_model_metrics:
        final_results_df = pd.DataFrame([{'Fold': 'Final_Model', **final_model_metrics}])
        results_df = pd.concat([results_df, final_results_df], ignore_index=True)
    
    # Generate plots
    def generate_plot(true_vals, pred_vals, title, is_test=False):
        plt.figure(figsize=(8, 6))
        if is_test and not has_val_phenotypes:
            pred_values = pred_vals.dropna()
            if len(pred_values) > 0:
                plt.hist(pred_values, bins=30)
                plt.xlabel('Predicted Phenotype')
                plt.ylabel('Frequency')
            else:
                plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
        else:
            valid_mask = (~pd.isna(pred_vals)) & (~pd.isna(true_vals))
            if valid_mask.any():
                plt.scatter(true_vals[valid_mask], pred_vals[valid_mask], alpha=0.5)
                plt.xlabel('True Phenotype')
                plt.ylabel('Predicted Phenotype')
                coef = np.polyfit(true_vals[valid_mask], pred_vals[valid_mask], 1)
                poly1d_fn = np.poly1d(coef)
                plt.plot(true_vals[valid_mask], poly1d_fn(true_vals[valid_mask]), '--k')
                
                # Add metrics to plot
                mse, rmse, corr, r2 = calculate_metrics(true_vals[valid_mask], pred_vals[valid_mask])
                plt.text(0.05, 0.9, f'R² = {r2:.3f}\nCorr = {corr:.3f}\nRMSE = {rmse:.3f}',
                         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
            else:
                plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
        
        plt.title(title)
        plot_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_file
    
    train_plot_file = generate_plot(train_pred_df['True_Phenotype'], 
                                  train_pred_df['Predicted_Phenotype'], 
                                  'Training Set Predictions')
    
    val_plot_file = generate_plot(val_pred_df['True_Phenotype'], 
                                val_pred_df['Predicted_Phenotype'], 
                                'Validation Set Predictions')
    
    test_plot_file = None
    if not test_pred_final_df.empty:
        test_plot_file = generate_plot(test_pred_final_df.get('True_Phenotype', None),
                                      test_pred_final_df['Predicted_Phenotype'],
                                      'Test Set Predictions (Final Model)', 
                                      is_test=not has_val_phenotypes)
    
    # Save predictions to temporary files with unique names
    def save_to_temp(df, prefix, fold=None):
        try:
            temp_dir = tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)
            timestamp = int(time.time())
            fold_str = f"_fold{fold}" if fold is not None else ""
            path = os.path.join(temp_dir, f"{prefix}_predictions_{timestamp}{fold_str}.csv")
            df.to_csv(path, index=False)
            return path
        except Exception as e:
            print(f"Warning: Failed to save {prefix} predictions: {e}")
            return None
    
    train_csv = save_to_temp(train_pred_df, "train")
    val_csv = save_to_temp(val_pred_df, "val")
    test_csv = save_to_temp(test_pred_final_df, "test") if not test_pred_final_df.empty else None
    
    # Feature importance analysis
    feature_importance_df = None
    if feature_selection and feature_importances:
        try:
            # Create a comprehensive feature importance dataframe
            all_features = training_additive.columns[1:]  # Exclude sample_id column
            importance_data = []
            
            for fold_data in feature_importances:
                for feat_idx, (feat_name, importance) in enumerate(zip(all_features, fold_data['importances'])):
                    importance_data.append({
                        'Fold': fold_data['fold'],
                        'Feature': feat_name,
                        'Importance': importance,
                        'Selected': fold_data['selected_features'][feat_idx]
                    })
            
            feature_importance_df = pd.DataFrame(importance_data)
        except Exception as e:
            print(f"Warning: Could not create feature importance dataframe: {e}")
    
    # Return comprehensive results
    return {
        'results': results_df,
        'train_predictions': train_pred_df,
        'val_predictions': val_pred_df,
        'test_predictions': test_pred_final_df,
        'train_plot': train_plot_file,
        'val_plot': val_plot_file,
        'test_plot': test_plot_file,
        'train_csv': train_csv,
        'val_csv': val_csv,
        'test_csv': test_csv,
        'feature_importances': feature_importance_df
    }
###################################################### Hyperparameter tuning #############################################
def tune_hyperparameters(outer_train_data, outer_train_additive, param_grid, 
                        feature_selection=True, inner_n_splits=2): #increase inner_n_splits=5
    
    # Data validation
    assert isinstance(outer_train_data, pd.DataFrame), "Training data must be DataFrame"
    assert isinstance(outer_train_additive, pd.DataFrame), "Training additive data must be DataFrame"
    
    # Prepare data
    training_additive_raw = outer_train_additive.iloc[:, 1:].values
    phenotypic_info = outer_train_data['phenotypes'].values
    
    inner_kf = KFold(n_splits=inner_n_splits, shuffle=True, random_state=RANDOM_STATE)
    inner_results = []
    
    for params in ParameterGrid(param_grid):
        inner_scores = []
        inner_thresholds = []
        
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_kf.split(training_additive_raw), 1):
            # Split inner fold data
            X_inner_train_raw = training_additive_raw[inner_train_idx]
            y_inner_train = phenotypic_info[inner_train_idx]
            X_inner_val_raw = training_additive_raw[inner_val_idx]
            y_inner_val = phenotypic_info[inner_val_idx]
            
            inner_train_ids = outer_train_data.iloc[inner_train_idx, 0].values
            inner_val_ids = outer_train_data.iloc[inner_val_idx, 0].values
            
            # Compute genomic features for inner training data
            X_inner_train_genomic, ref_features = compute_genomic_features(
                X_inner_train_raw, 
                sample_ids=inner_train_ids,
                ref_features=None, 
                is_train=True
            )
            
            # Compute genomic features for inner validation data
            X_inner_val_genomic, _ = compute_genomic_features(
                X_inner_val_raw,
                sample_ids=inner_val_ids,
                ref_features=ref_features,
                is_train=False,
                train_ids=inner_train_ids
            )
            
            # Perform feature selection inside inner loop if enabled
            if feature_selection:
                selector = SelectFromModel(
                    RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE + inner_fold), 
                    threshold="median"
                )
                y_inner_selector = StandardScaler().fit_transform(y_inner_train.reshape(-1, 1)).flatten()
                selector.fit(X_inner_train_genomic, y_inner_selector)
                selected_features = selector.get_support()
                
                if np.sum(selected_features) == 0:
                    print("Warning: No features selected in inner fold, skipping")
                    continue
                
                X_inner_train_selected = selector.transform(X_inner_train_genomic)
                X_inner_val_selected = selector.transform(X_inner_val_genomic)
                inner_thresholds.append(selector.threshold_)
            else:
                X_inner_train_selected = X_inner_train_genomic
                X_inner_val_selected = X_inner_val_genomic
            
            # Target scaling for inner fold
            target_scaler = StandardScaler()
            y_inner_train_scaled = target_scaler.fit_transform(y_inner_train.reshape(-1, 1)).flatten()
            y_inner_val_scaled = target_scaler.transform(y_inner_val.reshape(-1, 1)).flatten()
            
            # Train model with current parameters
            _, val_pred_scaled, _, _, _ = BreedSightTuning(
                trainX=X_inner_train_selected, 
                trainy=y_inner_train_scaled,
                valX=X_inner_val_selected,
                valy=y_inner_val_scaled,
                testX=None,
                testy=None,
                epochs=1,  # Reduced for tuning speed
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                l2_reg=0.0001,
                dropout_rate=params['dropout_rate'],
                rf_n_estimators=params['rf_n_estimators'],
                rf_max_depth=42,
                alpha=0.6,
                verbose=0
            )
            
            # Inverse transform val pred
            val_pred = target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
            
            # Get validation score
            _, rmse, corr, r2 = calculate_metrics(y_inner_val, val_pred)
            inner_scores.append(r2)  # Using R² for optimization
        
        if inner_scores:  # Only store if we have valid scores
            mean_inner_score = np.mean(inner_scores)
            mean_threshold = np.mean(inner_thresholds) if inner_thresholds else None
            inner_results.append({
                'params': params,
                'mean_val_score': mean_inner_score,
                'mean_threshold': mean_threshold
            })
    
    if not inner_results:
        raise ValueError("No valid inner loop results - possibly all feature selections failed")
    
    # Select best parameters from inner loop
    best_inner_result = max(inner_results, key=lambda x: x['mean_val_score'])
    best_params = best_inner_result['params']
    best_threshold = best_inner_result['mean_threshold']
    
    return best_params, best_threshold, inner_results

###################################################### Running cross validation ##########################################

def run_cross_validation(training_file, training_additive_file, testing_file=None, testing_additive_file=None, 
                        feature_selection=True, outer_n_splits=10, inner_n_splits=5):
    """
    Main function to run nested cross-validation with hyperparameter tuning
    
    Args:
        training_file: Path to training data CSV or Gradio File object
        training_additive_file: Path to training additive data CSV or Gradio File object
        testing_file: Path to testing data CSV or Gradio File object (optional)
        testing_additive_file: Path to testing additive data CSV or Gradio File object (optional)
        feature_selection: Whether to perform feature selection
        outer_n_splits: Number of outer CV folds (default: 10)
        inner_n_splits: Number of inner CV folds (default: 5)
    
    Returns:
        Tuple of (train_predictions, val_predictions, test_predictions, train_plot, val_plot, test_plot, 
                  train_csv, val_csv, test_csv, tuning_results_df, feature_importance_df)
    """
    if not all([training_file, training_additive_file]):
        raise ValueError("At least training files must be provided")
    
    try:
        def get_path(file_obj):
            return file_obj.name if hasattr(file_obj, 'name') else file_obj
        
        training_data = pd.read_csv(get_path(training_file))
        training_additive = pd.read_csv(get_path(training_additive_file))
        
        has_test_data = testing_file is not None and testing_additive_file is not None
        if has_test_data:
            testing_data = pd.read_csv(get_path(testing_file))
            testing_additive = pd.read_csv(get_path(testing_additive_file))
        else:
            testing_data = None
            testing_additive = None
            
    except Exception as e:
        raise ValueError(f"Error reading CSV files: {e}")

    if 'phenotypes' not in training_data.columns:
        raise ValueError("Training data must contain 'phenotypes' column")
    if training_data.shape[0] != training_additive.shape[0]:
        raise ValueError("Mismatch in number of samples between training data and additive data")
    if has_test_data:
        if testing_data.shape[0] != testing_additive.shape[0]:
            raise ValueError("Mismatch in number of samples between testing data and additive data")
        if training_additive.shape[1] != testing_additive.shape[1]:
            raise ValueError("Mismatch in number of features between training and testing additive data")

    validate_sample_ids(training_data, "training data")
    if has_test_data:
        validate_sample_ids(testing_data, "testing data")
        train_ids = set(training_data.iloc[:, 0].values)
        test_ids = set(testing_data.iloc[:, 0].values)
        overlap = train_ids & test_ids
        if overlap:
            raise ValueError(f"Data leakage: {len(overlap)} samples appear in both training and testing sets")

    param_grid = {
        'learning_rate': [0.00005, 0.0001, 0.0005, 0.001, 0.005],
        'batch_size': [16, 32, 64, 128],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'rf_n_estimators': [100, 200, 300, 400]
    }

    outer_kf = KFold(n_splits=outer_n_splits, shuffle=True, random_state=RANDOM_STATE)
    outer_results = []
    train_predictions = []
    val_predictions = []
    feature_importances = []
    
    training_additive_raw = training_additive.iloc[:, 1:].values
    phenotypic_info = training_data['phenotypes'].values
    train_sample_ids = training_data.iloc[:, 0].values
    
    for outer_fold, (outer_train_idx, outer_val_idx) in enumerate(outer_kf.split(training_additive_raw), 1):
        print(f"\n=== Outer Fold {outer_fold}/{outer_n_splits} ===")
        
        outer_train_data = training_data.iloc[outer_train_idx]
        outer_train_additive = training_additive.iloc[outer_train_idx]
        outer_val_data = training_data.iloc[outer_val_idx]
        outer_val_additive = training_additive.iloc[outer_val_idx]
        
        best_params, best_threshold, tuning_results_fold = tune_hyperparameters(
            outer_train_data=outer_train_data,
            outer_train_additive=outer_train_additive,
            param_grid=param_grid,
            feature_selection=feature_selection,
            inner_n_splits=inner_n_splits
        )
        
        X_outer_train_genomic, ref_features = compute_genomic_features(
            outer_train_additive.iloc[:, 1:].values, 
            sample_ids=outer_train_data.iloc[:, 0].values, 
            ref_features=None, 
            is_train=True
        )
        y_outer_train = outer_train_data['phenotypes'].values
        
        X_outer_val_genomic, _ = compute_genomic_features(
            outer_val_additive.iloc[:, 1:].values,
            sample_ids=outer_val_data.iloc[:, 0].values,
            ref_features=ref_features,
            is_train=False,
            train_ids=outer_train_data.iloc[:, 0].values
        )
        y_outer_val = outer_val_data['phenotypes'].values
        
        if feature_selection:
            y_outer_selector = StandardScaler().fit_transform(y_outer_train.reshape(-1, 1)).flatten()
            selector = SelectFromModel(
                RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE + outer_fold), 
                threshold=best_threshold if best_threshold is not None else "1.25*median"
            )
            selector.fit(X_outer_train_genomic, y_outer_selector)
            selected_features = selector.get_support()
            
            if np.sum(selected_features) == 0:
                print(f"Warning: No features selected in outer fold {outer_fold}, using all features")
                X_outer_train_selected = X_outer_train_genomic
                X_outer_val_selected = X_outer_val_genomic
            else:
                X_outer_train_selected = selector.transform(X_outer_train_genomic)
                X_outer_val_selected = selector.transform(X_outer_val_genomic)
                
            if hasattr(selector.estimator_, 'feature_importances_'):
                feature_importances.append({
                    'fold': outer_fold,
                    'importances': selector.estimator_.feature_importances_,
                    'selected_features': selected_features
                })
        else:
            X_outer_train_selected = X_outer_train_genomic
            X_outer_val_selected = X_outer_val_genomic
        
        target_scaler = StandardScaler()
        y_outer_train_scaled = target_scaler.fit_transform(y_outer_train.reshape(-1, 1)).flatten()
        y_outer_val_scaled = target_scaler.transform(y_outer_val.reshape(-1, 1)).flatten()
        
        model_path = os.path.join('saved_models', f'BreedSight_fold_{outer_fold}.h5')
        rf_path = os.path.join('saved_models', f'rf_fold_{outer_fold}.joblib')
        os.makedirs('saved_models', exist_ok=True)
        
        pred_train_scaled, pred_val_scaled, _, _, _ = BreedSightTuning(
            trainX=X_outer_train_selected, 
            trainy=y_outer_train_scaled,
            valX=X_outer_val_selected,
            valy=y_outer_val_scaled,
            testX=None,
            testy=None,
            epochs=1,
            batch_size=best_params['batch_size'],
            learning_rate=best_params['learning_rate'],
            l2_reg=0.0001,
            dropout_rate=best_params['dropout_rate'],
            rf_n_estimators=best_params['rf_n_estimators'],
            rf_max_depth=42,
            alpha=0.6,
            verbose=1,
            model_save_path=model_path,
            rf_save_path=rf_path
        )
        
        pred_train = target_scaler.inverse_transform(pred_train_scaled.reshape(-1, 1)).flatten()
        pred_val = target_scaler.inverse_transform(pred_val_scaled.reshape(-1, 1)).flatten()
        
        train_predictions.append(pd.DataFrame({
            'Sample_ID': outer_train_data.iloc[:, 0],
            'True_Phenotype': y_outer_train,
            'Predicted_Phenotype': pred_train,
            'Fold': outer_fold
        }))
        
        val_predictions.append(pd.DataFrame({
            'Sample_ID': outer_val_data.iloc[:, 0],
            'True_Phenotype': y_outer_val,
            'Predicted_Phenotype': pred_val,
            'Fold': outer_fold
        }))
        
        mse_val, rmse_val, corr_val, r2_val = calculate_metrics(y_outer_val, pred_val)
        
        outer_results.append({
            'params': best_params,
            'mean_val_r2': r2_val,
            'mean_val_corr': corr_val,
            'mean_val_rmse': rmse_val,
            'outer_fold': outer_fold,
            'n_features': X_outer_train_selected.shape[1],
            'best_threshold': best_threshold
        })
        
        print(f"Fold {outer_fold} Results:")
        print(f"  Validation: R² = {r2_val:.4f}, Corr = {corr_val:.4f}, RMSE = {rmse_val:.4f}")
    
    test_pred_final_df = pd.DataFrame()
    final_model_metrics = {}
    
    if has_test_data:
        print("\n=== Training Final Model on ALL Training Data ===")
        
        param_counts = {}
        for result in outer_results:
            param_tuple = tuple(sorted(result['params'].items()))
            param_counts[param_tuple] = param_counts.get(param_tuple, 0) + 1
        best_params_tuple = max(param_counts, key=param_counts.get)
        best_params = dict(best_params_tuple)
        
        all_thresholds = [r['best_threshold'] for r in outer_results if r['best_threshold'] is not None]
        mean_final_threshold = np.mean(all_thresholds) if all_thresholds else "1.25*median"
        
        X_train_genomic, ref_features = compute_genomic_features(
            training_additive_raw, 
            sample_ids=train_sample_ids,
            ref_features=None, 
            is_train=True
        )
        y_train_raw = phenotypic_info
        
        X_test_genomic, _ = compute_genomic_features(
            testing_additive.iloc[:, 1:].values,
            sample_ids=testing_data.iloc[:, 0].values,
            ref_features=ref_features,
            is_train=False,
            train_ids=train_sample_ids
        )
        
        if feature_selection:
            print("Performing final feature selection...")
            final_selector = SelectFromModel(
                RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE), 
                threshold=mean_final_threshold
            )
            y_final_selector = StandardScaler().fit_transform(y_train_raw.reshape(-1, 1)).flatten()
            final_selector.fit(X_train_genomic, y_final_selector)
            selected_features = final_selector.get_support()
            
            if np.sum(selected_features) == 0:
                raise ValueError("Feature selector selected zero features")
            
            X_train_final = final_selector.transform(X_train_genomic)
            X_test_final = final_selector.transform(X_test_genomic)
        else:
            X_train_final = X_train_genomic
            X_test_final = X_test_genomic
        
        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        
        X_train_final, y_train_scaled = shuffle(X_train_final, y_train_scaled, random_state=RANDOM_STATE)
        
        final_model_path = os.path.join('saved_models', 'BreedSight_final.h5')
        final_rf_path = os.path.join('saved_models', 'rf_final.joblib')
        os.makedirs('saved_models', exist_ok=True)
        
        _, _, pred_test_scaled, final_history, _ = BreedSightTuning(
            trainX=X_train_final, 
            trainy=y_train_scaled,
            valX=None,
            valy=None,
            testX=X_test_final,
            testy=None,
            epochs=1,
            batch_size=best_params['batch_size'],
            learning_rate=best_params['learning_rate'],
            l2_reg=0.0001,
            dropout_rate=best_params['dropout_rate'],
            rf_n_estimators=best_params['rf_n_estimators'],
            rf_max_depth=42,
            alpha=0.6,
            verbose=1,
            model_save_path=final_model_path,
            rf_save_path=final_rf_path
        )
        
        pred_test = target_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).flatten()
        
        test_pred_final_df = pd.DataFrame({
            'Sample_ID': testing_data.iloc[:, 0],
            'Predicted_Phenotype': pred_test,
            'Model': 'Final'
        })
        
        has_test_phenotypes = 'phenotypes' in testing_data.columns
        if has_test_phenotypes:
            test_pred_final_df['True_Phenotype'] = testing_data['phenotypes']
            mse_test_final, rmse_test_final, corr_test_final, r2_test_final = calculate_metrics(
                testing_data['phenotypes'], pred_test
            )
            final_model_metrics = {
                'Test_MSE': mse_test_final,
                'Test_RMSE': rmse_test_final,
                'Test_R2': r2_test_final,
                'Test_Corr': corr_test_final,
                'N_Features': X_train_final.shape[1],
                'Epochs': len(final_history.history['loss'])
            }
            
            print(f"\n=== Final Test Results ===")
            print(f"MSE: {mse_test_final:.4f}, RMSE: {rmse_test_final:.4f}")
            print(f"R²: {r2_test_final:.4f}, Correlation: {corr_test_final:.4f}")
    
    train_pred_df = pd.concat(train_predictions, ignore_index=True)
    val_pred_df = pd.concat(val_predictions, ignore_index=True)
    results_df = pd.DataFrame(outer_results)
    
    if final_model_metrics:
        final_results_df = pd.DataFrame([{'Fold': 'Final_Model', **final_model_metrics}])
        results_df = pd.concat([results_df, final_results_df], ignore_index=True)
    
    def generate_plot(true_vals, pred_vals, title, is_test=False):
        plt.figure(figsize=(8, 6))
        if is_test and not has_test_phenotypes:
            pred_values = pred_vals.dropna()
            if len(pred_values) > 0:
                plt.hist(pred_values, bins=30)
                plt.xlabel('Predicted Phenotype')
                plt.ylabel('Frequency')
            else:
                plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
        else:
            valid_mask = (~pd.isna(pred_vals)) & (~pd.isna(true_vals))
            if valid_mask.any():
                plt.scatter(true_vals[valid_mask], pred_vals[valid_mask], alpha=0.5)
                plt.xlabel('True Phenotype')
                plt.ylabel('Predicted Phenotype')
                coef = np.polyfit(true_vals[valid_mask], pred_vals[valid_mask], 1)
                poly1d_fn = np.poly1d(coef)
                plt.plot(true_vals[valid_mask], poly1d_fn(true_vals[valid_mask]), '--k')
                
                mse, rmse, corr, r2 = calculate_metrics(true_vals[valid_mask], pred_vals[valid_mask])
                plt.text(0.05, 0.9, f'R² = {r2:.3f}\nCorr = {corr:.3f}\nRMSE = {rmse:.3f}',
                         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
            else:
                plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
        
        plt.title(title)
        plot_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_file
    
    train_plot_file = generate_plot(train_pred_df['True_Phenotype'], 
                                  train_pred_df['Predicted_Phenotype'], 
                                  'Training Set Predictions')
    
    val_plot_file = generate_plot(val_pred_df['True_Phenotype'], 
                                val_pred_df['Predicted_Phenotype'], 
                                'Validation Set Predictions')
    
    test_plot_file = None
    if not test_pred_final_df.empty:
        test_plot_file = generate_plot(test_pred_final_df.get('True_Phenotype', None),
                                      test_pred_final_df['Predicted_Phenotype'],
                                      'Test Set Predictions (Final Model)', 
                                      is_test=not has_test_phenotypes)
    
    def save_to_temp(df, prefix, fold=None):
        try:
            temp_dir = tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)
            timestamp = int(time.time())
            fold_str = f"_fold{fold}" if fold is not None else ""
            path = os.path.join(temp_dir, f"{prefix}_predictions_{timestamp}{fold_str}.csv")
            df.to_csv(path, index=False)
            return path
        except Exception as e:
            print(f"Warning: Failed to save {prefix} predictions: {e}")
            return None
    
    train_csv = save_to_temp(train_pred_df, "train")
    val_csv = save_to_temp(val_pred_df, "val")
    test_csv = save_to_temp(test_pred_final_df, "test") if not test_pred_final_df.empty else None
    
    tuning_results_df = pd.DataFrame(outer_results)
    
    feature_importance_df = None
    if feature_selection and feature_importances:
        try:
            all_features = training_additive.columns[1:]
            importance_data = []
            
            for fold_data in feature_importances:
                for feat_idx, (feat_name, importance) in enumerate(zip(all_features, fold_data['importances'])):
                    importance_data.append({
                        'Fold': fold_data['fold'],
                        'Feature': feat_name,
                        'Importance': importance,
                        'Selected': fold_data['selected_features'][feat_idx]
                    })
            
            feature_importance_df = pd.DataFrame(importance_data)
        except Exception as e:
            print(f"Warning: Could not create feature importance dataframe: {e}")
    
    return (
        train_pred_df,
        val_pred_df,
        test_pred_final_df if not test_pred_final_df.empty else None,
        train_plot_file,
        val_plot_file,
        test_plot_file,
        train_csv,
        val_csv,
        test_csv,
        tuning_results_df,
        feature_importance_df
    )
