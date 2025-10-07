# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 18:52:23 2025
Modified on Mon Oct 06 2025 to address data leakage issues

@author: Ashmitha
"""

############################# Required Packages ####################################
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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import tempfile
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer

################### Seed Default #################
RANDOM_STATE = 60
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

########################## Model Initialization #############

def BreedSight(trainX, trainy, valX=None, valy=None, testX=None, testy=None,
              epochs=1000, batch_size=64, learning_rate=0.0001,
              l2_reg=0.001, dropout_rate=0.5,
              rf_n_estimators=200, rf_max_depth=30,
              alpha=0.5, verbose=1):
    """
    Train a hybrid DNN and Random Forest model on pre-scaled data.
    Assumes input features and targets are already scaled to prevent leakage.
    """
    # Check for overlap between train and validation sets
    if valX is not None and valy is not None:
        if hasattr(trainX, 'index') and hasattr(valX, 'index'):
            if not set(trainX.index).isdisjoint(set(valX.index)):
                raise ValueError("Training and validation sets contain overlapping samples")
    
    # Build DNN Model
    def build_dnn_model(input_shape):
        inputs = tf.keras.Input(shape=(input_shape,))
        x = Dense(512, kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(l2_reg))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # First residual block
        res = x
        x = Dense(64, kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        if res.shape[-1] != x.shape[-1]:
            res = Dense(64, kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(l2_reg))(res)
        x = Add()([x, res])
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Second residual block
        res = x
        x = Dense(16, kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        if res.shape[-1] != x.shape[-1]:
            res = Dense(16, kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(l2_reg))(res)
        x = Add()([x, res])
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Final layers
        x = Dense(8, kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Dense(4, kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        outputs = Dense(1, activation="linear")(x)
        model = tf.keras.Model(inputs, outputs)
        
        model.compile(loss=tf.keras.losses.Huber(delta=0.1),
                      optimizer=Adam(learning_rate=learning_rate, clipvalue=0.1),
                      metrics=['mse'])
        return model
    
    dnn_model = build_dnn_model(trainX.shape[1])
    
    # Train DNN Model
    callbacks = [
        EarlyStopping(monitor='val_loss', verbose=verbose,
                      restore_best_weights=True, patience=20)
    ]
    
    validation_data = (valX, valy) if valX is not None and valy is not None else None
    
    history = dnn_model.fit(
        trainX,
        trainy,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=verbose,
        callbacks=callbacks
    )
    
    # Train Random Forest Model
    rf_model = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(trainX, trainy.ravel())
    
    # Make Predictions
    predicted_train_dnn = dnn_model.predict(trainX).flatten()
    predicted_val_dnn = dnn_model.predict(valX).flatten() if valX is not None else None
    predicted_test_dnn = dnn_model.predict(testX).flatten() if testX is not None else None
    
    predicted_train_rf = rf_model.predict(trainX)
    predicted_val_rf = rf_model.predict(valX) if valX is not None else None
    predicted_test_rf = rf_model.predict(testX) if testX is not None else None
    
    # Ensemble Predictions
    predicted_train = alpha * predicted_train_dnn + (1 - alpha) * predicted_train_rf
    predicted_val = alpha * predicted_val_dnn + (1 - alpha) * predicted_val_rf if valX is not None else None
    predicted_test = alpha * predicted_test_dnn + (1 - alpha) * predicted_test_rf if testX is not None else None
    
    if verbose > 0:
        print("\n=== Training Summary ===")
        print(f"Train samples: {len(trainX)}, Validation samples: {len(valX) if valX is not None else 'N/A'}")
    
    return predicted_train, predicted_val, predicted_test, history, rf_model, None

def compute_genomic_features(X, trainX=None, ref_features=None, is_train=False):
    """
    Compute genomic relationship features with strict separation to prevent leakage
    
    Parameters:
    -----------
    X: Input genomic data
    trainX: Training genomic data (required for non-training data)
    ref_features: Reference features from training data (None for training data)
    is_train: Boolean indicating if this is training data
    
    Returns:
    --------
    X_final: Transformed features
    ref_features: Dictionary of reference statistics (if is_train=True)
    """
    # Handle missing values with imputer
    imputer = SimpleImputer(strategy='mean')
    
    if is_train and ref_features is None:
        X = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_markers = X_scaled.shape[1]
        
        # Compute genomic relationship matrix for training data
        G_train = np.dot(X_scaled, X_scaled.T) / n_markers
        I_train = G_train * G_train
        
        mean_diag = np.mean(np.diag(I_train))
        I_train_norm = I_train / mean_diag if mean_diag != 0 else I_train
        
        X_final = np.concatenate([X_scaled, I_train_norm], axis=1)
        
        ref_features = {
            'imputer': imputer,
            'scaler': scaler,
            'mean_diag': mean_diag,
            'X_train_scaled': X_scaled
        }
        print(f"Training features processed: X_final shape {X_final.shape}")
        
    elif not is_train and ref_features is not None and trainX is not None:
        assert 'imputer' in ref_features, "ref_features must include 'imputer' for non-training data"
        # Check for overlap between X and trainX
        X_set = set(tuple(row) for row in X)
        trainX_set = set(tuple(row) for row in trainX)
        if X_set & trainX_set:
            raise ValueError("Input data X contains samples present in trainX")
        
        X = ref_features['imputer'].transform(X)
        X_scaled = ref_features['scaler'].transform(X)
        n_markers = ref_features['X_train_scaled'].shape[1]
        
        # Compute relationship with training samples only
        G_val = np.dot(X_scaled, ref_features['X_train_scaled'].T) / n_markers
        I_val = G_val * G_val
        
        I_val_norm = I_val / ref_features['mean_diag'] if ref_features['mean_diag'] != 0 else I_val
        
        X_final = np.concatenate([X_scaled, I_val_norm], axis=1)
        print(f"Non-training features processed: X_final shape {X_final.shape}")
    
    else:
        raise ValueError("Invalid combination of is_train, trainX, and ref_features parameters")
    
    assert X_final.shape[0] == X.shape[0], "Feature transformation changed number of samples"
    
    return X_final, ref_features

def calculate_metrics(true_values, predicted_values):
    """Compute performance metrics between true and predicted values"""
    mask = ~np.isnan(predicted_values) & ~np.isnan(true_values)
    if np.sum(mask) == 0:
        return np.nan, np.nan, np.nan, np.nan
    true_values = true_values[mask]
    predicted_values = predicted_values[mask]
    
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    corr, _ = pearsonr(true_values, predicted_values)
    return mse, rmse, corr, r2

def KFoldCrossValidation(training_data, training_additive, testing_data, testing_additive,
                        epochs=1000, learning_rate=0.0001, batch_size=64,
                        outer_n_splits=5, output_file='cross_validation_results.csv',
                        train_pred_file='train_predictions.csv',
                        val_pred_file='validation_predictions.csv',
                        test_pred_file='test_predictions.csv',
                        feature_selection=True):
    """
    Perform k-fold cross-validation with strict data leakage prevention
    """
    # Data Validation
    assert isinstance(training_data, pd.DataFrame), "Training data must be DataFrame"
    assert isinstance(testing_data, pd.DataFrame), "Testing data must be DataFrame"
    
    train_ids = set(training_data.iloc[:, 0].values)
    test_ids = set(testing_data.iloc[:, 0].values)
    assert len(train_ids & test_ids) == 0, "Training and testing sets must be distinct"
    
    train_features = training_additive.iloc[:, 1:].values
    test_features = testing_additive.iloc[:, 1:].values
    train_feature_set = set(tuple(row) for row in train_features)
    test_feature_set = set(tuple(row) for row in test_features)
    if train_feature_set & test_feature_set:
        raise ValueError("Training and testing feature sets contain identical rows")
    
    # Check for duplicate samples
    if len(train_feature_set) < train_features.shape[0]:
        raise ValueError("Training data contains duplicate feature rows")
    if len(test_feature_set) < test_features.shape[0]:
        raise ValueError("Testing data contains duplicate feature rows")
    
    # Data Preparation
    training_additive_raw = training_additive.iloc[:, 1:].values
    testing_additive_raw = testing_additive.iloc[:, 1:].values
    phenotypic_info = training_data['phenotypes'].values
    
    has_test_phenotypes = 'phenotypes' in testing_data.columns
    phenotypic_test_info = testing_data['phenotypes'].values if has_test_phenotypes else None
    test_sample_ids = testing_data.iloc[:, 0].values
    
    # Choose KFold or StratifiedKFold based on target distribution
    if np.std(phenotypic_info) > np.mean(phenotypic_info):
        discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        strata = discretizer.fit_transform(phenotypic_info.reshape(-1, 1)).flatten()
        outer_kf = StratifiedKFold(n_splits=outer_n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        outer_kf = KFold(n_splits=outer_n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    # Outer CV Loop
    results = []
    train_predictions = []
    val_predictions = []

    for outer_fold, (outer_train_index, outer_val_index) in enumerate(outer_kf.split(training_additive_raw, strata if 'strata' in locals() else None), 1):
        print(f"\n=== Outer Fold {outer_fold}/{outer_n_splits} ===")
        
        # Split data
        outer_trainX = training_additive_raw[outer_train_index]
        outer_valX = training_additive_raw[outer_val_index]
        outer_trainy = phenotypic_info[outer_train_index]
        outer_valy = phenotypic_info[outer_val_index]
        
        # Check for overlap in fold features
        train_fold_set = set(tuple(row) for row in outer_trainX)
        val_fold_set = set(tuple(row) for row in outer_valX)
        if train_fold_set & val_fold_set:
            raise ValueError(f"Overlapping features in train and val for fold {outer_fold}")
        
        # Feature scaling
        feature_scaler = StandardScaler()
        outer_trainX_scaled = feature_scaler.fit_transform(outer_trainX)
        outer_valX_scaled = feature_scaler.transform(outer_valX)
        
        # Target scaling
        target_scaler = StandardScaler()
        outer_trainy_scaled = target_scaler.fit_transform(outer_trainy.reshape(-1, 1)).flatten()
        outer_valy_scaled = target_scaler.transform(outer_valy.reshape(-1, 1)).flatten()
        
        # Process genomic features
        X_train_genomic, ref_features = compute_genomic_features(
            outer_trainX_scaled,
            ref_features=None,
            is_train=True
        )
        X_val_genomic, _ = compute_genomic_features(
            outer_valX_scaled,
            trainX=outer_trainX_scaled,
            ref_features=ref_features,
            is_train=False
        )
        
        # Feature selection
        if feature_selection:
            selector = SelectFromModel(
                RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
                threshold="median"
            )
            selector.fit(X_train_genomic, outer_trainy)
            X_train_final = selector.transform(X_train_genomic)
            X_val_final = selector.transform(X_val_genomic)
            print(f"Fold {outer_fold}: Selected {X_train_final.shape[1]} features")
        else:
            X_train_final = X_train_genomic
            X_val_final = X_val_genomic
        
        # Model training
        pred_train_scaled, pred_val_scaled, _, history, _, _ = BreedSight(
            trainX=X_train_final,
            trainy=outer_trainy_scaled,
            valX=X_val_final,
            valy=outer_valy_scaled,
            testX=None,
            testy=None,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=1
        )
        
        # Inverse transform predictions
        pred_train = target_scaler.inverse_transform(pred_train_scaled.reshape(-1, 1)).flatten()
        pred_val = target_scaler.inverse_transform(pred_val_scaled.reshape(-1, 1)).flatten()
        
        # Store predictions
        train_predictions.append(pd.DataFrame({
            'Sample_ID': training_data.iloc[outer_train_index, 0].values,
            'True_Phenotype': outer_trainy,
            'Predicted_Phenotype': pred_train,
            'Fold': outer_fold
        }))
        
        val_predictions.append(pd.DataFrame({
            'Sample_ID': training_data.iloc[outer_val_index, 0].values,
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
            'Val_R2': r2_val, 'Val_Corr': corr_val
        })
    
    # Final Model Training
    print("\n=== Training Final Model on ALL Training Data ===")
    
    X_train_raw = training_additive_raw
    y_train_raw = phenotypic_info
    
    # Feature scaling
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train_raw)
    X_test_scaled = feature_scaler.transform(testing_additive_raw)
    
    # Check test set isolation
    test_feature_set = set(tuple(row) for row in testing_additive_raw)
    train_feature_set = set(tuple(row) for row in X_train_raw)
    if train_feature_set & test_feature_set:
        raise ValueError("Test set contains samples present in training set")
    
    # Target scaling
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(phenotypic_test_info.reshape(-1, 1)).flatten() if has_test_phenotypes else None
    
    # Feature processing
    X_train_genomic, ref_features = compute_genomic_features(
        X_train_scaled,
        ref_features=None,
        is_train=True
    )
    X_test_genomic, _ = compute_genomic_features(
        X_test_scaled,
        trainX=X_train_scaled,
        ref_features=ref_features,
        is_train=False
    )

    # Feature selection
    if feature_selection:
        selector = SelectFromModel(
            RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
            threshold="median"
        )
        selector.fit(X_train_genomic, y_train_raw)
        X_train_final = selector.transform(X_train_genomic)
        X_test_final = selector.transform(X_test_genomic)
        print(f"Final model: Selected {X_train_final.shape[1]} features")
    else:
        X_train_final = X_train_genomic
        X_test_final = X_test_genomic

    # Train final model
    _, _, pred_test_scaled, _, _, _ = BreedSight(
        trainX=X_train_final,
        trainy=y_train_scaled,
        valX=None,
        valy=None,
        testX=X_test_final,
        testy=y_test_scaled if has_test_phenotypes else None,
        epochs=100,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=1
    )
    
    # Inverse transform test predictions
    pred_test_final = target_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).flatten() if pred_test_scaled is not None else None
    
    # Create final test predictions
    test_pred_final_df = pd.DataFrame({
        'Sample_ID': test_sample_ids,
        'Predicted_Phenotype': pred_test_final,
        'Model': 'Final'
    })
    if has_test_phenotypes:
        test_pred_final_df['True_Phenotype'] = phenotypic_test_info
    
    # Calculate final test metrics if available
    if has_test_phenotypes:
        mse_test_final, rmse_test_final, corr_test_final, r2_test_final = calculate_metrics(
            phenotypic_test_info, pred_test_final
        )
        results.append({
            'Fold': 'Final_Model',
            'Test_MSE': mse_test_final, 'Test_RMSE': rmse_test_final,
            'Test_R2': r2_test_final, 'Test_Corr': corr_test_final
        })
        
        print(f"\n=== Final Test Results ===")
        print(f"MSE: {mse_test_final:.4f}, RMSE: {rmse_test_final:.4f}")
        print(f"R²: {r2_test_final:.4f}, Correlation: {corr_test_final:.4f}")
    
    # Output Preparation
    train_pred_df = pd.concat(train_predictions, ignore_index=True)
    val_pred_df = pd.concat(val_predictions, ignore_index=True)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    train_pred_df.to_csv(train_pred_file, index=False)
    val_pred_df.to_csv(val_pred_file, index=False)
    test_pred_final_df.to_csv(test_pred_file, index=False)
    
    # Generate Plots
    def generate_plot(true_vals, pred_vals, title, is_test=False):
        plt.figure(figsize=(10, 6), dpi=300)
        plt.style.use('ggplot')
        
        if is_test and not has_test_phenotypes:
            pred_values = pred_vals.dropna()
            if len(pred_values) > 0:
                plt.hist(pred_values, bins=30, color='skyblue', edgecolor='black')
                plt.xlabel('Predicted Phenotype', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
            else:
                plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center', fontsize=12)
        else:
            valid_mask = (~pd.isna(pred_vals)) & (~pd.isna(true_vals))
            if valid_mask.any():
                plt.scatter(true_vals[valid_mask], pred_vals[valid_mask], alpha=0.5, color='blue')
                plt.xlabel('True Phenotype', fontsize=12)
                plt.ylabel('Predicted Phenotype', fontsize=12)
                coef = np.polyfit(true_vals[valid_mask], pred_vals[valid_mask], 1)
                poly1d_fn = np.poly1d(coef)
                plt.plot(true_vals[valid_mask], poly1d_fn(true_vals[valid_mask]), '--k', linewidth=2)
                plt.grid(True, linestyle='--', alpha=0.7)
            else:
                plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center', fontsize=12)
        
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plot_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        plt.savefig(plot_file, dpi=300)
        plt.close()
        return plot_file
    
    plot_files_train = [generate_plot(train_pred_df['True_Phenotype'],
                                    train_pred_df['Predicted_Phenotype'],
                                    'Training Set Predictions')]
    
    plot_files_val = [generate_plot(val_pred_df['True_Phenotype'],
                                  val_pred_df['Predicted_Phenotype'],
                                  'Validation Set Predictions')]
    
    plot_files_test = [generate_plot(test_pred_final_df.get('True_Phenotype', None),
                                   test_pred_final_df['Predicted_Phenotype'],
                                   'Test Set Predictions (Final Model)',
                                   is_test=True)]
    
    return results_df, train_pred_df, val_pred_df, test_pred_final_df, plot_files_train, plot_files_val, plot_files_test

def run_cross_validation(training_file, training_additive_file, testing_file, testing_additive_file,
                        feature_selection=True, learning_rate=0.0001, **kwargs):
    """
    Run cross-validation with the fixed model that prevents data leakage
    """
    # Load data
    training_data = pd.read_csv(training_file)
    training_additive = pd.read_csv(training_additive_file)
    testing_data = pd.read_csv(testing_file)
    testing_additive = pd.read_csv(testing_additive_file)
    
    # Check required columns
    if 'phenotypes' not in training_data.columns:
        raise ValueError("Training data must contain 'phenotypes' column")
    
    # Run cross-validation
    results, train_pred, val_pred, test_pred, train_plot, val_plot, test_plot = KFoldCrossValidation(
        training_data=training_data,
        training_additive=training_additive,
        testing_data=testing_data,
        testing_additive=testing_additive,
        epochs=1000,
        batch_size=64,
        learning_rate=learning_rate,
        feature_selection=feature_selection,
        outer_n_splits=5
    )
    
    # Prepare files for download
    def save_to_temp(df, prefix):
        path = os.path.join(tempfile.gettempdir(), f"{prefix}_predictions.csv")
        df.to_csv(path, index=False)
        return path
    
    train_csv = save_to_temp(train_pred, "train")
    val_csv = save_to_temp(val_pred, "val")
    test_csv = save_to_temp(test_pred, "test")
    
    return (
        train_pred,
        val_pred,
        test_pred,
        train_plot[0] if train_plot else None,
        val_plot[0] if val_plot else None,
        test_plot[0] if test_plot else None,
        train_csv,
        val_csv,
        test_csv
    )

