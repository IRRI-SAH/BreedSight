# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 18:52:23 2025

@author: Ashmitha
"""

#############################Required Package####################################
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import tempfile
import os
from sklearn.model_selection import train_test_split
###################Seed default#################
RANDOM_STATE = 60 
##########################Model Initialization#############
# User can change the number of layers and neurons accordingly 

def BreedSight(trainX, trainy, valX=None, valy=None, testX=None, testy=None, 
            epochs=1000, batch_size=64, learning_rate=0.0001, 
            l2_reg=0.001, dropout_rate=0.5, 
            rf_n_estimators=200, rf_max_depth=30, 
            alpha=0.5, verbose=1):
    
    # Initialize results
    predicted_test = None
    
    # -------------------------------- Feature Scaling
    # Only fit on training data, transform others
    feature_scaler = StandardScaler()
    trainX_scaled = feature_scaler.fit_transform(trainX)
    valX_scaled = feature_scaler.transform(valX) if valX is not None else None
    testX_scaled = feature_scaler.transform(testX) if testX is not None else None
    
    # -------------------------------- Target Scaling
    # Only fit on training data, transform others
    target_scaler = StandardScaler()
    trainy_scaled = target_scaler.fit_transform(trainy.reshape(-1, 1)).flatten()
    valy_scaled = target_scaler.transform(valy.reshape(-1, 1)).flatten() if valy is not None else None
    testy_scaled = target_scaler.transform(testy.reshape(-1, 1)).flatten() if testy is not None else None
    
    # -------------------------------- Build DNN Model
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
        
        outputs = Dense(1, activation="relu")(x) #change to linear if required#
        model = tf.keras.Model(inputs, outputs)
        
        model.compile(loss=tf.keras.losses.Huber(delta=0.1), 
                     optimizer=Adam(learning_rate=learning_rate, clipvalue=0.1), 
                     metrics=['mse'])
        return model
    
    dnn_model = build_dnn_model(trainX.shape[1])
    
    # -------------------------------- Train DNN Model
    ""' un comment this if the model is overfitting ""'
    #callbacks = [
       # EarlyStopping(monitor='val_loss', verbose=verbose, 
                    # restore_best_weights=True, patience=20)
   # ]
    
    if valX is not None and valy is not None:
        validation_data = (valX_scaled, valy_scaled)
        validation_split = 0.0
    else:
        validation_data = None
        validation_split = 0.05
    
    history = dnn_model.fit(
        trainX_scaled, 
        trainy_scaled, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=validation_data,
        validation_split=validation_split,
        verbose=verbose, 
        #callbacks=callbacks
    )
    
    # -------------------------------- Train Random Forest Model
    rf_model = RandomForestRegressor(
        n_estimators=rf_n_estimators, 
        max_depth=rf_max_depth, 
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(trainX, trainy.ravel())
    
    # -------------------------------- Make Predictions
    predicted_train_dnn_scaled = dnn_model.predict(trainX_scaled).flatten()
    predicted_val_dnn_scaled = dnn_model.predict(valX_scaled).flatten() if valX is not None else None
    predicted_test_dnn_scaled = dnn_model.predict(testX_scaled).flatten() if testX is not None else None
    
    predicted_train_rf = rf_model.predict(trainX)
    predicted_val_rf = rf_model.predict(valX) if valX is not None else None
    predicted_test_rf = rf_model.predict(testX) if testX is not None else None
    
    # Inverse transform to get predictions back to original scale
    predicted_train_dnn = target_scaler.inverse_transform(
        predicted_train_dnn_scaled.reshape(-1, 1)).flatten()
    predicted_val_dnn = target_scaler.inverse_transform(
        predicted_val_dnn_scaled.reshape(-1, 1)).flatten() if valX is not None else None
    predicted_test_dnn = target_scaler.inverse_transform(
        predicted_test_dnn_scaled.reshape(-1, 1)).flatten() if testX is not None else None
    
    # -------------------------------- Ensemble Predictions
    predicted_train = alpha * predicted_train_dnn + (1 - alpha) * predicted_train_rf
    predicted_val = alpha * predicted_val_dnn + (1 - alpha) * predicted_val_rf if valX is not None else None
    predicted_test = alpha * predicted_test_dnn + (1 - alpha) * predicted_test_rf if testX is not None else None
    
    if verbose > 0:
        print("\n=== Training Summary ===")
        print(f"Train samples: {len(trainX)}, Validation samples: {len(valX) if valX is not None else 'N/A'}")
        
    return predicted_train, predicted_val, predicted_test, history, rf_model

def compute_genomic_features(X, ref_features=None, is_train=False):
    """
    Compute genomic relationship features without data leakage
    
    Parameters:
    -----------
    X: Input genomic data
    ref_features: Reference features from training data (None for training data)
    is_train: Boolean indicating if this is training data
    
    Returns:
    --------
    X_final: Transformed features
    ref_features: Dictionary of reference statistics (if is_train=True)
    """
    if is_train and ref_features is None:
        # For training data when no reference provided
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_markers = X_scaled.shape[1]
        
        # Compute relationship matrices for training data only
        G_train = np.dot(X_scaled, X_scaled.T) / n_markers
        I_train = G_train * G_train
        
        # Store reference statistics
        mean_diag = np.mean(np.diag(I_train))
        I_train_norm = I_train / mean_diag
        
        # Combine features
        X_final = np.concatenate([G_train, I_train_norm], axis=1)
        
        # Store reference info for validation/test
        ref_features = {
            'scaler': scaler,
            'mean_diag': mean_diag,
            'X_train_scaled': X_scaled
        }
        
    elif not is_train and ref_features is not None:
        # For validation/test data with reference features
        X_scaled = ref_features['scaler'].transform(X)
        n_markers = X_scaled.shape[1]
        
        # Compute relationship with training samples only
        G_val = np.dot(X_scaled, ref_features['X_train_scaled'].T) / n_markers
        I_val = G_val * G_val
        
        # Normalize using training statistics
        I_val_norm = I_val / ref_features['mean_diag']
        
        # Construct features
        X_final = np.concatenate([G_val, I_val_norm], axis=1)
    
    else:
        raise ValueError("Invalid combination of is_train and ref_features parameters")
    
    return X_final, ref_features

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

def KFoldCrossValidation(training_data, training_additive, testing_data, testing_additive,
                        epochs=1000, learning_rate=0.001, batch_size=64,
                        outer_n_splits=10, output_file='cross_validation_results.csv',
                        train_pred_file='train_predictions.csv', 
                        val_pred_file='validation_predictions.csv',
                        test_pred_file='test_predictions.csv',
                        feature_selection=True):
    
    # -------------------------------- Data Validation
    assert isinstance(training_data, pd.DataFrame), "Training data must be DataFrame"
    assert isinstance(testing_data, pd.DataFrame), "Testing data must be DataFrame"
    
    train_ids = set(training_data.iloc[:, 0].values)
    test_ids = set(testing_data.iloc[:, 0].values)
    assert len(train_ids & test_ids) == 0, "Training and testing sets must be distinct"
    
    # -------------------------------- Data Preparation
    training_additive_raw = training_additive.iloc[:, 1:].values
    testing_additive_raw = testing_additive.iloc[:, 1:].values
    phenotypic_info = training_data['phenotypes'].values
    
    has_test_phenotypes = 'phenotypes' in testing_data.columns
    phenotypic_test_info = testing_data['phenotypes'].values if has_test_phenotypes else None
    test_sample_ids = testing_data.iloc[:, 0].values
    
    # -------------------------------- Outer CV Loop
    outer_kf = KFold(n_splits=outer_n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = []
    train_predictions = []
    val_predictions = []

    for outer_fold, (outer_train_index, outer_val_index) in enumerate(outer_kf.split(training_additive_raw), 1):
        print(f"\n=== Outer Fold {outer_fold}/{outer_n_splits} ===")
        
        # Split data
        outer_trainX = training_additive_raw[outer_train_index]
        outer_valX = training_additive_raw[outer_val_index]
        outer_trainy = phenotypic_info[outer_train_index]
        outer_valy = phenotypic_info[outer_val_index]
        
        # Process features without leakage
        X_train_genomic, ref_features = compute_genomic_features(
            outer_trainX, 
            ref_features=None,
            is_train=True
        )
        
        X_val_genomic, _ = compute_genomic_features(
            outer_valX, 
            ref_features=ref_features,
            is_train=False
        )
        
        # Feature selection without leakage
        if feature_selection:
            selector = SelectFromModel(
                RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE), 
                threshold="mean"
            )
            selector.fit(X_train_genomic, outer_trainy)
            X_train_final = selector.transform(X_train_genomic)
            X_val_final = selector.transform(X_val_genomic)
        else:
            X_train_final = X_train_genomic
            X_val_final = X_val_genomic
            
        # Model training
        pred_train, pred_val, _, history, _ = BreedSight(
            trainX=X_train_final, 
            trainy=outer_trainy,
            valX=X_val_final,
            valy=outer_valy,
            testX=None,
            testy=None,
            epochs=1000, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            verbose=1
        )
        
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
    
    # -------------------------------- FINAL MODEL TRAINING
    print("\n==============================Training Final model on ALL training data")
    
    # Process ALL training data
    X_train_raw = training_additive_raw
    y_train_raw = phenotypic_info

    # Feature processing
    X_train_genomic, ref_features = compute_genomic_features(
        X_train_raw, 
        ref_features=None, 
        is_train=True
    )
    X_test_genomic, _ = compute_genomic_features(
        testing_additive_raw, 
        ref_features=ref_features, 
        is_train=False
    )

    # Feature selection
    if feature_selection:
        selector = SelectFromModel(
            RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE), 
            threshold="1.25*median"
        )
        selector.fit(X_train_genomic, y_train_raw)
        X_train_final = selector.transform(X_train_genomic)
        X_test_final = selector.transform(X_test_genomic)
    else:
        X_train_final = X_train_genomic
        X_test_final = X_test_genomic

    # Train final model
    _, _, pred_test_final, _, _ = BreedSight(
        trainX=X_train_final, 
        trainy=y_train_raw,
        valX=None,
        valy=None,
        testX=X_test_final,
        testy=phenotypic_test_info if has_test_phenotypes else None,
        epochs=100,  # Fixed epoch count
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=1
    )
    
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
    
    # -------------------------------- Output Preparation
    train_pred_df = pd.concat(train_predictions, ignore_index=True)
    val_pred_df = pd.concat(val_predictions, ignore_index=True)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    train_pred_df.to_csv(train_pred_file, index=False)
    val_pred_df.to_csv(val_pred_file, index=False)
    test_pred_final_df.to_csv(test_pred_file, index=False)
    
    # -------------------------------- Generate Plots
    def generate_plot(true_vals, pred_vals, title, is_test=False):
        plt.figure(figsize=(10, 6))
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
            else:
                plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
        
        plt.title(title)
        plot_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        plt.savefig(plot_file)
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
        epochs=100,  # Reasonable number for demonstration
        batch_size=64,
        learning_rate=learning_rate,
        feature_selection=feature_selection,
        outer_n_splits=10
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
        train_pred,  # train_output
        val_pred,    # val_output
        test_pred,   # test_output
        train_plot[0] if train_plot else None,
        val_plot[0] if val_plot else None,
        test_plot[0] if test_plot else None,
        train_csv,
        val_csv,
        test_csv
    )
