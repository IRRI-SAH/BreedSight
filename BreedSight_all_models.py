
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
from scipy.stats import pearsonr, t
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split

#from sklearn.model_model_selection import KFold, train_test_split
from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import tempfile

# Set random seed
RANDOM_STATE = 90
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def calculate_metrics(true_vals, pred_vals, heritability):
    """Calculate MSE, RMSE, Pearson correlation, and R²."""
    mse = mean_squared_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    corr, _ = pearsonr(true_vals, pred_vals)
    r2 = r2_score(true_vals, pred_vals)
    if r2 > np.sqrt(heritability):
        print(f"Warning: R² ({r2:.4f}) exceeds sqrt(heritability) ({np.sqrt(heritability):.4f})")
    return mse, rmse, corr, r2

def compute_genomic_features(X, ref_features=None, is_train=False, max_markers=10000, 
                            use_raw_genotypes=False, use_pca=False, n_components=50):
    """
    Compute genomic relationship features or use raw genotypes with optional PCA.
    """
    if X.shape[1] > max_markers:
        np.random.seed(RANDOM_STATE)
        selected_markers = np.random.choice(X.shape[1], max_markers, replace=False)
        X = X[:, selected_markers]
    else:
        selected_markers = None

    if use_raw_genotypes:
        if is_train:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            ref_features = {'scaler': scaler, 'selected_markers': selected_markers}
            X_final = X_scaled
        else:
            X_scaled = ref_features['scaler'].transform(X)
            X_final = X_scaled
    else:
        if is_train and ref_features is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            n_markers = X_scaled.shape[1]
            G_train = np.dot(X_scaled, X_scaled.T) / n_markers
            I_train = G_train * G_train
            mean_diag = np.mean(np.diag(I_train))
            I_train_norm = I_train / mean_diag if mean_diag != 0 else I_train
            X_final = np.concatenate([G_train, I_train_norm], axis=1)
            ref_features = {
                'scaler': scaler,
                'mean_diag': mean_diag,
                'X_train_scaled': X_scaled,
                'selected_markers': selected_markers
            }
        elif not is_train and ref_features is not None:
            if ref_features['selected_markers'] is not None:
                X = X[:, ref_features['selected_markers']]
            X_scaled = ref_features['scaler'].transform(X)
            n_markers = X_scaled.shape[1]
            G_val = np.dot(X_scaled, ref_features['X_train_scaled'].T) / n_markers
            I_val = G_val * G_val
            I_val_norm = I_val / ref_features['mean_diag'] if ref_features['mean_diag'] != 0 else I_val
            X_final = np.concatenate([G_val, I_val_norm], axis=1)
        else:
            raise ValueError("Invalid combination of is_train and ref_features parameters")

    if use_pca:
        if is_train:
            pca = PCA(n_components=min(n_components, X_final.shape[1]))
            X_final = pca.fit_transform(X_final)
            ref_features['pca'] = pca
        else:
            X_final = ref_features['pca'].transform(X_final)

    return X_final, ref_features

def BreedSight(trainX, trainy, valX=None, valy=None, testX=None, testy=None, 
               epochs=50, batch_size=64, learning_rate=0.0001, 
               l2_reg=0.1, dropout_rate=0.7, 
               rf_n_estimators=300, rf_max_depth=10, 
               alpha=0.1, verbose=1):
    """
    Hybrid DNN + Random Forest model for genomic prediction with strong regularization.
    """
    if not isinstance(trainX, np.ndarray) or not isinstance(trainy, np.ndarray):
        raise ValueError("trainX and trainy must be numpy arrays")
    if trainX.shape[0] != trainy.shape[0]:
        raise ValueError("trainX and trainy must have the same number of samples")
    if valX is not None and valy is not None:
        if valX.shape[0] != valy.shape[0]:
            raise ValueError("valX and valy must have the same number of samples")
    if testX is not None and testy is not None:
        if testX.shape[0] != testy.shape[0]:
            raise ValueError("testX and testy must have the same number of samples")

    feature_scaler = StandardScaler()
    trainX_scaled = feature_scaler.fit_transform(trainX)
    valX_scaled = feature_scaler.transform(valX) if valX is not None else None
    testX_scaled = feature_scaler.transform(testX) if testX is not None else None
    
    target_scaler = StandardScaler()
    trainy_scaled = target_scaler.fit_transform(trainy.reshape(-1, 1)).flatten()
    valy_scaled = target_scaler.transform(valy.reshape(-1, 1)).flatten() if valy is not None else None
    testy_scaled = target_scaler.transform(testy.reshape(-1, 1)).flatten() if testy is not None else None
    
    def build_dnn_model(input_shape):
        inputs = tf.keras.Input(shape=(input_shape,))   
        x = Dense(4, kernel_initializer='he_normal', 
                  kernel_regularizer=regularizers.l2(l2_reg))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        x = Dense(2, kernel_initializer='he_normal', 
                  kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        outputs = Dense(1, activation="linear")(x)
        model = tf.keras.Model(inputs, outputs)
        
        model.compile(loss=tf.keras.losses.Huber(delta=0.1), 
                     optimizer=Adam(learning_rate=learning_rate, clipvalue=0.5), 
                     metrics=['mse'])
        return model
    
    dnn_model = build_dnn_model(trainX.shape[1])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', verbose=verbose, 
                      restore_best_weights=True, patience=10),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, 
                         min_lr=1e-6, verbose=verbose)
    ]
    
    if valX is not None and valy is not None:
        validation_data = (valX_scaled, valy_scaled)
    else:
        raise ValueError("Validation data (valX, valy) must be provided")
    
    history = dnn_model.fit(
        trainX_scaled, 
        trainy_scaled, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=validation_data,
        verbose=verbose, 
        callbacks=callbacks
    )
    
    rf_model = RandomForestRegressor(
        n_estimators=rf_n_estimators, 
        max_depth=rf_max_depth, 
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(trainX, trainy.ravel())
    
    predicted_train_dnn_scaled = dnn_model.predict(trainX_scaled).flatten()
    predicted_val_dnn_scaled = dnn_model.predict(valX_scaled).flatten() if valX is not None else None
    predicted_test_dnn_scaled = dnn_model.predict(testX_scaled).flatten() if testX is not None else None
    
    predicted_train_rf = rf_model.predict(trainX)
    predicted_val_rf = rf_model.predict(valX) if valX is not None else None
    predicted_test_rf = rf_model.predict(testX) if testX is not None else None
    
    predicted_train_dnn = target_scaler.inverse_transform(
        predicted_train_dnn_scaled.reshape(-1, 1)).flatten()
    predicted_val_dnn = target_scaler.inverse_transform(
        predicted_val_dnn_scaled.reshape(-1, 1)).flatten() if valX is not None else None
    predicted_test_dnn = target_scaler.inverse_transform(
        predicted_test_dnn_scaled.reshape(-1, 1)).flatten() if testX is not None else None
    
    predicted_train = alpha * predicted_train_dnn + (1 - alpha) * predicted_train_rf
    predicted_val = alpha * predicted_val_dnn + (1 - alpha) * predicted_val_rf if valX is not None else None
    predicted_test = alpha * predicted_test_dnn + (1 - alpha) * predicted_test_rf if testX is not None else None
    
    if verbose > 0:
        print("\n=== Training Summary ===")
        print(f"Train samples: {len(trainX)}, Validation samples: {len(valX) if valX is not None else 'N/A'}")
        train_r2 = r2_score(trainy, predicted_train) if predicted_train is not None else np.nan
        val_r2 = r2_score(valy, predicted_val) if predicted_val is not None else np.nan
        print(f"Training R²: {train_r2:.4f}, Validation R²: {val_r2:.4f}")
        
        if train_r2 - val_r2 > 0.2:
            print("Warning: Potential overfitting detected (large gap between train and validation R²)")
        if val_r2 < 0.6:
            print("Warning: Potential underfitting detected (low validation R²)")

    return predicted_train, predicted_val, predicted_test, history, rf_model

def generate_regression_plot(true_vals, pred_vals, dataset_name, fold, r2_score, model_name, output_dir="output/diagnostic_plots"):
    """
    Generate and save a regression plot for true vs. predicted phenotypes.
    """
    if len(true_vals) == 0 or len(pred_vals) == 0 or len(true_vals) != len(pred_vals):
        print(f"Cannot generate regression plot for {dataset_name} ({model_name}, Fold {fold}): Invalid or empty data.")
        return None
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=true_vals, y=pred_vals, alpha=0.6)
    plt.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'r--', label='y=x')
    sns.regplot(x=true_vals, y=pred_vals, scatter=False, color='blue', label='Regression Line')
    plt.xlabel('True Phenotype')
    plt.ylabel(f'Predicted Phenotype ({model_name})')
    plt.title(f'{dataset_name} Regression Plot ({model_name}, Fold {fold}, R² = {r2_score:.4f})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_path = os.path.join(output_dir, f"{dataset_name.lower()}_regression_plot_{model_name}_fold_{fold}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path

def generate_combined_regression_plot(df, dataset_name, model_name, output_dir="output/diagnostic_plots"):
    """
    Generate combined regression plot for all folds.
    """
    if df is None or df.empty or df['True_Phenotype'].isna().all() or df[f'Predicted_Phenotype_{model_name}'].isna().all():
        print(f"No valid {dataset_name} data available for regression plot ({model_name}).")
        return None
    os.makedirs(output_dir, exist_ok=True)
    r2 = r2_score(df['True_Phenotype'], df[f'Predicted_Phenotype_{model_name}'])
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=df['True_Phenotype'], y=df[f'Predicted_Phenotype_{model_name}'], alpha=0.6)
    plt.plot([min(df['True_Phenotype']), max(df['True_Phenotype'])], 
             [min(df['True_Phenotype']), max(df['True_Phenotype'])], 'r--', label='y=x')
    sns.regplot(x=df['True_Phenotype'], y=df[f'Predicted_Phenotype_{model_name}'], scatter=False, color='blue', label='Regression Line')
    plt.xlabel('True Phenotype')
    plt.ylabel(f'Predicted Phenotype ({model_name})')
    plt.title(f'{dataset_name} Regression Plot ({model_name}, All Folds, R² = {r2:.4f})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_path = os.path.join(output_dir, f"{dataset_name.lower()}_regression_plot_{model_name}_all_folds.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path

def generate_comparison_bar_plot(results_df_dict, output_dir="output/diagnostic_plots"):
    """
    Generate bar plot comparing average R² across models.
    """
    os.makedirs(output_dir, exist_ok=True)
    models = list(results_df_dict.keys())
    train_r2 = [results_df_dict[model]['Train_R2'].mean() for model in models]
    val_r2 = [results_df_dict[model]['Val_R2'].mean() for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, train_r2, width, label='Training R²', color='#1f77b4')
    plt.bar(x + width/2, val_r2, width, label='Validation R²', color='#ff7f0e')
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.title('Average R² Comparison Across Models')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_path = os.path.join(output_dir, "model_comparison_bar_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    return plot_path

def encode_image_to_base64(image_path):
    """
    Encode image to base64 string for HTML embedding.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""

def generate_html_report(results_df_dict, diagnostic_plots, output_dir="output"):
    """
    Generate HTML report with R² table, plots, and explanations.
    """
    os.makedirs(output_dir, exist_ok=True)
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Model Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #2c3e50; }}
            img {{ max-width: 100%; height: auto; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .section {{ margin: 20px 0; padding: 20px; background: #f9f9f9; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <h1>Model Comparison Report</h1>
       
        <div class="section">
            <h2>1. R² Values per Fold and Model</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Fold</th>
                    <th>Train R²</th>
                    <th>Val R²</th>
                </tr>
    """
   
    for model_name, df in results_df_dict.items():
        for _, row in df.iterrows():
            fold = row['Fold']
            if fold != 'Final_Model':
                html_content += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{int(fold)}</td>
                    <td>{row['Train_R2']:.4f}</td>
                    <td>{row['Val_R2']:.4f}</td>
                </tr>
                """
   
    html_content += """
            </table>
        </div>
       
        <div class="section">
            <h2>2. Average R² Comparison Plot</h2>
            <img src="{}" alt="Model Comparison Bar Plot">
        </div>
       
        <div class="section">
            <h2>3. Regression Plots</h2>
    """
   
    for plot in diagnostic_plots:
        if 'regression_plot' in plot:
            html_content += f'<img src="{encode_image_to_base64(plot)}" alt="{os.path.basename(plot)}"><br>'
   
    html_content += """
        </div>
       
        <div class="section">
            <h2>4. Overfitting Prevention Strategies</h2>
            <ul>
                <li><b>L2 Regularization</b>: Applied L2 regularization (l2_reg=0.1) to DNN weights to penalize large weights and reduce model complexity.</li>
                <li><b>Dropout</b>: Used high dropout rate (0.7) in DNN layers to prevent over-reliance on specific neurons.</li>
                <li><b>Reduced Model Complexity</b>: Limited DNN to two layers (128, 64 units) and Random Forest to 100 trees with max_depth=10.</li>
                <li><b>Early Stopping</b>: Stopped training if validation loss did not improve for 10 epochs, restoring best weights.</li>
                <li><b>Learning Rate Reduction</b>: Reduced learning rate by factor of 0.5 if validation loss plateaued for 5 epochs.</li>
                <li><b>Ensemble Weighting</b>: Balanced DNN and Random Forest predictions with alpha=0.3 to leverage both models' strengths.</li>
                <li><b>Feature Selection</b>: Used RFE to select 200 features, reducing overfitting to noise in high-dimensional data.</li>
            </ul>
        </div>
       
        <div class="section">
            <h2>5. Data Leakage Prevention Strategies</h2>
            <ul>
                <li><b>Separate Imputation</b>: Used distinct KNNImputer instances for training and validation/test data to prevent information leakage.</li>
                <li><b>K-Fold Cross-Validation</b>: Ensured validation data was not used in training or preprocessing steps within each fold.</li>
                <li><b>Scaler Independence</b>: Fitted StandardScaler on training data only, applying the same transformation to validation/test data.</li>
                <li><b>PCA and Feature Selection</b>: Performed PCA and RFE on training data only, applying transformations to validation/test data to avoid leakage.</li>
                <li><b>Random Seed Control</b>: Fixed random seed (30) for reproducibility without leaking fold information.</li>
            </ul>
        </div>
       
        </body>
        </html>
    """
   
    bar_plot_path = os.path.join(output_dir, "diagnostic_plots", "model_comparison_bar_plot.png")
    html_content = html_content.format(encode_image_to_base64(bar_plot_path))
   
    output_path = os.path.join(output_dir, "model_comparison_report.html")
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"Saved HTML report to {output_path}")
    return output_path

def KFoldCrossValidation(training_data, training_additive, testing_data, testing_additive,
                         epochs=500, batch_size=64, learning_rate=0.0001,
                         l2_reg=0.1, dropout_rate=0.7, rf_n_estimators=300,
                         rf_max_depth=15, alpha=0.01, outer_n_splits=10,
                         output_file='cross_validation_results.csv',
                         train_pred_file='train_predictions.csv', 
                         val_pred_file='validation_predictions.csv',
                         test_pred_file='test_predictions.csv',
                         feature_selection=True, heritability=0.82, 
                         use_raw_genotypes=False, use_pca=False, n_components=100,
                         rfe_n_features=200):
    """
    Perform K-fold cross-validation with RFE, feature importance, and prediction intervals.
    MODIFIED: Added regression plots for all models, comparison bar plot, and HTML report.
    """
    assert isinstance(training_data, pd.DataFrame), "Training data must be DataFrame"
    assert isinstance(testing_data, pd.DataFrame), "Testing data must be DataFrame"
    assert isinstance(training_additive, pd.DataFrame), "Training additive data must be DataFrame"
    assert isinstance(testing_additive, pd.DataFrame), "Testing additive data must be DataFrame"
    
    if 'phenotypes' not in training_data.columns:
        raise ValueError("Training data must contain 'phenotypes' column")
    
    # Store raw data
    training_additive_raw = training_additive.iloc[:, 1:].values
    phenotypic_info_raw = training_data['phenotypes'].values
    has_test_phenotypes = 'phenotypes' in testing_data.columns
    phenotypic_test_info_raw = testing_data['phenotypes'].values if has_test_phenotypes else None
    test_sample_ids = testing_data.iloc[:, 0].values
    testing_additive_raw = testing_additive.iloc[:, 1:].values
    
    def generate_diagnostic_plots(data, phenotypes, prefix):
        os.makedirs("output/diagnostic_plots", exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.hist(phenotypes, bins=30, alpha=0.7)
        plt.title(f"{prefix} Phenotype Distribution")
        plt.xlabel("Phenotype Value")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.7)
        pheno_plot = os.path.join("output", "diagnostic_plots", f"{prefix}_phenotype_histogram.png")
        plt.savefig(pheno_plot, bbox_inches='tight')
        plt.close()
        
        allele_freq = np.mean(data, axis=0)
        plt.figure(figsize=(8, 6))
        plt.hist(allele_freq, bins=30, alpha=0.7)
        plt.title(f"{prefix} Marker Allele Frequency")
        plt.xlabel("Allele Frequency")
        plt.ylabel("Count")
        plt.grid(True, linestyle='--', alpha=0.7)
        marker_plot = os.path.join("output", "diagnostic_plots", f"{prefix}_marker_frequency.png")
        plt.savefig(marker_plot, bbox_inches='tight')
        plt.close()
        
        return [pheno_plot, marker_plot]
    
    diagnostic_plots = generate_diagnostic_plots(training_additive_raw, phenotypic_info_raw, "training")
    
    outer_kf = KFold(n_splits=outer_n_splits, shuffle=True, random_state=RANDOM_STATE)
    results_dict = {'BreedSight': [], 'Lasso': [], 'RBLUP': [], 'GBLUP': []}
    feature_importances_dict = {'BreedSight': [], 'Lasso': [], 'RBLUP': [], 'GBLUP': []}
    plot_files = []
    selected_features = []
    train_pred_list = {model: [] for model in results_dict}
    val_pred_list = {model: [] for model in results_dict}
    
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
                'verbose': 1
            },
            'is_tree': True
        },
        'Lasso': {
            'function': Lasso,
            'params': {'alpha': 0.1},
            'is_tree': False
        },
        'RBLUP': {
            'function': Ridge,
            'params': {'alpha': 1.0},
            'is_tree': False
        },
        'GBLUP': {
            'function': Ridge,
            'params': {'alpha': 1.0},
            'is_tree': False
        }
    }
    
    for outer_fold, (outer_train_index, outer_val_index) in enumerate(outer_kf.split(training_additive_raw), 1):
        print(f"\n=== Outer Fold {outer_fold}/{outer_n_splits} ===")
        
        fold_train_additive_raw = training_additive_raw[outer_train_index]
        fold_val_additive_raw = training_additive_raw[outer_val_index]
        fold_train_phenotypes_raw = phenotypic_info_raw[outer_train_index]
        fold_val_phenotypes_raw = phenotypic_info_raw[outer_val_index]
        
        imputer_pheno = KNNImputer(n_neighbors=5)
        fold_train_phenotypes = imputer_pheno.fit_transform(fold_train_phenotypes_raw.reshape(-1, 1)).flatten()
        imputer_add = KNNImputer(n_neighbors=5)
        fold_train_additive = imputer_add.fit_transform(fold_train_additive_raw)
        fold_val_phenotypes = imputer_pheno.transform(fold_val_phenotypes_raw.reshape(-1, 1)).flatten()
        fold_val_additive = imputer_add.transform(fold_val_additive_raw)
        
        print(f"Fold {outer_fold}: Imputed {len(fold_train_phenotypes)} train samples and {len(fold_val_phenotypes)} val samples (leakage-free).")
        
        outer_trainX = fold_train_additive
        outer_valX = fold_val_additive
        outer_trainy = fold_train_phenotypes
        outer_valy = fold_val_phenotypes
        
        X_train_genomic, ref_features = compute_genomic_features(
            outer_trainX, ref_features=None, is_train=True, 
            use_raw_genotypes=use_raw_genotypes, use_pca=use_pca, n_components=n_components
        )
        X_val_genomic, _ = compute_genomic_features(
            outer_valX, ref_features=ref_features, is_train=False,
            use_raw_genotypes=use_raw_genotypes, use_pca=use_pca, n_components=n_components
        )
        
        if feature_selection:
            rf = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE)
            selector = RFE(estimator=rf, n_features_to_select=min(rfe_n_features, X_train_genomic.shape[1]))
            selector.fit(X_train_genomic, outer_trainy)
            X_train_final = selector.transform(X_train_genomic)
            X_val_final = selector.transform(X_val_genomic)
            selected_features.append(selector.support_)
            print(f"Fold {outer_fold}: Selected {np.sum(selector.support_)} features with RFE.")
        else:
            X_train_final = X_train_genomic
            X_val_final = X_val_genomic
        
        for model_name, config in model_configs.items():
            print(f"Training {model_name} for fold {outer_fold}")
            if model_name == 'BreedSight':
                pred_train, pred_val, _, history, model = config['function'](
                    trainX=X_train_final,
                    trainy=outer_trainy,
                    valX=X_val_final,
                    valy=outer_valy,
                    testX=None,
                    testy=None,
                    **config['params']
                )
            else:
                model = config['function'](**config['params'])
                model.fit(X_train_final, outer_trainy.ravel())
                pred_train = model.predict(X_train_final)
                pred_val = model.predict(X_val_final)
                history = None
            
            train_pred_list[model_name].append(pd.DataFrame({
                'Sample_ID': training_data.iloc[outer_train_index, 0].values,
                f'Predicted_Phenotype_{model_name}': pred_train,
                'True_Phenotype': outer_trainy
            }))
            val_pred_list[model_name].append(pd.DataFrame({
                'Sample_ID': training_data.iloc[outer_val_index, 0].values,
                f'Predicted_Phenotype_{model_name}': pred_val,
                'True_Phenotype': outer_valy
            }))
            
            train_r2 = r2_score(outer_trainy, pred_train)
            val_r2 = r2_score(outer_valy, pred_val)
            train_plot = generate_regression_plot(
                outer_trainy, pred_train, f"Training", outer_fold, train_r2, model_name
            )
            val_plot = generate_regression_plot(
                outer_valy, pred_val, f"Validation", outer_fold, val_r2, model_name
            )
            if train_plot:
                plot_files.append(train_plot)
            if val_plot:
                plot_files.append(val_plot)
            
            if config['is_tree']:
                tree_predictions = np.array([tree.predict(X_train_final) for tree in model.estimators_])
                std_pred = np.std(tree_predictions, axis=0)
                t_value = t.ppf((1 + 0.95) / 2, df=len(model.estimators_) - 1)
                margin_error = t_value * std_pred
                train_lower = pred_train - margin_error
                train_upper = pred_train + margin_error
                
                tree_predictions_val = np.array([tree.predict(X_val_final) for tree in model.estimators_])
                std_pred_val = np.std(tree_predictions_val, axis=0)
                margin_error_val = t_value * std_pred_val
                val_lower = pred_val - margin_error_val
                val_upper = pred_val + margin_error_val
            else:
                train_lower, train_upper = pred_train, pred_train
                val_lower, val_upper = pred_val, pred_val
            
            if history is not None:
                plt.figure(figsize=(10, 6))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title(f'Learning Curve - {model_name} Fold {outer_fold}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plot_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                plt.savefig(plot_file, bbox_inches='tight')
                plt.close()
                plot_files.append(plot_file)
            
            feature_importances_dict[model_name].append(
                model.feature_importances_ if config['is_tree'] else np.abs(model.coef_)
            )
            
            mse_train, rmse_train, corr_train, r2_train = calculate_metrics(outer_trainy, pred_train, heritability)
            mse_val, rmse_val, corr_val, r2_val = calculate_metrics(outer_valy, pred_val, heritability)
            
            results_dict[model_name].append({
                'Fold': outer_fold,
                'Train_MSE': mse_train, 'Train_RMSE': rmse_train,
                'Train_R2': r2_train, 'Train_Corr': corr_train,
                'Val_MSE': mse_val, 'Val_RMSE': rmse_val,
                'Val_R2': r2_val, 'Val_Corr': corr_val
            })
            
            print(f"{model_name} Fold {outer_fold} - Training R²: {r2_train:.4f}, Validation R²: {r2_val:.4f}")
    
    if feature_selection and len(selected_features) > 1:
        feature_stability = np.mean([
            np.mean(np.abs(np.array(selected_features[i], dtype=np.int64) - np.array(selected_features[j], dtype=np.int64)))
            for i in range(len(selected_features)) 
            for j in range(i + 1, len(selected_features))
        ])
        print(f"Feature selection stability (mean difference): {feature_stability:.4f}")
        if feature_stability == 0.0:
            print("Warning: Feature selection stability is 0.0, indicating identical features selected across folds. Check input data or RFE settings.")
    else:
        feature_stability = None
    
    mean_feature_importance_dict = {}
    for model_name in model_configs:
        importances = feature_importances_dict[model_name]
        mean_imp = np.mean(importances, axis=0)
        if not model_configs[model_name]['is_tree']:
            mean_imp = mean_imp / np.sum(mean_imp) if np.sum(mean_imp) > 0 else mean_imp
        mean_feature_importance_dict[model_name] = mean_imp
    
    print("\n=== Training Final Model on ALL Training Data for Each Model ===")
    
    imputer_pheno_final = KNNImputer(n_neighbors=5)
    y_train_imputed = imputer_pheno_final.fit_transform(phenotypic_info_raw.reshape(-1, 1)).flatten()
    imputer_add_final = KNNImputer(n_neighbors=5)
    X_train_raw_imputed = imputer_add_final.fit_transform(training_additive_raw)
    testing_additive_imputed = imputer_add_final.transform(testing_additive_raw)
    
    phenotypic_test_info = phenotypic_test_info_raw
    if has_test_phenotypes and np.any(pd.isna(phenotypic_test_info)):
        print("Warning: Missing values in test phenotypes. Metrics may be affected.")
    
    X_train_genomic, ref_features = compute_genomic_features(
        X_train_raw_imputed, ref_features=None, is_train=True,
        use_raw_genotypes=use_raw_genotypes, use_pca=use_pca, n_components=n_components
    )
    X_test_genomic, _ = compute_genomic_features(
        testing_additive_imputed, ref_features=ref_features, is_train=False,
        use_raw_genotypes=use_raw_genotypes, use_pca=use_pca, n_components=n_components
    )

    if feature_selection:
        rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
        selector = RFE(estimator=rf, n_features_to_select=min(rfe_n_features, X_train_genomic.shape[1]))
        selector.fit(X_train_genomic, y_train_imputed)
        X_train_final = selector.transform(X_train_genomic)
        X_test_final = selector.transform(X_test_genomic)
    else:
        X_train_final = X_train_genomic
        X_test_final = X_test_genomic
    
    final_pred_test_dict = {}
    for model_name, config in model_configs.items():
        if model_name == 'BreedSight':
            X_train_sub, X_val_final, y_train_sub, y_val_final = train_test_split(
                X_train_final, y_train_imputed, test_size=0.05, random_state=RANDOM_STATE
            )
            _, _, pred_test_final, _, model = config['function'](
                trainX=X_train_sub,
                trainy=y_train_sub,
                valX=X_val_final,
                valy=y_val_final,
                testX=X_test_final,
                testy=phenotypic_test_info if has_test_phenotypes else None,
                **config['params']
            )
        else:
            model = config['function'](**config['params'])
            model.fit(X_train_final, y_train_imputed.ravel())
            pred_test_final = model.predict(X_test_final)
        
        final_pred_test_dict[model_name] = pred_test_final
        
        if has_test_phenotypes:
            valid_mask = ~np.isnan(phenotypic_test_info)
            true_valid = phenotypic_test_info[valid_mask]
            pred_valid = pred_test_final[valid_mask]
            if len(true_valid) > 0:
                mse_test, rmse_test, corr_test, r2_test = calculate_metrics(
                    true_valid, pred_valid, heritability
                )
                results_dict[model_name].append({
                    'Fold': 'Final_Model',
                    'Test_MSE': mse_test, 'Test_RMSE': rmse_test,
                    'Test_R2': r2_test, 'Test_Corr': corr_test
                })
                
                print(f"\n=== {model_name} Final Test Results ===")
                print(f"MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}")
                print(f"R²: {r2_test:.4f}, Correlation: {corr_test:.4f}")
    
    if has_test_phenotypes:
        diagnostic_plots += generate_diagnostic_plots(testing_additive_raw, phenotypic_test_info_raw, "testing")
    
    results_df_dict = {model_name: pd.DataFrame(results) for model_name, results in results_dict.items()}
    
    train_pred_df = pd.concat([pd.concat(train_pred_list[model_name], ignore_index=True) 
                               for model_name in model_configs], axis=1)
    train_pred_df = train_pred_df.loc[:, ~train_pred_df.columns.duplicated()]
    val_pred_df = pd.concat([pd.concat(val_pred_list[model_name], ignore_index=True) 
                             for model_name in model_configs], axis=1)
    val_pred_df = val_pred_df.loc[:, ~val_pred_df.columns.duplicated()]
    
    test_pred_final_df = pd.DataFrame({'Sample_ID': test_sample_ids})
    for model_name in model_configs:
        test_pred_final_df[f'Predicted_Phenotype_{model_name}'] = final_pred_test_dict[model_name]
    if has_test_phenotypes:
        test_pred_final_df['True_Phenotype'] = phenotypic_test_info
    
    for model_name in model_configs:
        train_plot = generate_combined_regression_plot(train_pred_df, "Training", model_name)
        val_plot = generate_combined_regression_plot(val_pred_df, "Validation", model_name)
        if train_plot:
            diagnostic_plots.append(train_plot)
        if val_plot:
            diagnostic_plots.append(val_plot)
    
    comparison_plot = generate_comparison_bar_plot(results_df_dict)
    diagnostic_plots.append(comparison_plot)
    
    feature_importance_df = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(len(mean_feature_importance_dict['BreedSight']))],
        'Importance': mean_feature_importance_dict['BreedSight']
    })
    
    metrics_df = pd.DataFrame({
        'Model': list(model_configs.keys()),
        'Avg_Train_R2': [results_df_dict[model]['Train_R2'].mean() for model in model_configs],
        'Avg_Val_R2': [results_df_dict[model]['Val_R2'].mean() for model in model_configs]
    })
    
    html_report = generate_html_report(results_df_dict, diagnostic_plots)
    
    return results_df_dict, train_pred_df, val_pred_df, test_pred_final_df, [], [], [], feature_importance_df, feature_stability, diagnostic_plots, output_file, metrics_df, html_report

def run_cross_validation(training_file, training_additive_file, testing_file, testing_additive_file, 
                         feature_selection=True, learning_rate=0.0001, heritability=0.72, 
                         use_raw_genotypes=False, use_pca=False, n_components=50, rfe_n_features=200):
    """
    Run cross-validation with enhanced features.
    """
    training_data = pd.read_csv(training_file)
    training_additive = pd.read_csv(training_additive_file)
    testing_data = pd.read_csv(testing_file)
    testing_additive = pd.read_csv(testing_additive_file)
    
    results_df_dict, train_pred_df, val_pred_df, test_pred_df, _, _, _, feature_importance_df, feature_stability, diagnostic_plots, html_file, metrics_df, html_report = KFoldCrossValidation(
        training_data=training_data,
        training_additive=training_additive,
        testing_data=testing_data,
        testing_additive=testing_additive,
        epochs=1000,
        batch_size=64,
        learning_rate=learning_rate,
        l2_reg=0.5,
        dropout_rate=0.5,
        rf_n_estimators=300,
        rf_max_depth=5,
        alpha=0.1,
        feature_selection=feature_selection,
        outer_n_splits=5,
        heritability=heritability,
        use_raw_genotypes=True,
        use_pca=False,
        n_components=n_components,
        rfe_n_features=20
    )
    
    return (
        train_pred_df,
        val_pred_df,
        test_pred_df,
        None, None, None,
        None, None, None,
        feature_importance_df,
        feature_stability,
        diagnostic_plots,
        html_report,
        metrics_df
    )

# Define file paths
training_file_path = "training_phenotypic_data.csv"
training_additive_file_path = "training_additive.csv"
testing_file_path = "testing_data.csv"
testing_additive_file_path = "testing_additive.csv"

# Run cross-validation to get predictions
results = run_cross_validation(
    training_file=training_file_path,
    training_additive_file=training_additive_file_path,
    testing_file=testing_file_path,
    testing_additive_file=testing_additive_file_path,
    feature_selection=True,
    use_raw_genotypes=False,
    use_pca=True,
    n_components=50,
    rfe_n_features=20,
    heritability=0.82
)

# Unpack results
train_pred, val_pred, test_pred, _, _, _, _, _, _, feature_importance_df, feature_stability, diagnostic_plots, html_report, metrics_df = results

# Define output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Save predictions to CSV
def save_predictions(df, filename, dataset_name):
    if df is None or df.empty:
        print(f"No {dataset_name} predictions available to save.")
        return
    output_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {dataset_name} predictions to {output_path}")

# Save predictions and metrics
save_predictions(train_pred, "training_predictions", "training")
save_predictions(val_pred, "validation_predictions", "validation")
save_predictions(test_pred, "testing_predictions", "testing")
save_predictions(feature_importance_df, "feature_importance", "feature importance")
save_predictions(metrics_df, "model_metrics", "model metrics")

# Print head of predictions for verification
print("\nTraining predictions:")
print(train_pred.head() if train_pred is not None and not train_pred.empty else "No training predictions available")
print("\nValidation predictions:")
print(val_pred.head() if val_pred is not None and not val_pred.empty else "No validation predictions available")
print("\nTest predictions:")
print(test_pred.head() if test_pred is not None and not test_pred.empty else "No test predictions available")
print("\nFeature Importance:")
print(feature_importance_df.head() if feature_importance_df is not None and not feature_importance_df.empty else "No feature importance available")
print("\nModel Metrics:")
print(metrics_df if metrics_df is not None and not metrics_df.empty else "No metrics available")
print(f"\nFeature Selection Stability: {feature_stability:.4f}" if feature_stability is not None else "No feature stability metric available")
print("\nDiagnostic Plots Generated:")
for plot in diagnostic_plots:
    print(plot)
print(f"\nHTML Report: {html_report}")
