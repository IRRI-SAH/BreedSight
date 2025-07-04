<h1 align="center">
  <b>🧬 BreedSight</b>
</h1>

<h3 align="center">
  <i>An ensemble deep learning genomic prediction model</i><br>
  <small>for utilizing natural variation in rice breeding</small>
</h3>

---

## 📝 Description

BreedSight is an parallel ensemble based python package that combines deep learning with random forest to estimate Genomic Estimated Breeding Values (GEBVs) and recommend optimal crosses, enabling breeders to accelerate genetic gain and streamline selection decisions in rice improvement programs.

**Key Features:**
- Elite X Exotic
- Elite X Elite 
- Genetic gain and Genetic Diversity prediction

---

## 🧠 Model Information
The BreedSight framework  

BreedSight is an ensemble machine learning model that combines deep neural networks (DNNs) and random forests (RFs) using a weighted average approach for genomic prediction (Fig. 1). RFs leveraged special advantages for genomic data over other machine learning models. Unlike methods like XGBoost (which can overfit small datasets by repeatedly correcting errors), RF reduced overfitting in two ways 1) by bootstrap aggregation (bagging) and 2) by random feature selection. In bagging, RFs build each decision tree on a different random subset of the data (with replacement). This introduced variability among the trees, reducing the likelihood of the model overfitting noise in the training data and thereby enhancing its ability to generalize to unseen data. 

 

Random Feature Selection 

During the construction of each tree, RFs randomly selected a subset of features (e.g., genetic markers) to consider splitting at each node. This limited the influence of dominant predictors and encouraged diversity among trees, which further reduced the model’s variance and helped prevent overfitting. This makes RFs better at handling high-dimensional genomic data (where there are many features but few samples) by creating smoother decision boundaries and improving generalization (Herrerra V.M et al., 2019). The parallel ensemble model combining RF and DNN architectures leveraged RF's efficiency and robust feature selection alongwith DNNs' ability to capture complex nonlinear relationships while mitigating their individual weaknesses for genomic prediction tasks. 

 

 For model prediction, the input data were checked for feature consistency, missing values, and dimensional alignment with standardization applied to both features, Z-score normalization, and phenotypic targets. The DNN architecture employed configurable fully connected layers (default:256-128-64-32 units) with residual connections, batch normalization, LeakyReLU activation (α=0.1) and dropout (rate=0.5) for regularization. Training used the Huber Loss function, adam optimization (learning rate =0.0001) and early stopping (patience=3) with gradient magnitude monitored via custom callbacks. The RF components comprised 200 trees (max depth =42), and ensemble predictions combined DNN and RF outputs through the fixed weighted average (default α=0.6) to prevent NAN values. For Evaluation, implemented 10-fold cross-validation, computing additive (A = XXᵀ/*m*) and epistatic (I = A⊙A normalized by mean diagonal) relationship matrices as features. Performance assessed using  RMSE Pearson correlation and R² with validation split within folds and on the independent test set. Visualization includes prediction of scatter plots and distributions. 

 

Epistatic Interaction  

The model extended the standard additive framework by incorporating additive-by-additive epistatic effects, following the theoretical foundation established by Henderson (1985). This enhanced model formulation is expressed as 

y = 1μ + Zg + Zi + ε, 

where y represents the phenotypic observations, μ is the overall mean, Z is the design matrix, g denotes additive genetic effects [g ∼ N(0, σ²_G G)], i represents epistatic effects [i ∼ N(0, σ²_I I)], and ε is the residual error term. The epistatic relationship matrix I is constructed through the Hadamard product (element-wise multiplication) of the additive genomic relationship matrix G (I = A ⊙ A), following the standardization approach described by Vitezica et al. (2017). The inclusion of the epistatic term enables the model to capture non-additive genetic variance. The resulting model provided a computationally efficient yet biologically meaningful framework for estimating breeding values in traits influenced by epistatic interactions, while preserving the interpretability and robustness of the linear mixed model approach. 

 

Ten-fold-cross validation 

Cross-validation represents a standard methodology for assessing the predictive accuracy of genomic selection (GS) models (Estaghvirou et al 2013). In this study, we implemented a tenfold cross-validation scheme. For each iteration, nine subsets (90% of samples) served as the training set to develop the prediction model, while the remaining subset (10% of samples) functioned as the validation set. Following model training, phenotypic values for individuals in the test group were predicted exclusively from their genotypic data. 

 

Hyperparameter Optimization  

BreedSight integrates DNNs with an RF ensemble, utilizing an advanced hyperparameter optimization framework to strike a balance between predictive accuracy and computational efficiency. The DNN features a five-layer feedforward architecture (512-256-128-64-32 neurons) with residual connections to mitigate vanishing gradients and LeakyReLU activations (α=0.1) to ensure robust gradient flow. Hyperparameters were tuned using nested 10-fold cross-validation, with the outer loop assessing generalization and the inner loop optimizing via grid search over: learning rate ([1e-5, 1e-2], default 0.0001 with Adam optimizer), L2 regularization ([1e-4, 1e-1], selected 0.001), dropout rate ([0.5, 0.9], chosen 0.8), batch size ([32, 256], chosen  64), and network depth (3–5 layers, selected 5). Early stopping (patience=15) prevents overfitting. The RF component uses 200 estimators with a maximum depth of 42, optimized via out-of-bag error analysis. Predictions are combined via a weighted ensemble (α tuned in [0.6-0.8] based on validation performance. This hybrid architecture, validated through 10-fold cross-validation, leverages the DNN’s capacity for complex patterns and the RF’s robustness, achieving high accuracy and efficiency. 

## 📂 Input Data Requirements
**Accepted Formats:**
- Input data format is given in Example_files folder
- For generating Training and Testing set use clustering.py 
- For calculating Additive matrix use Additive_Dominance.R
##  User's Note:
- Please refer key_note_to_users file for package installation

