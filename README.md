ECN372 HW2

## Environment

- **R**: 4.0 or higher (standard installation from [CRAN](https://cran.r-project.org/) or your system package manager).
- **R package**: `xgboost` only. Install once via `make install_deps` or from R: `install.packages("xgboost", repos = "https://cloud.r-project.org")`.
- No other dependencies; the repo uses only base R and `xgboost`.

## Final model

The submission uses a **single final model**: an **XGBoost** (gradient boosting) regression model for the continuous target `shares`. It was chosen by the structured process below (variable selection and cross-validation). No ensemble or multiple models are used at evaluation time.

## Methodology

### 1. Preprocessing

- **Excluded columns**: `url` (non-predictive identifier) and `shares` (target).
- **Features**: All 58 predictive attributes from the dataset are considered initially.
- **Target**: `shares` (continuous). No transformation applied; we predict raw shares to minimize MSE directly.
- **Missing values**: None in the dataset per the `.names` file.
- **Scaling**: None; tree-based models (XGBoost) are scale-invariant.

### 2. Variable / Feature Selection

Feature selection is performed using **XGBoost feature importance**:

1. Fit a preliminary XGBoost model on all 58 features.
2. Rank features by importance (via `xgb.importance()`).
3. Retain the top 45 features for the final model.

This reduces noise and overfitting while keeping the most informative predictors (e.g., keyword metrics, LDA topics, self-reference shares, content statistics). We use 45 features as a balance: enough to capture signal, few enough to avoid overfitting (the importance ranking drops the tail of weak predictors). The selection logic is in `src/train.R` in the function `select_features()`.

### 3. Model Choice

We use **XGBoost** (gradient boosting, `xgboost` package):

- **Rationale**: The original paper (Fernandes et al., EPIA 2015) reports Random Forest as best for a binary classification task. For regression (predicting continuous shares), gradient boosting typically outperforms Random Forest due to its additive, stage-wise fitting and better handling of continuous targets.
- **Regularization**: `min_child_weight = 5`, `subsample = 0.8` to reduce overfitting.

### 4. Hyperparameter Tuning

Hyperparameters are chosen via **5-fold cross-validation** on the training set:

- **Grid**: `nrounds` ∈ {200, 300, 400}, `max_depth` ∈ {6, 7}, `eta` (learning rate) = 0.05. We fix `eta = 0.05` to keep training stable and search over depth and number of rounds.
- **Metric**: MSE (minimize).
- **Process**: Implemented in `src/train.R` in the function `tune_hyperparameters()`; 5-fold CV is used so that the choice is based on out-of-fold performance and the final model is fit on the full training set.

### 5. Training and Evaluation Flow

1. **Train** (`make train` or automatic during `make evaluate`):
   - Load `data/raw/train.csv`
   - Select features, tune hyperparameters, fit final model
   - Save artifact to `model/model.rds` (model + feature names + params)

2. **Evaluate** (`make evaluate`):
   - Load model (or run `train.R` if missing)
   - Load `data/raw/test.csv`
   - Predict shares, compute MSE
   - Print `MSE: <value>` to stdout

## Project Structure

```
ecn372-hw2-1/
├── data/
│   └── raw/
│       ├── train.csv      # Training data (included)
│       └── test.csv       # Test data (added at grading)
├── model/
│   └── model.rds          # Trained model artifact (created by make train)
├── src/
│   ├── train.R            # Training and model persistence
│   └── evaluate.R         # Load model, predict, print MSE
├── Makefile
├── requirements.txt       # R package note (xgboost)
├── OnlineNewsPopularity.names
├── README.md
└── AI_USAGE.md
```
## Model selection (visibility in code)

The structured process—variable selection and cross-validation—is fully visible in the code:

- **Variable selection**: `src/train.R`, function `select_features()` (XGBoost importance, top 45 features).
- **Hyperparameter tuning**: `src/train.R`, function `tune_hyperparameters()` (5-fold CV over the grid above).
- **Final fit**: `src/train.R`, `main()` (single XGBoost model saved to `model/model.rds`).

There is no separate “exploration” script; the same pipeline that produces the submitted model is in `train.R` and is documented in this README.

## Prediction performance

The final model is chosen to be competitive with a hidden baseline: we use a strong regression method (XGBoost), variable selection to reduce overfitting, and cross-validated tuning so the chosen hyperparameters generalize. Outperforming the baseline depends on test data; this setup is intended to give a good chance of doing so.

## References

- K. Fernandes, P. Vinagre, P. Cortez. *A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News*. EPIA 2015.
- [UCI Machine Learning Repository: Online News Popularity](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)

