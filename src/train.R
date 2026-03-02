##
## train.R -- Train the single final model for article popularity (shares).
##
## Pipeline: load data -> variable selection (XGBoost importance) -> 5-fold CV
## tuning -> fit final XGBoost model -> save artifact to model/model.rds.
##
## Usage: Rscript src/train.R [--train-path PATH] [--model-dir DIR]
##        Run from project root; defaults: data/raw/train.csv, model/
##

# --- Constants (match dataset schema) ---
NON_FEATURE_COLS <- c("url", "shares")
TARGET <- "shares"

# --- Data loading ---
load_data <- function(path) {
  df <- read.csv(path)
  feature_cols <- setdiff(names(df), NON_FEATURE_COLS)
  X <- df[, feature_cols, drop = FALSE]
  y <- df[[TARGET]]
  list(X = X, y = y, feature_cols = feature_cols)
}

# --- Variable selection: top n_features by XGBoost importance ---
select_features <- function(X, y, n_features = 45) {
  if (!requireNamespace("xgboost", quietly = TRUE)) stop("Package xgboost required")
  dtrain <- xgboost::xgb.DMatrix(
    data = as.matrix(X),
    label = y
  )
  fit <- xgboost::xgb.train(
    data = dtrain,
    params = list(
      objective = "reg:squarederror",
      max_depth = 5,
      eta = 0.1,
      subsample = 0.8,
      min_child_weight = 5,
      seed = 42
    ),
    nrounds = 100,
    verbose = 0
  )
  imp <- xgboost::xgb.importance(model = fit, feature_names = colnames(X))
  cols <- as.character(imp$Feature)
  if (length(cols) > n_features) cols <- cols[seq_len(n_features)]
  cols
}

# --- Hyperparameter tuning via n_folds cross-validation (minimize MSE) ---
tune_hyperparameters <- function(X, y, n_folds = 5, seed = 42) {
  if (!requireNamespace("xgboost", quietly = TRUE)) stop("Package xgboost required")
  set.seed(seed)
  n <- nrow(X)
  folds <- sample(rep(seq_len(n_folds), length.out = n))

  param_grid <- list(
    list(nrounds = 200, max_depth = 6, eta = 0.05),
    list(nrounds = 300, max_depth = 6, eta = 0.05),
    list(nrounds = 200, max_depth = 7, eta = 0.05),
    list(nrounds = 300, max_depth = 7, eta = 0.05),
    list(nrounds = 400, max_depth = 6, eta = 0.05)
  )

  best_mse <- Inf
  best_params <- param_grid[[1]]

  for (params in param_grid) {
    mse_fold <- numeric(n_folds)
    for (k in seq_len(n_folds)) {
      i_val <- which(folds == k)
      i_tr <- which(folds != k)
      dtrain <- xgboost::xgb.DMatrix(
        data = as.matrix(X[i_tr, , drop = FALSE]),
        label = y[i_tr]
      )
      dval <- xgboost::xgb.DMatrix(
        data = as.matrix(X[i_val, , drop = FALSE]),
        label = y[i_val]
      )
      fit <- xgboost::xgb.train(
        data = dtrain,
        params = list(
          objective = "reg:squarederror",
          max_depth = params$max_depth,
          eta = params$eta,
          subsample = 0.8,
          min_child_weight = 5,
          seed = seed
        ),
        nrounds = params$nrounds,
        verbose = 0
      )
      pred <- predict(fit, dval)
      mse_fold[k] <- mean((y[i_val] - pred)^2)
    }
    mse <- mean(mse_fold)
    if (mse < best_mse) {
      best_mse <- mse
      best_params <- params
    }
  }
  best_params
}

# --- Main: load -> select features -> tune -> fit final model -> save ---
main <- function(train_path = "data/raw/train.csv", model_dir = "model") {
  if (!file.exists(train_path)) {
    stop("Training data not found: ", train_path)
  }

  data_list <- load_data(train_path)
  X <- data_list$X
  y <- data_list$y

  feature_cols <- select_features(X, y, n_features = 45)
  X_sel <- X[, feature_cols, drop = FALSE]

  params <- tune_hyperparameters(X_sel, y, n_folds = 5)

  dtrain <- xgboost::xgb.DMatrix(
    data = as.matrix(X_sel),
    label = y
  )
  model <- xgboost::xgb.train(
    data = dtrain,
    params = list(
      objective = "reg:squarederror",
      max_depth = params$max_depth,
      eta = params$eta,
      subsample = 0.8,
      min_child_weight = 5,
      seed = 42
    ),
    nrounds = params$nrounds,
    verbose = 0
  )

  dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
  artifact <- list(
    model = model,
    feature_cols = feature_cols,
    params = params
  )
  saveRDS(artifact, file.path(model_dir, "model.rds"))
  message("Model saved to ", model_dir, "/model.rds")
}

# --- Parse command-line args and run (when executed via Rscript) ---
args <- commandArgs(trailingOnly = TRUE)
train_path <- "data/raw/train.csv"
model_dir <- "model"
i <- 1
while (i <= length(args)) {
  if (args[i] == "--train-path" && i < length(args)) {
    train_path <- args[i + 1]
    i <- i + 2
    next
  }
  if (args[i] == "--model-dir" && i < length(args)) {
    model_dir <- args[i + 1]
    i <- i + 2
    next
  }
  i <- i + 1
}
main(train_path = train_path, model_dir = model_dir)
