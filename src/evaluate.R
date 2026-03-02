##
## evaluate.R -- Compute test MSE for the final model.
##
## Loads model/model.rds (or runs train.R if missing), reads data/raw/test.csv,
## predicts shares, and prints only "MSE: <value>" to stdout.
##
## Usage: Rscript src/evaluate.R   (run from project root; see Makefile)
##

if (!requireNamespace("xgboost", quietly = TRUE)) {
  stop("Package xgboost required. Run: install.packages(\"xgboost\")")
}
suppressPackageStartupMessages(library(xgboost))

NON_FEATURE_COLS <- c("url", "shares")
TARGET <- "shares"

# Paths: assume run from project root (Makefile ensures this)
project_root <- getwd()
test_path <- file.path(project_root, "data", "raw", "test.csv")
model_path <- file.path(project_root, "model", "model.rds")

if (!file.exists(test_path)) {
  write(paste("Test data not found:", test_path), stderr())
  quit(save = "no", status = 1)
}

# Train on the fly if no saved model exists (subprocess; no stdout to keep output clean)
if (!file.exists(model_path)) {
  train_script <- file.path(project_root, "src", "train.R")
  train_path <- file.path(project_root, "data", "raw", "train.csv")
  model_dir <- file.path(project_root, "model")
  status <- system2("Rscript", c(train_script, "--train-path", train_path, "--model-dir", model_dir),
                    stdout = FALSE)
  if (status != 0) stop("Training failed")
}

artifact <- readRDS(model_path)
model <- artifact$model
feature_cols <- artifact$feature_cols

test_df <- read.csv(test_path)
y_true <- test_df[[TARGET]]
X_test <- test_df[, setdiff(names(test_df), NON_FEATURE_COLS), drop = FALSE]
# Align columns with training (missing features filled with 0)
for (c in feature_cols) {
  if (!c %in% names(X_test)) X_test[[c]] <- 0
}
X_test <- X_test[, feature_cols, drop = FALSE]

dtest <- xgboost::xgb.DMatrix(data = as.matrix(X_test))
y_pred <- predict(model, dtest)
mse <- mean((y_true - y_pred)^2)
cat("MSE: ", round(mse, 2), "\n", sep = "")
