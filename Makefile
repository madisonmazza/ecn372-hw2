# ECN372 HW2: Online News Popularity Prediction (R)
# Run all targets from project root.
# For grading: place test.csv in data/raw/; then "make evaluate" prints test MSE only.
# Requires: R >= 4.0, package xgboost (make install_deps)

RSCRIPT ?= Rscript

.PHONY: train evaluate install_deps

install_deps:
	$(RSCRIPT) -e 'if (!requireNamespace("xgboost", quietly = TRUE)) install.packages("xgboost", repos = "https://cloud.r-project.org", quiet = TRUE)'

train: install_deps
	$(RSCRIPT) src/train.R --train-path data/raw/train.csv --model-dir model

evaluate: install_deps
	$(RSCRIPT) src/evaluate.R
