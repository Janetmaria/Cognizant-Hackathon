"""
Module 5 - Shared evaluation metrics for all forecasting models.

Scores predictions from src/models/*.py against data/processed/model_holdout.csv
(config.PROCESSED_HOLDOUT_CSV, config.TARGET_COL) so every model is compared
on the same metric. Do not duplicate metric logic inside individual model files.
"""

# TODO(Module 5): implement evaluate(y_true, y_pred, weights=None) -> dict of metrics.
