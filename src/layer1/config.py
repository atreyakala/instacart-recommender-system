"""Layer 1 configuration for the 200-user pilot."""

N_USERS = 200
RANDOM_STATE = 42
MIN_ORDERS = 3
TARGET_K = 10

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbose": -1,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

GB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": RANDOM_STATE,
}

LR_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
