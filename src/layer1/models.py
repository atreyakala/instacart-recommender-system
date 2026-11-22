"""Model wrappers for Layer 1."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def train_logistic_regression(X, y, **kwargs):
    params = {"class_weight": "balanced", "max_iter": 1000, "n_jobs": -1}
    params.update(kwargs)
    model = LogisticRegression(**params)
    model.fit(X, y)
    return model


def train_random_forest(X, y, **kwargs):
    params = {"class_weight": "balanced", "n_estimators": 200, "max_depth": 10, "n_jobs": -1}
    params.update(kwargs)
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model


def train_gradient_boosting(X, y, **kwargs):
    params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5}
    params.update(kwargs)
    model = GradientBoostingClassifier(**params)
    model.fit(X, y)
    return model


def train_lightgbm(X, y, X_val=None, y_val=None, **kwargs):
    try:
        import lightgbm as lgb
    except ImportError:
        print("Warning: lightgbm not installed; falling back to GradientBoosting.")
        return train_gradient_boosting(X, y, **kwargs)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbose": -1,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
    }
    params.update(kwargs)
    model = lgb.LGBMClassifier(**params)
    if X_val is not None and y_val is not None:
        model.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    else:
        model.fit(X, y)
    return model


def train_xgboost(X, y, X_val=None, y_val=None, **kwargs):
    try:
        import xgboost as xgb
    except ImportError:
        print("Warning: xgboost not installed; falling back to GradientBoosting.")
        return train_gradient_boosting(X, y, **kwargs)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
    }
    params.update(kwargs)
    model = xgb.XGBClassifier(**params)
    if X_val is not None and y_val is not None:
        model.fit(X, y, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
    else:
        model.fit(X, y)
    return model
