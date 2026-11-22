"""Layer 1 orchestration pipeline: baseline -> comparison -> tuning -> impact."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.data.split import temporal_split
from src.features.targets import build_target_pairs
from src.layer1.featurize import FeatureEngineer
from src.layer1.preprocess import PairTableBuilder
from src.layer1.models import (
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
    train_lightgbm,
    train_xgboost,
)
from src.layer1.evaluate import compute_all_metrics
from src.layer1.imbalance import find_optimal_threshold


def _history_products(orders_hist: pd.DataFrame, order_products_prior: pd.DataFrame) -> pd.DataFrame:
    op_hist = order_products_prior.merge(
        orders_hist[["order_id", "user_id"]], on="order_id"
    )
    return op_hist[["user_id", "product_id"]].drop_duplicates()


def run_layer1_pipeline(
    orders: pd.DataFrame,
    order_products_prior: pd.DataFrame,
    order_products_train: pd.DataFrame,
    products: pd.DataFrame,
) -> dict:
    """Run the full Layer 1 pipeline on the sampled data."""
    split = temporal_split(orders, min_orders=3)
    target_orders = split["target_orders"]
    history_orders = split["history_orders"]

    history_products = _history_products(history_orders, order_products_prior)
    target_pairs = build_target_pairs(target_orders, order_products_train, history_products)

    if target_pairs.empty:
        return {"error": "No target pairs generated (too few users/orders)."}

    # Feature engineering
    target_order_numbers = target_orders.merge(
        orders[["order_id", "user_id", "order_number"]], on="order_id"
    ).set_index("user_id")["order_number"]

    fe = FeatureEngineer(orders, order_products_prior, products)
    user_feats, product_feats, up_feats = fe.build_all_features(target_order_numbers)

    # Build pair table
    builder = PairTableBuilder()
    X, y, user_ids, product_ids = builder.build_pairs(
        target_pairs, user_feats, product_feats, up_feats
    )

    if X.empty or len(X) < 10:
        return {"error": "Insufficient data for training."}

    # Train/validation split by user (time-aware already done via target = last order)
    # Use a simple random split for pilot validation
    n_val = max(1, int(len(X) * 0.2))
    val_idx = np.random.default_rng(42).choice(len(X), size=n_val, replace=False)
    train_mask = np.ones(len(X), dtype=bool)
    train_mask[val_idx] = False

    X_train, X_val = X.iloc[train_mask], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_mask], y.iloc[val_idx]
    user_ids_val = user_ids.iloc[val_idx]

    results = {}

    # 1. Baseline: Logistic Regression
    lr = train_logistic_regression(X_train, y_train)
    lr_proba = lr.predict_proba(X_val)[:, 1]
    lr_thresh = find_optimal_threshold(y_val.values, lr_proba, metric="f1")
    results["LogisticRegression"] = compute_all_metrics(
        y_val.values, lr_proba, user_ids_val.values, threshold=lr_thresh
    )

    # 2. Comparison: Random Forest
    rf = train_random_forest(X_train, y_train)
    rf_proba = rf.predict_proba(X_val)[:, 1]
    rf_thresh = find_optimal_threshold(y_val.values, rf_proba, metric="f1")
    results["RandomForest"] = compute_all_metrics(
        y_val.values, rf_proba, user_ids_val.values, threshold=rf_thresh
    )

    # 3. Comparison: Gradient Boosting
    gb = train_gradient_boosting(X_train, y_train)
    gb_proba = gb.predict_proba(X_val)[:, 1]
    gb_thresh = find_optimal_threshold(y_val.values, gb_proba, metric="f1")
    results["GradientBoosting"] = compute_all_metrics(
        y_val.values, gb_proba, user_ids_val.values, threshold=gb_thresh
    )

    # 4. Primary: LightGBM
    lgb = train_lightgbm(X_train, y_train, X_val=X_val, y_val=y_val)
    lgb_proba = lgb.predict_proba(X_val)[:, 1]
    lgb_thresh = find_optimal_threshold(y_val.values, lgb_proba, metric="f1")
    results["LightGBM"] = compute_all_metrics(
        y_val.values, lgb_proba, user_ids_val.values, threshold=lgb_thresh
    )

    # 5. XGBoost comparison
    xgb = train_xgboost(X_train, y_train, X_val=X_val, y_val=y_val)
    xgb_proba = xgb.predict_proba(X_val)[:, 1]
    xgb_thresh = find_optimal_threshold(y_val.values, xgb_proba, metric="f1")
    results["XGBoost"] = compute_all_metrics(
        y_val.values, xgb_proba, user_ids_val.values, threshold=xgb_thresh
    )

    results_df = pd.DataFrame(results).T
    print(results_df.round(4))

    return {
        "results": results_df,
        "models": {"lr": lr, "rf": rf, "gb": gb, "lightgbm": lgb, "xgboost": xgb},
        "n_users": orders["user_id"].nunique(),
        "n_pairs": len(X),
        "n_single_order_excluded": orders.groupby("user_id")["order_id"].nunique().pipe(
            lambda s: int((s < 3).sum())
        ),
    }
