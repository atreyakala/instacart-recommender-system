"""Target label construction: (user, product, reordered_next)."""

import pandas as pd


def build_target_pairs(
    target_orders: pd.DataFrame,
    order_products_train: pd.DataFrame,
    history_products: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build (user_id, product_id, reordered_next) label frame.

    positives = products in each user's target order (label=1)
    candidates = all products user ever bought in history (label=0 if not in target)
    """
    positives = order_products_train.merge(
        target_orders[["order_id", "user_id"]], on="order_id"
    )[["user_id", "product_id"]].drop_duplicates()
    positives["reordered_next"] = 1

    pairs = history_products.merge(positives, on=["user_id", "product_id"], how="left")
    pairs["reordered_next"] = pairs["reordered_next"].fillna(0).astype("int8")
    return pairs[["user_id", "product_id", "reordered_next"]]
