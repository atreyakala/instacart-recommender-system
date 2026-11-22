"""Time-aware split with leakage guard."""

import pandas as pd


def user_last_order(orders: pd.DataFrame) -> pd.DataFrame:
    """Most recent order per user (highest order_number)."""
    return orders.loc[orders.groupby("user_id")["order_number"].idxmax()]


def temporal_split(orders: pd.DataFrame, min_orders: int = 3) -> dict[str, pd.DataFrame]:
    """
    Per-user time-aware split.

    - target order = last order per user (highest order_number)
    - history = all orders strictly before the target order
    - Users with < min_orders are excluded (cold start, incl. single-order users)

    Returns {'target_orders': [...], 'history_orders': [...]}.
    Leakage guard: no order_id appears in both history and target for the same user.
    """
    order_counts = orders.groupby("user_id")["order_id"].nunique()
    valid_users = set(order_counts[order_counts >= min_orders].index)
    ord_f = orders[orders["user_id"].isin(valid_users)].copy()

    last = user_last_order(ord_f)

    # Build multi-index for fast set difference
    ord_idx = ord_f.set_index(["user_id", "order_number"])
    last_idx = last.set_index(["user_id", "order_number"])
    history_idx = ord_idx.index.difference(last_idx.index)
    history = ord_idx.loc[history_idx].reset_index()

    return {
        "target_orders": last[["order_id", "user_id", "order_number"]].reset_index(drop=True),
        "history_orders": history.reset_index(drop=True),
    }
