"""Test: leakage guard and label sanity."""

import numpy as np
import pandas as pd
from src.data.split import temporal_split


def test_no_leakage():
    """No order_id appears in both history and target for the same user."""
    orders = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "user_id": [1, 1, 2, 2, 2],
        "order_number": [1, 2, 1, 2, 3],
    })
    split = temporal_split(orders, min_orders=2)
    target_ids = set(split["target_orders"]["order_id"])
    history_ids = set(split["history_orders"]["order_id"])
    overlap = target_ids & history_ids
    assert len(overlap) == 0, f"Leakage: order_ids in both history and target: {overlap}"


def test_target_is_last_order():
    """Target order_number equals user's max order_number."""
    orders = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "user_id": [1, 1, 2, 2],
        "order_number": [1, 2, 1, 2],
    })
    split = temporal_split(orders, min_orders=2)
    targets = split["target_orders"].set_index("user_id")["order_number"]
    max_orders = orders.groupby("user_id")["order_number"].max()
    for uid in targets.index:
        assert targets[uid] == max_orders[uid], f"User {uid} target is not last order"
