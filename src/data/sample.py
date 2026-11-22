"""User sampling for the 200-user pilot."""

import numpy as np
import pandas as pd

SAMPLE_SIZE = 200
RANDOM_STATE = 42


def sample_users(orders: pd.DataFrame, n_users: int = SAMPLE_SIZE, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Return orders restricted to a random sample of n_users."""
    user_ids = orders["user_id"].unique()
    if n_users >= len(user_ids):
        return orders.copy()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(user_ids, size=n_users, replace=False)
    return orders[orders["user_id"].isin(sampled)].copy()


def restrict_to_sample(dfs: dict[str, pd.DataFrame], user_ids) -> dict[str, pd.DataFrame]:
    """Filter order-product tables to orders belonging to sampled users."""
    orders_sample = dfs["orders"][dfs["orders"]["user_id"].isin(user_ids)].copy()
    order_ids = set(orders_sample["order_id"])
    out = {"orders": orders_sample}
    for key in ("order_products_prior", "order_products_train"):
        mask = dfs[key]["order_id"].isin(order_ids)
        out[key] = dfs[key][mask].copy()
    out["products"] = dfs["products"]
    out["aisles"] = dfs["aisles"]
    out["departments"] = dfs["departments"]
    return out
