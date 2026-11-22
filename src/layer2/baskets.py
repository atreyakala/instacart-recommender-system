"""Build baskets from order-product data."""

from __future__ import annotations

import pandas as pd


def build_baskets(order_products: pd.DataFrame, orders: pd.DataFrame, min_basket_size: int = 2):
    """
    Group order_products by order_id -> set of product_ids per basket.

    Returns:
        baskets: dict {order_id: set[product_id]}
        product_to_idx: dict {product_id: column index}
        idx_to_product: dict {column index: product_id}
    """
    baskets_series = order_products.groupby("order_id")["product_id"].apply(set)
    baskets = {oid: set(products) for oid, products in baskets_series.items() if len(products) >= min_basket_size}

    all_products = sorted(set().union(*baskets.values())) if baskets else []
    product_to_idx = {pid: i for i, pid in enumerate(all_products)}
    idx_to_product = {i: pid for pid, i in product_to_idx.items()}

    return baskets, product_to_idx, idx_to_product
