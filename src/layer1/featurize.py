"""Feature engineering: user, product, and user-product interaction features.

All features computed strictly on orders before the target order (no leakage).
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, orders: pd.DataFrame, order_products_prior: pd.DataFrame, products: pd.DataFrame):
        self.orders = orders
        self.order_products_prior = order_products_prior
        self.products = products

    def _history_mask(self, orders_df: pd.DataFrame, max_order_number: pd.Series) -> pd.Series:
        """Keep only rows with order_number < max_order_number per user (leakage guard)."""
        merged = orders_df.merge(
            max_order_number.rename("max_order_number"), on="user_id", how="left"
        )
        return merged["order_number"] < merged["max_order_number"]

    def build_user_features(self, max_order_number: pd.Series) -> pd.DataFrame:
        """User-level features aggregated from history only."""
        orders_hist = self.orders[self._history_mask(self.orders, max_order_number)].copy()
        op_hist = self.order_products_prior.merge(
            orders_hist[["order_id", "user_id"]], on="order_id"
        )

        user_stats = (
            orders_hist.groupby("user_id")
            .agg(
                user_total_orders=("order_id", "nunique"),
                user_mean_days_since_prior=("days_since_prior_order", "mean"),
                user_std_days_since_prior=("days_since_prior_order", "std"),
            )
            .reset_index()
        )

        basket_stats = (
            op_hist.groupby(["user_id", "order_id"])["product_id"]
            .nunique()
            .groupby("user_id")
            .agg(user_mean_basket_size="mean", user_total_products="sum", user_distinct_products="nunique")
            .reset_index()
        )

        reorder_ratio = (
            op_hist.groupby("user_id")["reordered"]
            .agg(user_reorder_ratio="mean")
            .reset_index()
        )

        features = user_stats.merge(basket_stats, on="user_id", how="outer").merge(
            reorder_ratio, on="user_id", how="outer"
        )
        return features.set_index("user_id")

    def build_product_features(self, max_order_number: pd.Series) -> pd.DataFrame:
        """Product-level features aggregated from history only."""
        orders_hist = self.orders[self._history_mask(self.orders, max_order_number)].copy()
        op_hist = self.order_products_prior.merge(
            orders_hist[["order_id", "user_id"]], on="order_id"
        )

        prod_stats = (
            op_hist.groupby("product_id")
            .agg(
                prod_total_orders=("order_id", "nunique"),
                prod_total_reorders=("reordered", "sum"),
                prod_mean_add_to_cart_order=("add_to_cart_order", "mean"),
                prod_std_add_to_cart_order=("add_to_cart_order", "std"),
            )
            .reset_index()
        )
        prod_stats["prod_reorder_rate"] = (
            prod_stats["prod_total_reorders"] / prod_stats["prod_total_orders"]
        )
        return prod_stats.set_index("product_id")

    def build_up_features(self, max_order_number: pd.Series) -> pd.DataFrame:
        """User-product interaction features from history only."""
        orders_hist = self.orders[self._history_mask(self.orders, max_order_number)].copy()
        op_hist = self.order_products_prior.merge(
            orders_hist[["order_id", "user_id", "order_number"]], on="order_id"
        )

        up_stats = (
            op_hist.groupby(["user_id", "product_id"])
            .agg(
                up_orders_with_product=("order_id", "nunique"),
                up_reorders_of_product=("reordered", "sum"),
                up_first_order_number=("order_number", "min"),
                up_last_order_number=("order_number", "max"),
                up_mean_position_in_basket=("add_to_cart_order", "mean"),
                up_std_position_in_basket=("add_to_cart_order", "std"),
            )
            .reset_index()
        )
        up_stats["up_reorder_prob_given_seen"] = (
            up_stats["up_reorders_of_product"] / up_stats["up_orders_with_product"]
        )
        up_stats["up_orders_since_last"] = up_stats["up_last_order_number"]

        # max order_number per user (target boundary)
        user_max = orders_hist.groupby("user_id")["order_number"].max().rename("user_max_order")
        up_stats = up_stats.merge(user_max, on="user_id", how="left")
        up_stats["up_orders_since_last"] = up_stats["user_max_order"] - up_stats["up_last_order_number"]
        up_stats = up_stats.drop(columns=["user_max_order"])

        return up_stats.set_index(["user_id", "product_id"])

    def build_all_features(self, target_order_numbers: pd.Series) -> pd.DataFrame:
        """Join user, product, and UP features. Strictly historical."""
        max_order_number = target_order_numbers
        user_feats = self.build_user_features(max_order_number)
        product_feats = self.build_product_features(max_order_number)
        up_feats = self.build_up_features(max_order_number)
        return user_feats, product_feats, up_feats
