"""Pair table builder: merge target labels with features."""

import pandas as pd
import numpy as np


class PairTableBuilder:
    def build_pairs(
        self,
        target_pairs: pd.DataFrame,
        user_features: pd.DataFrame,
        product_features: pd.DataFrame,
        up_features: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Index, pd.Index]:
        """
        Merge (user_id, product_id, reordered_next) labels with features.

        Returns X (feature DataFrame), y (label Series), user_ids, product_ids.
        Cold-start pairs (missing UP features) are filled with 0.
        """
        df = target_pairs.merge(
            user_features, on="user_id", how="left"
        ).merge(
            product_features, on="product_id", how="left"
        ).merge(
            up_features, on=["user_id", "product_id"], how="left"
        )

        y = df["reordered_next"].astype(int)
        user_ids = df["user_id"]
        product_ids = df["product_id"]

        feature_cols = [
            c for c in df.columns
            if c not in ("user_id", "product_id", "reordered_next")
        ]
        X = df[feature_cols].fillna(0)

        return X, y, user_ids, product_ids
