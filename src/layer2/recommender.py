"""Recommendation lookup API."""

from __future__ import annotations


class Recommender:
    def __init__(self, topk_dict: dict, id2name: dict, idx_to_product: dict):
        """
        topk_dict: {product_index: [(related_product_index, score), ...]}
        id2name: {product_id: product_name}
        idx_to_product: {product_index: product_id}
        """
        self.topk_dict = topk_dict
        self.id2name = id2name
        self.idx_to_product = idx_to_product
        self.product_to_idx = {pid: idx for idx, pid in idx_to_product.items()}

    def recommend(self, product_id, k: int = 5):
        """Return top-k recommendations for a product_id."""
        idx = self.product_to_idx.get(product_id)
        if idx is None:
            return []
        related = self.topk_dict.get(idx, [])[:k]
        return [
            (self.id2name.get(self.idx_to_product[rel_idx], str(self.idx_to_product[rel_idx])), score)
            for rel_idx, score in related
        ]

    def recommend_by_name(self, name: str, k: int = 5):
        """Return recommendations by product name (fuzzy match by exact name)."""
        for pid, pname in self.id2name.items():
            if pname == name:
                return self.recommend(pid, k)
        return []

    def recommend_for_basket(self, product_ids, k: int = 5):
        """Aggregate scores across basket items, dedupe."""
        scores = {}
        for pid in product_ids:
            for rel_pid, score in self.topk_dict.get(self.product_to_idx.get(pid, -1), []):
                scores[rel_pid] = scores.get(rel_pid, 0) + score
        # Remove items already in basket
        basket_set = set(product_ids)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [
            (self.id2name.get(pid, str(pid)), score)
            for pid, score in ranked
            if pid not in basket_set
        ][:k]
