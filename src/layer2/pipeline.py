"""Layer 2 orchestration pipeline: baskets -> cooccurrence -> cosine -> lookup."""

from __future__ import annotations

from src.layer2.baskets import build_baskets
from src.layer2.cooccurrence import build_cooccurrence
from src.layer2.similarity import compute_similarity
from src.layer2.recommender import Recommender


def run_layer2_pipeline(order_products: pd.DataFrame, orders: pd.DataFrame, products: pd.DataFrame, top_k: int = 10):
    """Run the full Layer 2 pipeline."""
    baskets, product_to_idx, idx_to_product = build_baskets(order_products, orders)
    B, C = build_cooccurrence(baskets, product_to_idx)
    topk_dict = compute_similarity(B, C, top_k=top_k)

    id2name = dict(zip(products["product_id"], products["product_name"])) if "product_name" in products.columns else {}
    recommender = Recommender(topk_dict, id2name, idx_to_product)

    # Sample recommendations
    sample_products = list(product_to_idx.keys())[:5]
    print("Layer 2 sample recommendations:")
    for pid in sample_products:
        recs = recommender.recommend(pid, k=3)
        name = id2name.get(pid, str(pid))
        print(f"  {name}: {recs}")

    return {
        "recommender": recommender,
        "n_baskets": len(baskets),
        "n_products": len(product_to_idx),
        "topk_dict": topk_dict,
    }
