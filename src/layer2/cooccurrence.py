"""Co-occurrence matrix from baskets."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def build_cooccurrence(baskets: dict, product_to_idx: dict):
    """
    Build sparse binary incidence matrix B (baskets x products) and
    co-occurrence matrix C = B.T @ B (products x products).

    Zeroes the diagonal so a product doesn't co-occur with itself.
    """
    n_baskets = len(baskets)
    n_products = len(product_to_idx)

    rows = []
    cols = []
    for bid, products in baskets.items():
        for pid in products:
            if pid in product_to_idx:
                rows.append(bid)
                cols.append(product_to_idx[pid])

    # Use row position as basket index
    basket_to_row = {bid: i for i, bid in enumerate(baskets.keys())}
    rows = [basket_to_row[bid] for bid in rows]

    B = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_baskets, n_products),
    )

    C = B.T @ B
    C.setdiag(0)
    C.eliminate_zeros()

    return B, C
