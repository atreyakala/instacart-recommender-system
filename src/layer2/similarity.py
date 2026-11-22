"""Cosine similarity with TF-IDF-style normalization."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


def compute_similarity(B: sparse.csr_matrix, C: sparse.csr_matrix, top_k: int = 10):
    """
    TF-IDF weighting + cosine similarity.

    idf = log(n_baskets / (1 + df)) where df = column sums of B.
    B_tfidf = B * idf
    C_norm = B_tfidf.T @ B_tfidf
    Cosine = L2-normalize rows of C_norm, then matmul.
    Returns dict {product_id: [(related_product_id, score)]}.
    """
    n_baskets = B.shape[0]
    df = np.asarray(B.sum(axis=0)).ravel().astype(np.float64)
    idf = np.log(n_baskets / (1.0 + df))
    idf_diag = sparse.diags(idf)

    B_tfidf = B @ idf_diag
    C_norm = B_tfidf.T @ B_tfidf

    Cn = normalize(C_norm, norm="l2", axis=1)
    sim = Cn @ Cn.T

    topk_dict = {}
    for i in range(sim.shape[0]):
        row = sim.getrow(i)
        col_idx = row.indices
        scores = row.data
        order = np.argsort(-scores)[:top_k]
        topk_dict[i] = [(int(col_idx[j]), float(scores[j])) for j in order]

    return topk_dict
