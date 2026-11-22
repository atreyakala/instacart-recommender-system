# Instacart Reorder Prediction & Recommendation System

## Project Description

Build a **two-layer system** on Instacart order-history data.

**Layer 1 (Predictive):** Predict the probability that a user reorders a specific product in their next order — at the `(user, product)` pair level.

**Layer 2 (Recommendation):** Global co-occurrence matrix + cosine similarity — "people who buy X also buy Y" — independent of user history.

## Tech Stack

| Category | Tools |
| :--- | :--- |
| Programming | Python, Google Colab |
| Data Handling | Pandas, NumPy, SciPy (sparse matrices) |
| Modeling | Scikit-Learn (Logistic Regression, Random Forest, Gradient Boosting), LightGBM, XGBoost |
| Recommendation Layer | Scikit-Learn `cosine_similarity`, SciPy sparse matrices |
| Explainability | SHAP |
| Optimization | GridSearchCV, RandomizedSearchCV |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Custom F1-on-basket function |

## Core Instructions

### Data & Sampling
- Start with EDA — order frequency, basket size, reorder rate, days-since-prior-order.
- Use a 20k user sample.
- Compute all feature engineering strictly on historical data prior to the target validation order — no future data leakage into training.

### Layer 1 — Predictive Modeling
- Use three feature groups: user, product, user-product interaction.
- Predict at the `(user, product)` pair level.
- Use a time-aware split.
- Define handling for users with only 1 order (no prior order to build features from).
- Handle class imbalance.
- Primary model: LightGBM / XGBoost (decision pending — see [Open Questions](#open-questions)).
- Baseline: Logistic Regression.
- Comparison: Random Forest, Gradient Boosting.
- Pipeline order: baseline → comparison → tuning → SHAP → business impact.
- Use SHAP for explainability.

### Layer 2 — Recommendation
- Use co-occurrence matrix + cosine similarity — not GNN/TwoTower/ALS, not Apriori/FP-Growth.
- Normalize co-occurrence counts (not raw counts) so very popular products don't dominate every pairing.
- Recommendation pipeline: baskets → co-occurrence matrix → cosine similarity → lookup.

### Evaluation & Business Impact
- Report F1 on predicted vs. actual next-basket.
- For each model show: accuracy, precision, recall, F1, lift.
- Report a lift number for business impact.

### Constraints
- No deployment, no Streamlit, no model persistence.
- Keep Layer 1 and Layer 2 as separate pipelines.
- Add a `.gitignore` covering: virtual environments, cache directories, checkpoints, local CSVs.
