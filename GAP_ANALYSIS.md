# Gap Analysis — Instacart Recommender System (200-User Pilot)

## Status
All code files have been written but **not yet executed** — no Instacart CSV data is present in `data/raw/`, so the pipeline cannot run end-to-end.

## Missing pieces vs README requirements

| Requirement | Status |
|---|---|
| EDA (order frequency, basket size, reorder rate, days-since-prior-order) | Not implemented |
| SHAP explainability | Not implemented |
| RandomizedSearchCV / GridSearchCV tuning | Not implemented |
| Visualizations (ROC, PR, lift curves, SHAP plots) | Not implemented |
| `.gitignore` | Not created |
| Model persistence check (none allowed) | Compliant (none implemented) |
| Separate Layer 1 / Layer 2 pipelines | Compliant |

## Unverified assumptions

1. **CSV column names** match the standard Instacart dataset (orders, order_products__prior/train, products, aisles, departments). Not yet validated against real data.
2. **LightGBM / XGBoost availability** — code falls back to GradientBoosting if missing, but this should be confirmed.
3. **`days_since_prior_order` NaN handling** for first orders — not explicitly handled in split/features.
4. **`eval_set` field** — code ignores it and uses `order_number` for time-aware split; this matches the README's intent but needs real-data validation.

## Scaling from 200 to 20,000 users

- **Memory**: Feature matrix at 200 users is small; at 20k users with candidate-pair generation, could grow large. Sparse matrices already used in Layer 2; Layer 1 uses dense pandas — may need sparse conversion.
- **Runtime**: 5 models × grid-search threshold tuning is fast at 200 users; at 20k users, LightGBM/XGBoost training with early stopping should still be manageable.
- **Candidate generation**: Negative sampling (products never bought by user) is not yet implemented — at 20k users, all-pairs would be too large; need sampling strategy.

## What's needed for end-to-end execution

1. Download Instacart CSVs into `data/raw/`.
2. Create `.gitignore` (venv, `__pycache__`, `*.csv`, `*.pkl`, `*.ipynb`, `.DS_Store`).
3. Run `pip install -r requirements.txt`.
4. Run `pytest tests/` to verify leakage guard.
5. Run the pipeline with a driver script that loads data → samples 200 users → runs Layer 1 and Layer 2 → evaluation.

## Recommended next steps

1. Obtain Instacart data and place in `data/raw/`.
2. Add EDA module and SHAP explainer.
3. Add hyperparameter tuning (RandomizedSearchCV).
4. Add visualizations module.
5. Create `.gitignore`.
6. Run pilot on 200 users, verify metrics are non-degenerate, then scale to 20k.
