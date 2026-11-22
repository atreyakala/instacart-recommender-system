"""Instacart data loading utilities."""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

DTYPES_ORDERS = {
    "order_id": "int32",
    "user_id": "int32",
    "eval_set": "category",
    "order_number": "int8",
    "order_dow": "int8",
    "order_hour_of_day": "int8",
    "days_since_prior_order": "float32",
}

DTYPES_ORDER_PRODUCTS = {
    "order_id": "int32",
    "product_id": "int32",
    "add_to_cart_order": "int16",
    "reordered": "int8",
}

DTYPES_PRODUCTS = {
    "product_id": "int32",
    "aisle_id": "int16",
    "department_id": "int16",
}


class InstacartLoader:
    def __init__(self, raw_dir: str | Path = RAW_DIR):
        self.raw_dir = Path(raw_dir)

    def load_orders(self) -> pd.DataFrame:
        return pd.read_csv(self.raw_dir / "orders.csv", dtype=DTYPES_ORDERS)

    def load_order_products(self, split: str = "prior") -> pd.DataFrame:
        fname = f"order_products__{split}.csv"
        return pd.read_csv(self.raw_dir / fname, dtype=DTYPES_ORDER_PRODUCTS)

    def load_products(self) -> pd.DataFrame:
        return pd.read_csv(self.raw_dir / "products.csv", dtype=DTYPES_PRODUCTS)

    def load_aisles(self) -> pd.DataFrame:
        return pd.read_csv(self.raw_dir / "aisles.csv")

    def load_departments(self) -> pd.DataFrame:
        return pd.read_csv(self.raw_dir / "departments.csv")

    def load_all(self) -> dict[str, pd.DataFrame]:
        return {
            "orders": self.load_orders(),
            "order_products_prior": self.load_order_products("prior"),
            "order_products_train": self.load_order_products("train"),
            "products": self.load_products(),
            "aisles": self.load_aisles(),
            "departments": self.load_departments(),
        }
