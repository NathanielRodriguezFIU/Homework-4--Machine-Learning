import os

import joblib
import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE, "data", "product_sales.csv")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def iqr_bounds(s: pd.Series):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def main():
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", DATA_PATH, "shape:", df.shape)

    # Remove rows missing product identifiers
    df = df.dropna(subset=["product_id", "product_name"]).reset_index(drop=True)

    # numeric columns we care about
    numeric_cols = [
        "price",
        "cost",
        "units_sold",
        "promotion_frequency",
        "shelf_level",
        "profit",
    ]
    # Fill numeric missing with median
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # Fill category with mode if present
    if "category" in df.columns:
        df["category"] = df["category"].fillna(df["category"].mode().iloc[0])

    # Outlier capping (IQR)
    for c in numeric_cols:
        if c in df.columns:
            low, high = iqr_bounds(df[c])
            df[c] = df[c].clip(lower=low, upper=high)

    # Save cleaned data
    cleaned_path = os.path.join(RESULTS_DIR, "cleaned_data.csv")
    df.to_csv(cleaned_path, index=False)
    print("Saved cleaned data to:", cleaned_path)


if __name__ == "__main__":
    main()
