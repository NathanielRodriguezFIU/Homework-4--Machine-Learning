import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "product_with_clusters.csv"))
    features = [
        c
        for c in ["price", "cost", "units_sold", "promotion_frequency", "shelf_level"]
        if c in df.columns
    ]

    # cluster scatter (price vs units_sold)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df, x="price", y="units_sold", hue="cluster", palette="tab10")
    plt.title("Cluster: Price vs Units Sold")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "cluster_scatter.png"), dpi=150)
    plt.close()
    print("Saved cluster_scatter.png")

    # feature importance via RF trained on entire dataset
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(df[features], df["profit"])
    imp = pd.DataFrame(
        {"feature": features, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=True)

    plt.figure(figsize=(6, 4))
    plt.barh(imp["feature"], imp["importance"])
    plt.title("Feature importance (RandomForest)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("Saved feature_importance.png")


if __name__ == "__main__":
    main()
