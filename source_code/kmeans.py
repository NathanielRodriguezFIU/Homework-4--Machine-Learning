import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def kpp_init(X, k, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.integers(0, n)]
    dist_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = dist_sq / dist_sq.sum()
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]
        new_dist = np.sum((X - centers[i]) ** 2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist)
    return centers


def kmeans_scratch(X, k, max_iters=300, tol=1e-4, seed=42):
    rng = np.random.default_rng(seed)
    centers = kpp_init(X, k, seed=seed)
    n = X.shape[0]
    labels = np.zeros(n, dtype=int)
    for it in range(max_iters):
        # assign
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        # update centers
        new_centers = np.zeros_like(centers)
        for j in range(k):
            pts = X[new_labels == j]
            if len(pts) == 0:
                new_centers[j] = X[rng.integers(0, n)]
            else:
                new_centers[j] = pts.mean(axis=0)
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        labels = new_labels
        if shift < tol:
            break
    wcss = np.sum((X - centers[labels]) ** 2)
    return labels, centers, wcss


def main(k_opt=3):
    cleaned = os.path.join(RESULTS_DIR, "cleaned_data.csv")
    df = pd.read_csv(cleaned)
    features = [
        c
        for c in ["price", "cost", "units_sold", "promotion_frequency", "shelf_level"]
        if c in df.columns
    ]
    X_raw = df[features].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler_clustering.joblib"))

    # elbow
    wcss = {}
    for k in range(2, 9):
        _, _, w = kmeans_scratch(X, k)
        wcss[k] = float(w)

    # elbow plot
    plt.figure(figsize=(6, 4))
    ks = sorted(wcss.keys())
    plt.plot(ks, [wcss[k] for k in ks], marker="o")
    plt.xlabel("k")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "elbow.png"), dpi=150)
    plt.close()

    # final clustering
    labels, centers, _ = kmeans_scratch(X, k_opt)
    df["cluster"] = labels
    df.to_csv(os.path.join(RESULTS_DIR, "product_with_clusters.csv"), index=False)

    cluster_summary = (
        df.groupby("cluster")
        .agg(
            n_products=("product_id", "count"),
            avg_price=("price", "mean"),
            avg_units_sold=("units_sold", "mean"),
            avg_profit=("profit", "mean"),
            avg_promotion_frequency=("promotion_frequency", "mean"),
            avg_cost=("cost", "mean"),
        )
        .reset_index()
    )
    cluster_summary.to_csv(
        os.path.join(RESULTS_DIR, "cluster_summary.csv"), index=False
    )
    print(
        "K-means finished. cluster_summary.csv and product_with_clusters.csv saved to results/"
    )


if __name__ == "__main__":
    main(k_opt=3)
