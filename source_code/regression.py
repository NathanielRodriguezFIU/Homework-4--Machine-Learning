import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "product_with_clusters.csv"))
    features = [
        c
        for c in ["price", "cost", "units_sold", "promotion_frequency", "shelf_level"]
        if c in df.columns
    ]
    X = df[features]
    y = df["profit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    # Polynomial degree 2
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    lr_poly = LinearRegression()
    lr_poly.fit(X_train_poly, y_train)
    pred_poly = lr_poly.predict(X_test_poly)

    # RandomForest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    records = [
        {"model": "Linear", **metrics(y_test, pred_lr)},
        {"model": "Polynomial_deg2", **metrics(y_test, pred_poly)},
        {"model": "RandomForest", **metrics(y_test, pred_rf)},
    ]
    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "regression_metrics.csv"), index=False)
    print("Saved regression_metrics.csv")

    # Actual vs Predicted plot (RandomForest)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred_rf, alpha=0.7)
    lims = [min(min(y_test), min(pred_rf)), max(max(y_test), max(pred_rf))]
    plt.plot(lims, lims, "k--")
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.title("Actual vs Predicted (RandomForest)")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()
    print("Saved actual_vs_predicted.png")


if __name__ == "__main__":
    main()
