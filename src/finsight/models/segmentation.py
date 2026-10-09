"""Behavioural customer segmentation (customer-360 snapshot + K-Means)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from finsight.config import RANDOM_STATE
from finsight.features.build import (
    add_profile_features, card_events, month_start, window_features,
)
from finsight.models.common import save_json, save_model

PRODUCTS = ["credit_card", "loan", "insurance"]

SEGMENT_FEATURES = {
    "log_balance": "Average balance",
    "log_inflow": "Monthly inflow",
    "n_trans_mean_12m": "Transactions / month",
    "cash_share_12m": "Cash share of spend",
    "uses_card": "Card spending",
    "has_pension": "Pension income",
    "log_household": "Household bills",
    "age": "Age",
    "tenure_months": "Tenure",
    "n_products": "Products held",
}


def product_events(raw) -> pd.DataFrame:
    """First-acquisition date of each product per account."""
    cards = card_events(raw).set_index("account_id")["card_issued"].rename("credit_card")
    loans = raw["loan"].groupby("account_id")["date"].min().rename("loan")
    t = raw["trans"]
    ins = t[t["k_symbol"] == "insurance"].groupby("account_id")["date"].min().rename("insurance")
    return pd.concat([cards, loans, ins], axis=1)


def customer_snapshot(raw, panel, profile, asof) -> pd.DataFrame:
    """Customer-360 row for every account open at least 3 months before ``asof``."""
    asof = pd.Timestamp(asof)
    asof_m = asof.to_period("M").to_timestamp()
    acc = profile[profile["open_date"] < asof_m - pd.DateOffset(months=3)][["account_id"]].copy()
    acc["key"], acc["asof_month"], acc["asof"] = acc["account_id"], asof_m, asof
    df = acc.merge(window_features(panel, acc[["key", "account_id", "asof_month"]]), on="key", how="left")
    df = add_profile_features(df, profile, "asof")

    ev = product_events(raw).reindex(df["account_id"])
    for p in PRODUCTS:
        df[f"owns_{p}"] = (ev[p].values < asof).astype(int)
    df["n_products"] = df[[f"owns_{p}" for p in PRODUCTS]].sum(axis=1)
    df["log_balance"] = np.log1p(df["bal_mean_12m"].clip(lower=0))
    df["log_inflow"] = np.log1p(df["inflow_mean_12m"].clip(lower=0))
    df["log_household"] = np.log1p(df["household_mean_12m"].clip(lower=0))
    df["uses_card"] = (df["card_wd_12m"] > 0).astype(int)
    return df.drop(columns=["key", "asof_month"])


def _name_segment(z: pd.Series) -> str:
    """Readable name from the two most distinctive centroid features."""
    words = {
        ("log_balance", 1): "High-balance", ("log_balance", -1): "Low-balance",
        ("log_inflow", 1): "High-income", ("log_inflow", -1): "Low-income",
        ("n_trans_mean_12m", 1): "Active transactors", ("n_trans_mean_12m", -1): "Dormant",
        ("cash_share_12m", 1): "Cash-heavy", ("cash_share_12m", -1): "Digital/transfer",
        ("uses_card", 1): "Card spenders", ("uses_card", -1): "No card use",
        ("has_pension", 1): "Pensioners", ("has_pension", -1): "Working-age",
        ("log_household", 1): "Bill payers", ("log_household", -1): "No bills",
        ("age", 1): "Senior", ("age", -1): "Young",
        ("tenure_months", 1): "Long-tenure", ("tenure_months", -1): "New-to-bank",
        ("n_products", 1): "Multi-product", ("n_products", -1): "Single-product",
    }
    top = z.abs().sort_values(ascending=False).index[:2]
    return " · ".join(words[(f, 1 if z[f] > 0 else -1)] for f in top)


def train(snapshot: pd.DataFrame, k_range=range(4, 8)) -> dict:
    cols = list(SEGMENT_FEATURES)
    X = snapshot[cols].fillna(snapshot[cols].median())
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    scores = {}
    for k in k_range:
        labels = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z)
        scores[k] = float(silhouette_score(Z, labels, sample_size=3000, random_state=RANDOM_STATE))
    k = max(scores, key=scores.get)
    km = KMeans(k, n_init=20, random_state=RANDOM_STATE).fit(Z)
    seg = snapshot[["account_id"]].copy()
    seg["segment_id"] = km.labels_
    centroids = pd.DataFrame(km.cluster_centers_, columns=cols)
    names = {i: _name_segment(centroids.loc[i]) for i in range(k)}
    # Disambiguate identical names.
    seen: dict[str, int] = {}
    for i, n in names.items():
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[i] = f"{n} ({seen[n]})"
    seg["segment"] = seg["segment_id"].map(names)
    pcs = PCA(2, random_state=RANDOM_STATE).fit(Z)
    xy = pcs.transform(Z)
    seg["pc1"], seg["pc2"] = xy[:, 0], xy[:, 1]

    prof = snapshot.assign(segment=seg["segment"].values).groupby("segment").agg(
        customers=("account_id", "size"),
        avg_balance=("bal_mean_12m", "mean"), avg_inflow=("inflow_mean_12m", "mean"),
        avg_age=("age", "mean"), txn_per_month=("n_trans_mean_12m", "mean"),
        cash_share=("cash_share_12m", "mean"), pension_share=("has_pension", "mean"),
        card_share=("owns_credit_card", "mean"), loan_share=("owns_loan", "mean"),
        insurance_share=("owns_insurance", "mean"), avg_products=("n_products", "mean"),
    ).round(3).reset_index()

    save_model({"scaler": scaler, "kmeans": km, "features": cols, "names": names, "pca": pcs},
               "segmentation.joblib")
    report = {"k": k, "silhouette_by_k": scores, "profiles": prof.to_dict("records")}
    save_json(report, "segmentation_report.json")
    return {"report": report, "segments": seg}
