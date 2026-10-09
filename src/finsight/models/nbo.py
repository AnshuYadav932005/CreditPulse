"""Next-Best-Offer engine: per-product propensity models + value/eligibility ranking.

Training is point-in-time: features at T0 = 1998-01-01, label = customer acquired the product
during 1998 (and did not hold it at T0). The models then score the live snapshot (1999-01-01).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from finsight.config import RANDOM_STATE
from finsight.models.common import binary_metrics, lgbm_model, lift_at, save_json, save_model
from finsight.models.segmentation import PRODUCTS, SEGMENT_FEATURES, product_events

T0 = "1998-01-01"
OUTCOME_END = "1999-01-01"

# Illustrative 12-month margin per product (relative units); tune with finance.
PRODUCT_VALUE = {"credit_card": 1.0, "loan": 3.0, "insurance": 1.5}
PRODUCT_LABEL = {"credit_card": "Credit card", "loan": "Personal loan", "insurance": "Insurance"}
TARGET_SHARE = 0.20  # campaign capacity: contact the top 20% of eligible customers

# Tenure is excluded on purpose: in this bank standing-order products (insurance, household
# bills) are set up ~5 months after opening, so tenure would let the model learn the
# onboarding clock rather than genuine propensity.
EXCLUDE = {"account_id", "asof", "has_insurance_pmt", "has_loan_pmt", "tenure_months",
           "history_months", "penalties_3m", "penalties_6m", "penalties_12m"}
MIN_ADOPTERS = 50  # below this, a product's uplift is not statistically meaningful


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE and not c.startswith("y_")
            and pd.api.types.is_numeric_dtype(df[c])]


def label(snapshot: pd.DataFrame, raw) -> pd.DataFrame:
    ev = product_events(raw).reindex(snapshot["account_id"])
    t0, t1 = pd.Timestamp(T0), pd.Timestamp(OUTCOME_END)
    for p in PRODUCTS:
        d = ev[p].values
        snapshot[f"y_{p}"] = ((d >= t0) & (d < t1)).astype(int)
    return snapshot


def heuristic_score(df: pd.DataFrame, product: str) -> pd.Series:
    """What a campaign team without ML might do: target the highest-balance customers."""
    return df["bal_mean_12m"].fillna(0)


def train(train_snapshot: pd.DataFrame, raw) -> dict:
    df = label(train_snapshot.copy(), raw)
    feats = feature_columns(df)
    models, metrics = {}, {}
    for p in PRODUCTS:
        elig = df[df[f"owns_{p}"] == 0]
        y = elig[f"y_{p}"]
        tr, te = train_test_split(elig.index, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
        m = lgbm_model(n_estimators=250, num_leaves=15, min_child_samples=25)
        m.fit(elig.loc[tr, feats], y[tr])
        p_te = m.predict_proba(elig.loc[te, feats])[:, 1]
        model_lift = lift_at(y[te], p_te, TARGET_SHARE)
        heur_lift = lift_at(y[te], heuristic_score(elig.loc[te], p).values, TARGET_SHARE)
        metrics[p] = {
            **binary_metrics(y[te], p_te),
            "eligible_customers": int(len(elig)), "adopters": int(y.sum()),
            f"conversion_top{int(TARGET_SHARE * 100)}_model": model_lift["hit_rate"],
            f"conversion_top{int(TARGET_SHARE * 100)}_heuristic": heur_lift["hit_rate"],
            "conversion_random": model_lift["base_rate"],
            "lift_vs_random": model_lift["lift"],
            "uplift_vs_heuristic": model_lift["hit_rate"] / heur_lift["hit_rate"] - 1
            if heur_lift["hit_rate"] else None,
            "adopters_captured_top20": model_lift["captured"],
            "low_data": bool(y.sum() < MIN_ADOPTERS),
        }
        models[p] = clone(m).fit(elig[feats], y)  # refit on all eligible customers

    # Campaign-level: blended conversion when each product targets its top 20%.
    core = [p for p in PRODUCTS if not metrics[p]["low_data"]]
    tot_model = sum(metrics[p]["conversion_top20_model"] * metrics[p]["eligible_customers"] for p in core)
    tot_heur = sum(metrics[p]["conversion_top20_heuristic"] * metrics[p]["eligible_customers"] for p in core)
    tot_rand = sum(metrics[p]["conversion_random"] * metrics[p]["eligible_customers"] for p in core)
    summary = {
        "products_in_summary": core,
        "campaign_conversion_heuristic": tot_heur / sum(metrics[p]["eligible_customers"] for p in core),
        "campaign_conversion_random": tot_rand / sum(metrics[p]["eligible_customers"] for p in core),
        "campaign_conversion_model": tot_model / sum(metrics[p]["eligible_customers"] for p in core),
        "campaign_uplift_vs_heuristic": tot_model / tot_heur - 1,
        "campaign_uplift_vs_random": tot_model / tot_rand - 1,
    }
    save_model({"models": models, "features": feats}, "nbo_models.joblib")
    report = {"t0": T0, "outcome_window_end": OUTCOME_END, "target_share": TARGET_SHARE,
              "product_value": PRODUCT_VALUE, "per_product": metrics, "summary": summary}
    save_json(report, "nbo_report.json")
    return {"report": report}


def recommend(snapshot: pd.DataFrame, bundle: dict, loan_preapproved: pd.Series | None = None,
              ews_flagged: set | None = None) -> pd.DataFrame:
    """Rank offers per customer by expected value = propensity x margin, subject to eligibility.

    Eligibility rules: don't offer products already held; only offer credit (loan, card) to
    customers who pass credit pre-approval and are not on the EWS watchlist.
    """
    feats = bundle["features"]
    X = snapshot.reindex(columns=feats)
    rows = []
    for p in PRODUCTS:
        prop = bundle["models"][p].predict_proba(X)[:, 1]
        eligible = snapshot[f"owns_{p}"].values == 0
        reason = np.where(eligible, "", "already holds product")
        if p in ("loan", "credit_card"):
            if loan_preapproved is not None:
                ok = snapshot["account_id"].map(loan_preapproved).fillna(False).values.astype(bool)
                reason = np.where(eligible & ~ok, "fails credit pre-approval", reason)
                eligible &= ok
            if ews_flagged:
                flag = snapshot["account_id"].isin(ews_flagged).values
                reason = np.where(eligible & flag, "on early-warning watchlist", reason)
                eligible &= ~flag
        rows.append(pd.DataFrame({
            "account_id": snapshot["account_id"].values, "product": p,
            "propensity": prop, "expected_value": prop * PRODUCT_VALUE[p] * eligible,
            "eligible": eligible, "blocked_reason": reason,
        }))
    offers = pd.concat(rows, ignore_index=True)
    offers["rank"] = (offers[offers["eligible"]].groupby("account_id")["expected_value"]
                      .rank(ascending=False, method="first"))
    return offers.sort_values(["account_id", "rank"])
