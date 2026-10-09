"""Early Warning System for loan delinquency.

Delinquency onset = the first date after disbursement on which the account goes overdrawn or is
charged penalty interest (this covers ~95% of loans the bank ultimately marked B/D).

We observe every running loan at the start of each month m (using data up to the end of m-1)
and label it 1 if delinquency starts within the next 60 days (months m or m+1). The model is
evaluated with group-k-fold by loan, so no borrower appears in both train and test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from finsight.config import RANDOM_STATE, SNAPSHOT_DATE
from finsight.features.build import (
    account_profile, add_profile_features, month_start, months_between, window_features,
)
from finsight.models.common import binary_metrics, lgbm_model, save_json, save_model

MIN_LEAD_MONTHS = 1   # alert must fire at least one full month (~30 days) before onset
HORIZON_MONTHS = 2    # ...and at most MIN_LEAD + HORIZON months ahead
TARGET_PRECISION = 0.80
LEAKY = ("penalties_", "neg_months_")  # identical to the label by construction pre-onset


def delinquency_onsets(raw) -> pd.Series:
    t = raw["trans"][["account_id", "date", "balance", "k_symbol"]]
    loans = raw["loan"][["loan_id", "account_id", "date"]]
    m = t.merge(loans, on="account_id", suffixes=("", "_loan"))
    m = m[(m["date"] > m["date_loan"]) & ((m["balance"] < 0) | (m["k_symbol"] == "penalty_interest"))]
    return m.groupby("loan_id")["date"].min().rename("onset_date")


def build_panel(raw, panel, profile=None, credit_oof: pd.DataFrame | None = None,
                horizon: int = HORIZON_MONTHS, min_lead: int = MIN_LEAD_MONTHS) -> pd.DataFrame:
    profile = account_profile(raw) if profile is None else profile
    loans = raw["loan"].copy()
    loans = loans.join(delinquency_onsets(raw), on="loan_id")
    loans["loan_month"] = month_start(loans["date"])
    loans["onset_month"] = month_start(loans["onset_date"])
    # Bad loans with no observable delinquency event cannot be timed; leave them out.
    loans = loans[~(loans["status"].isin(["B", "D"]) & loans["onset_date"].isna())]

    last_obs = pd.Timestamp(SNAPSHOT_DATE).to_period("M").to_timestamp() - pd.DateOffset(months=min_lead + horizon - 1)
    rows = []
    for r in loans.itertuples():
        end_loan = r.loan_month + pd.DateOffset(months=int(r.duration))
        stop = min(last_obs, end_loan)
        if pd.notna(r.onset_month):
            stop = min(stop, r.onset_month - pd.DateOffset(months=min_lead))
        months = pd.date_range(r.loan_month + pd.DateOffset(months=2), stop, freq="MS")
        if len(months):
            rows.append(pd.DataFrame({"loan_id": r.loan_id, "account_id": r.account_id, "asof_month": months}))
    obs = pd.concat(rows, ignore_index=True)
    obs = obs.merge(loans[["loan_id", "date", "amount", "duration", "payments", "status",
                           "onset_date", "onset_month", "loan_month"]], on="loan_id")
    ahead = months_between(obs["onset_month"], obs["asof_month"])
    obs["y"] = ((ahead >= min_lead) & (ahead < min_lead + horizon)).astype(int)
    obs["key"] = np.arange(len(obs))

    return monitoring_frame(panel, profile, obs, credit_oof)


def monitoring_frame(panel, profile, obs: pd.DataFrame, credit_oof=None) -> pd.DataFrame:
    """Features for loan observations (needs key, loan_id, account_id, asof_month, loan terms)."""
    feats = window_features(panel, obs[["key", "account_id", "asof_month"]])
    df = obs.merge(feats, on="key", how="left")
    df = df.merge(instalment_stress(panel, obs), on="key", how="left")
    df = add_profile_features(df, profile, "asof_month")
    df = add_monitoring_features(df)
    if credit_oof is not None:
        df = df.merge(credit_oof[["loan_id", "pd_oof"]].rename(columns={"pd_oof": "origination_pd"}),
                      on="loan_id", how="left")
    return df


def instalment_stress(panel: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """How close the account came to missing an instalment in the last 6 months."""
    m = obs[["key", "account_id", "asof_month", "payments"]].merge(panel, on="account_id")
    m["lag"] = months_between(m["asof_month"], m["month"])
    m = m[(m["lag"] >= 1) & (m["lag"] <= 6)]
    m["short"] = (m["bal_min"] < m["payments"]).astype(int)
    m["tight"] = (m["bal_min"] < 2 * m["payments"]).astype(int)
    g = m.groupby("key")
    out = pd.DataFrame({
        "shortfall_months_6m": g["short"].sum(),
        "tight_months_6m": g["tight"].sum(),
        "bal_min_1m": m[m["lag"] == 1].set_index("key")["bal_min"],
        "inflow_1m": m[m["lag"] == 1].set_index("key")["inflow"],
        "outflow_1m": m[m["lag"] == 1].set_index("key")["outflow"],
    })
    # Slope of the monthly minimum balance (per month), negative = deteriorating.
    m["x"] = -m["lag"].astype(float)
    cov = g.apply(lambda d: np.polyfit(d["x"], d["bal_min"], 1)[0] if len(d) > 2 else np.nan,
                  include_groups=False)
    out["bal_min_slope_6m"] = cov
    return out.reset_index().rename(columns={"index": "key"})


def add_monitoring_features(df: pd.DataFrame) -> pd.DataFrame:
    pay = df["payments"].replace(0, np.nan)
    df["months_on_book"] = months_between(df["asof_month"], df["loan_month"])
    df["months_remaining"] = df["duration"] - df["months_on_book"]
    df["buffer_last"] = df["bal_end_last"] / pay          # months of instalment in the account
    df["buffer_min_3m"] = df["bal_min_3m"] / pay
    df["buffer_min_6m"] = df["bal_min_6m"] / pay
    df["pti_3m"] = df["payments"] / df["inflow_mean_3m"].replace(0, np.nan)
    df["netflow_after_pmt_3m"] = df["netflow_mean_3m"] / pay
    df["buffer_min_1m"] = df["bal_min_1m"] / pay
    df["bal_min_slope_vs_pmt"] = df["bal_min_slope_6m"] / pay
    df["spend_ratio_1m"] = df["outflow_1m"] / df["inflow_1m"].replace(0, np.nan)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"loan_id", "account_id", "asof_month", "date", "status", "onset_date", "onset_month",
            "loan_month", "y", "key"}
    return [c for c in df.columns if c not in drop and not c.startswith(LEAKY)
            and pd.api.types.is_numeric_dtype(df[c])]


def borrower_alerts(df: pd.DataFrame, p: np.ndarray, threshold: float) -> dict:
    """Borrower-level view: a flagged borrower is a hit if delinquency starts >= 30 days later."""
    d = df.assign(p=p)
    first = (d[d["p"] >= threshold].groupby("loan_id")
             .agg(alert=("asof_month", "min"), onset=("onset_date", "first")))
    lead = (first["onset"] - first["alert"]).dt.days
    hits = int((lead >= 30).sum())
    n_events = int(d.loc[d["onset_date"].notna(), "loan_id"].nunique())
    return {
        "threshold": float(threshold),
        "borrowers_flagged": int(len(first)),
        "true_flags": hits,
        "precision": hits / len(first) if len(first) else 0.0,
        "recall": hits / n_events if n_events else 0.0,
        "delinquent_borrowers": n_events,
        "median_lead_days": float(lead[lead >= 30].median()) if hits else None,
        "lead_days_p25": float(lead[lead >= 30].quantile(0.25)) if hits else None,
        "alerts_per_month_per_1k_loans": float((d["p"] >= threshold).sum()
                                               / d["asof_month"].nunique() / d["loan_id"].nunique() * 1000),
    }


def pick_threshold(df, p, min_precision: float, min_flags: int = 10) -> dict:
    """Highest-recall threshold whose borrower-level precision clears ``min_precision``."""
    best = None
    cands = np.unique(p[p >= np.quantile(p, 0.90)])
    if len(cands) > 800:
        cands = cands[np.linspace(0, len(cands) - 1, 800).astype(int)]
    for thr in cands:
        r = borrower_alerts(df, p, thr)
        if r["borrowers_flagged"] >= min_flags and r["precision"] >= min_precision:
            if best is None or r["recall"] > best["recall"]:
                best = r
    return best or borrower_alerts(df, p, float(np.quantile(p, 0.999)))


def train(raw, panel, profile=None, credit_oof=None) -> dict:
    df = build_panel(raw, panel, profile, credit_oof)
    feats = feature_columns(df)
    X, y, groups = df[feats], df["y"], df["loan_id"]

    model = lgbm_model(n_estimators=400, num_leaves=15, max_depth=4, min_child_samples=30)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        oof[te] = clone(model).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]

    red = pick_threshold(df, oof, TARGET_PRECISION)
    amber = pick_threshold(df, oof, 0.50)

    final = clone(model).fit(X, y)
    imp = pd.Series(final.booster_.feature_importance("gain"), index=feats)
    imp = (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict()
    save_model({"model": final, "features": feats, "threshold": red["threshold"],
                "watch_threshold": amber["threshold"]}, "ews_model.joblib")

    report = {
        "panel": {"rows": len(df), "loans": int(df["loan_id"].nunique()), "positives": int(y.sum()),
                  "positive_rate": float(y.mean()),
                  "label": f"delinquency onset {MIN_LEAD_MONTHS * 30}-{(MIN_LEAD_MONTHS + HORIZON_MONTHS) * 30} days ahead"},
        "cv": "StratifiedGroupKFold(5) grouped by loan, out-of-fold",
        "metrics_row_level": binary_metrics(y, oof),
        "red_alert": red,
        "amber_alert": amber,
        "note": f"Thresholds are tuned on out-of-fold scores over {red['delinquent_borrowers']} "
                "delinquent borrowers; expect roughly +/-10pp variation in precision on new data.",
        "top_features": imp,
    }
    save_json(report, "ews_report.json")
    return {"report": report, "panel": df.assign(p_oof=oof)}


def current_watchlist(raw, panel, profile, credit_oof=None) -> pd.DataFrame:
    """Score every running, not-yet-delinquent loan as of the month after the snapshot."""
    from finsight.models.common import load_model
    bundle = load_model("ews_model.joblib")
    loans = raw["loan"].copy()
    loans = loans.join(delinquency_onsets(raw), on="loan_id")
    loans = loans[(loans["status"] == "C") & loans["onset_date"].isna()].copy()
    loans["loan_month"] = month_start(loans["date"])
    loans["asof_month"] = pd.Timestamp(SNAPSHOT_DATE).to_period("M").to_timestamp() + pd.DateOffset(months=1)
    loans["key"] = loans["loan_id"]
    df = monitoring_frame(panel, profile, loans, credit_oof)
    df["risk"] = bundle["model"].predict_proba(df.reindex(columns=bundle["features"]))[:, 1]
    df["alert_level"] = np.select([df["risk"] >= bundle["threshold"], df["risk"] >= bundle["watch_threshold"]],
                                  ["red", "amber"], "green")
    cols = ["loan_id", "account_id", "date", "amount", "payments", "months_on_book", "months_remaining",
            "bal_end_last", "buffer_last", "buffer_min_3m", "inflow_trend", "pti_3m", "risk", "alert_level"]
    return df[cols].sort_values("risk", ascending=False)
