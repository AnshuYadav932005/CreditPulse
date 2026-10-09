"""AI-powered alternative credit scoring for thin-file applicants.

Every borrower in the Berka data is thin-file: there is no credit bureau, and the median
applicant has ~13 months of banking history. We score them from behaviour and transactions
observed strictly BEFORE the loan date, and compare against a "traditional" model that only
sees demographics + loan terms, i.e. what a lender without transaction data would have.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from finsight.config import RANDOM_STATE
from finsight.features.build import (
    account_profile, add_profile_features, card_events, month_start, window_features,
)
from finsight.models.common import binary_metrics, lgbm_model, logit_model, save_json, save_model

TRADITIONAL = ["amount", "duration", "payments", "age", "is_male", "has_disponent",
               "urban_ratio", "avg_salary", "unemployment_96", "entrepreneurs_per_1k", "crime_rate"]

# Scorecard scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds.
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20

FEATURE_LABELS = {
    "min_bal_cover": "Lowest balance vs. instalment",
    "bal_cover": "Average balance vs. instalment",
    "pti": "Instalment as share of monthly inflow",
    "bal_min_12m": "Lowest balance (12m)",
    "bal_min_6m": "Lowest balance (6m)",
    "bal_min_3m": "Lowest balance (3m)",
    "bal_mean_12m": "Average balance (12m)",
    "bal_mean_3m": "Average balance (3m)",
    "bal_end_last": "Balance at application",
    "low_bal_months_12m": "Months with balance < 5k",
    "inflow_mean_12m": "Average monthly inflow",
    "inflow_cv_12m": "Income volatility",
    "cash_share_12m": "Share of spend as cash",
    "netflow_mean_12m": "Average monthly net flow",
    "payments": "Monthly instalment",
    "amount": "Loan amount",
    "duration": "Loan duration",
    "history_months": "Months of banking history",
    "age": "Applicant age",
}


def build_dataset(raw, panel, profile=None) -> pd.DataFrame:
    profile = account_profile(raw) if profile is None else profile
    loans = raw["loan"].copy()
    loans["bad"] = loans["status"].isin(["B", "D"]).astype(int)
    loans["asof_month"] = month_start(loans["date"])
    keys = loans[["loan_id", "account_id", "asof_month"]].rename(columns={"loan_id": "key"})
    feats = window_features(panel, keys).rename(columns={"key": "loan_id"})
    df = loans.merge(feats, on="loan_id", how="left")
    df = add_profile_features(df, profile, "date")
    cards = card_events(raw)
    df = df.merge(cards[["account_id", "card_issued"]], on="account_id", how="left")
    df["had_card"] = (df["card_issued"] < df["date"]).astype(int)
    df = df.drop(columns=["card_issued"])
    return add_affordability(df)


def applicant_features(raw, panel, profile, account_ids, asof, amount: float, duration: int) -> pd.DataFrame:
    """Features for a hypothetical new loan application by existing customers at ``asof``.

    In this bank the monthly instalment is amount / duration (verified on the loan table).
    """
    asof = pd.Timestamp(asof)
    df = pd.DataFrame({"loan_id": np.arange(len(account_ids)), "account_id": list(account_ids)})
    df["date"], df["asof_month"] = asof, asof.to_period("M").to_timestamp()
    df["amount"], df["duration"] = float(amount), int(duration)
    df["payments"] = float(amount) / int(duration)
    keys = df[["loan_id", "account_id", "asof_month"]].rename(columns={"loan_id": "key"})
    feats = window_features(panel, keys).rename(columns={"key": "loan_id"})
    df = df.merge(feats, on="loan_id", how="left")
    df = add_profile_features(df, profile, "date")
    cards = card_events(raw)
    df = df.merge(cards[["account_id", "card_issued"]], on="account_id", how="left")
    df["had_card"] = (df["card_issued"] < df["date"]).astype(int)
    return add_affordability(df.drop(columns=["card_issued"]))


def add_affordability(df: pd.DataFrame) -> pd.DataFrame:
    pay = df["payments"].replace(0, np.nan)
    df["pti"] = df["payments"] / df["inflow_mean_12m"].replace(0, np.nan)
    df["bal_cover"] = df["bal_mean_12m"] / pay
    df["min_bal_cover"] = df["bal_min_12m"] / pay
    df["thin_file"] = (df["history_months"] < 6).astype(int)
    return df


def behavioural_features(df: pd.DataFrame) -> list[str]:
    drop = {"loan_id", "account_id", "date", "status", "bad", "asof_month", "thin_file",
            "penalties_3m", "penalties_6m", "penalties_12m"}  # penalties are ~never pre-loan
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def oof_predict(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> np.ndarray:
    """Out-of-fold PD averaged over repeated stratified 5-fold CV."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_STATE)
    preds, counts = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        preds[te] += m.predict_proba(X.iloc[te])[:, 1]
        counts[te] += 1
    return preds / counts


def pd_to_score(pd_: np.ndarray, base_rate: float = None) -> np.ndarray:
    pd_ = np.clip(pd_, 1e-4, 1 - 1e-4)
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    return np.round(offset + factor * np.log((1 - pd_) / pd_)).astype(int)


def approval_curve(y, p_model, p_base, rates=np.arange(0.5, 1.0001, 0.05)) -> list[dict]:
    """NPA rate of the approved book when approving the lowest-risk X% under each model."""
    y = np.asarray(y)
    rows = []
    for r in rates:
        k = int(round(len(y) * r))
        npa_m = y[np.argsort(p_model)[:k]].mean()
        npa_b = y[np.argsort(p_base)[:k]].mean()
        rows.append({"approval_rate": round(float(r), 2), "npa_model": float(npa_m),
                     "npa_traditional": float(npa_b), "npa_approve_all": float(y.mean()),
                     "npa_reduction_vs_traditional": float(1 - npa_m / npa_b) if npa_b else 0.0})
    return rows


def decision_bands(y, p, approve_below: float, decline_above: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    auto_ok, decline = p < approve_below, p >= decline_above
    review = ~(auto_ok | decline)
    return {
        "approve_below_pd": approve_below, "decline_above_pd": decline_above,
        "auto_approve_share": float(auto_ok.mean()), "auto_decline_share": float(decline.mean()),
        "manual_review_share": float(review.mean()),
        "straight_through_rate": float((auto_ok | decline).mean()),
        "npa_auto_approved": float(y[auto_ok].mean()) if auto_ok.any() else None,
        "bad_rate_auto_declined": float(y[decline].mean()) if decline.any() else None,
        "bads_caught_by_decline": float(y[decline].sum() / max(1, y.sum())),
    }


def train(raw, panel, profile=None) -> dict:
    df = build_dataset(raw, panel, profile)
    y = df["bad"]
    feats = behavioural_features(df)
    X, Xt = df[feats], df[TRADITIONAL]

    candidates = {
        "traditional_logit": (logit_model(), Xt),
        "behavioural_logit": (logit_model(C=0.1), X),
        "behavioural_lgbm": (lgbm_model(), X),
    }
    oof = {name: oof_predict(m, data, y) for name, (m, data) in candidates.items()}
    metrics = {k: binary_metrics(y, v) for k, v in oof.items()}
    champion = max(["behavioural_logit", "behavioural_lgbm"], key=lambda k: metrics[k]["auc"])
    p_champ = oof[champion]

    # Out-of-time check: train on the earliest 70% of loans, test on the latest 30%.
    cutoff = df["date"].quantile(0.7)
    early, late = df["date"] < cutoff, df["date"] >= cutoff
    oot = {}
    for name, (m, data) in candidates.items():
        fitted = clone(m).fit(data[early], y[early])
        oot[name] = binary_metrics(y[late], fitted.predict_proba(data[late])[:, 1])
    oot["cutoff_date"] = str(cutoff.date())

    thin = df["thin_file"].astype(bool)
    segment = {
        "thin_file_lt6m": binary_metrics(y[thin], p_champ[thin]) if y[thin].nunique() > 1 else None,
        "history_6m_plus": binary_metrics(y[~thin], p_champ[~thin]),
    }
    curve = approval_curve(y, p_champ, oof["traditional_logit"])
    at85 = next(r for r in curve if r["approval_rate"] == 0.85)

    # Decision bands chosen on OOF PDs: approve the safest ~60%, decline the riskiest ~10%.
    lo, hi = np.quantile(p_champ, [0.60, 0.90])
    bands = decision_bands(y, p_champ, float(lo), float(hi))

    # Final models on all data.
    final = clone(candidates[champion][0]).fit(X, y)
    lgbm_full = lgbm_model().fit(X, y)  # always kept for SHAP reason codes
    explainer = shap.TreeExplainer(lgbm_full)
    sv = explainer.shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    importance = (pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
                  .head(15).round(4).to_dict())

    save_model({"model": final, "features": feats, "champion": champion,
                "lgbm": lgbm_full, "bands": bands}, "credit_model.joblib")
    scored = df[["loan_id", "account_id", "date", "amount", "duration", "payments", "status",
                 "bad", "thin_file", "history_months"]].copy()
    scored["pd_oof"] = p_champ
    scored["score"] = pd_to_score(p_champ)
    scored["decision"] = np.select([p_champ < lo, p_champ >= hi], ["auto_approve", "decline"],
                                   "manual_review")

    report = {
        "dataset": {"n_loans": len(df), "n_bad": int(y.sum()), "bad_rate": float(y.mean()),
                    "median_history_months": float(df["history_months"].median()),
                    "thin_file_share_lt6m": float(thin.mean()), "n_features": len(feats)},
        "cv": "5-fold stratified x5 repeats, out-of-fold",
        "metrics": metrics, "out_of_time": oot, "champion": champion, "segment_metrics": segment,
        "approval_curve": curve, "npa_reduction_at_85pct_approval": at85["npa_reduction_vs_traditional"],
        "decision_bands": bands, "top_features": importance,
    }
    save_json(report, "credit_report.json")
    return {"report": report, "scored": scored, "data": df}


class CreditScorer:
    """Loads the trained bundle and scores/explains new applications."""

    def __init__(self, bundle: dict):
        self.model, self.features = bundle["model"], bundle["features"]
        self.explainer = shap.TreeExplainer(bundle["lgbm"])
        self.bands = bundle["bands"]

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.features)
        p = self.model.predict_proba(X)[:, 1]
        out = pd.DataFrame({"pd": p, "score": pd_to_score(p)}, index=X.index)
        out["decision"] = np.select(
            [p < self.bands["approve_below_pd"], p >= self.bands["decline_above_pd"]],
            ["auto_approve", "decline"], "manual_review")
        return out

    def reasons(self, X: pd.DataFrame, top: int = 4) -> list[list[dict]]:
        X = X.reindex(columns=self.features)
        sv = self.explainer.shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        result = []
        for i in range(len(X)):
            order = np.argsort(-np.abs(sv[i]))[:top]
            result.append([{
                "feature": self.features[j],
                "label": FEATURE_LABELS.get(self.features[j], self.features[j]),
                "value": None if pd.isna(X.iloc[i, j]) else float(X.iloc[i, j]),
                "impact": float(sv[i, j]),
                "direction": "increases risk" if sv[i, j] > 0 else "reduces risk",
            } for j in order])
        return result
