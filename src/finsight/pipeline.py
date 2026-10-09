"""End-to-end pipeline: ingest -> features -> 4 models -> DuckDB warehouse -> KPI summary.

Usage:  python -m finsight.pipeline
"""
from __future__ import annotations

import time
import warnings

import duckdb
import numpy as np
import pandas as pd

from finsight.config import DATA_PROCESSED, DATA_RAW, DUCKDB_PATH, SNAPSHOT_DATE
from finsight.data.ingest import ingest
from finsight.features.build import account_profile, load_raw, monthly_panel
from finsight.models import credit_scoring, ews, nbo, segmentation
from finsight.models.common import load_model, save_json

warnings.filterwarnings("ignore", category=UserWarning)

LIVE_ASOF = (pd.Timestamp(SNAPSHOT_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # 1999-01-01
# Standard pre-approval offer used to gate credit products: median loan in the book.
PREAPPROVAL_AMOUNT, PREAPPROVAL_DURATION = 120_000, 36


def _step(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def run() -> dict:
    t0 = time.time()
    if not (DATA_RAW / "trans.parquet").exists():
        ingest()
    raw = load_raw()
    panel = monthly_panel(raw)
    profile = account_profile(raw)
    panel.to_parquet(DATA_PROCESSED / "monthly_panel.parquet", index=False)
    profile.to_parquet(DATA_PROCESSED / "account_profile.parquet", index=False)
    _step(f"features: panel {panel.shape}", t0)

    credit = credit_scoring.train(raw, panel, profile)
    _step(f"credit scoring: champion={credit['report']['champion']} "
          f"AUC={credit['report']['metrics'][credit['report']['champion']]['auc']}", t0)

    ews_res = ews.train(raw, panel, profile, credit["scored"])
    watch = ews.current_watchlist(raw, panel, profile, credit["scored"])
    _step(f"EWS: red precision={ews_res['report']['red_alert']['precision']:.2f} "
          f"recall={ews_res['report']['red_alert']['recall']:.2f}", t0)

    snap_t0 = segmentation.customer_snapshot(raw, panel, profile, nbo.T0)
    nbo_res = nbo.train(snap_t0, raw)
    _step(f"NBO: campaign uplift vs heuristic="
          f"{nbo_res['report']['summary']['campaign_uplift_vs_heuristic']:.1%}", t0)

    live = segmentation.customer_snapshot(raw, panel, profile, LIVE_ASOF)
    seg = segmentation.train(live)
    _step(f"segmentation: k={seg['report']['k']}", t0)

    # Credit pre-approval for every live customer, used to gate credit offers.
    scorer = credit_scoring.CreditScorer(load_model("credit_model.joblib"))
    appl = credit_scoring.applicant_features(raw, panel, profile, live["account_id"], LIVE_ASOF,
                                             PREAPPROVAL_AMOUNT, PREAPPROVAL_DURATION)
    pre = scorer.score(appl)
    pre["account_id"] = appl["account_id"].values
    preapproved = pre.set_index("account_id")["decision"].eq("auto_approve")
    flagged = set(watch.loc[watch["alert_level"] != "green", "account_id"])
    offers = nbo.recommend(live, load_model("nbo_models.joblib"), preapproved, flagged)
    _step(f"offers: {int(offers['eligible'].sum()):,} eligible offers for {live['account_id'].nunique():,} customers", t0)

    customer_360 = (live.merge(seg["segments"], on="account_id")
                    .merge(pre[["account_id", "pd", "score", "decision"]]
                           .rename(columns={"pd": "preapproval_pd", "score": "preapproval_score",
                                            "decision": "preapproval_decision"}), on="account_id"))
    best = offers[offers["rank"] == 1][["account_id", "product", "propensity", "expected_value"]]
    customer_360 = customer_360.merge(
        best.rename(columns={"product": "next_best_offer", "propensity": "nbo_propensity",
                             "expected_value": "nbo_expected_value"}), on="account_id", how="left")
    customer_360 = customer_360.merge(
        watch[["account_id", "risk", "alert_level"]].rename(columns={"risk": "ews_risk"}),
        on="account_id", how="left")

    write_warehouse({
        "raw_account": raw["account"], "raw_client": raw["client"], "raw_disp": raw["disp"],
        "raw_district": raw["district"], "raw_loan": raw["loan"], "raw_card": raw["card"],
        "raw_order": raw["order"], "raw_trans": raw["trans"],
        "monthly_panel": panel, "customer_360": customer_360,
        "loan_scores": credit["scored"], "ews_watchlist": watch,
        "ews_backtest": ews_res["panel"][["loan_id", "account_id", "asof_month", "y", "p_oof",
                                          "onset_date", "status"]],
        "offers": offers, "segments": seg["segments"],
    })
    summary = build_summary(credit["report"], ews_res["report"], nbo_res["report"], seg["report"],
                            customer_360, offers, watch)
    save_json(summary, "summary.json")
    _step(f"warehouse written to {DUCKDB_PATH.name}; done", t0)
    return summary


def write_warehouse(tables: dict[str, pd.DataFrame]) -> None:
    con = duckdb.connect(str(DUCKDB_PATH))
    for name, df in tables.items():
        con.register("_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
    con.close()


def build_summary(cr, ew, nb, sg, c360, offers, watch) -> dict:
    champ = cr["champion"]
    return {
        "credit_scoring": {
            "auc": cr["metrics"][champ]["auc"], "gini": cr["metrics"][champ]["gini"],
            "auc_traditional": cr["metrics"]["traditional_logit"]["auc"],
            "auc_out_of_time": cr["out_of_time"][champ]["auc"],
            "auc_thin_file": (cr["segment_metrics"]["thin_file_lt6m"] or {}).get("auc"),
            "npa_reduction_at_85pct_approval": cr["npa_reduction_at_85pct_approval"],
            "straight_through_rate": cr["decision_bands"]["straight_through_rate"],
            "target": "15% NPA reduction, faster approvals",
        },
        "early_warning": {
            "row_auc": ew["metrics_row_level"]["auc"],
            "red_precision": ew["red_alert"]["precision"], "red_recall": ew["red_alert"]["recall"],
            "red_median_lead_days": ew["red_alert"]["median_lead_days"],
            "amber_precision": ew["amber_alert"]["precision"], "amber_recall": ew["amber_alert"]["recall"],
            "live_red": int((watch["alert_level"] == "red").sum()),
            "live_amber": int((watch["alert_level"] == "amber").sum()),
            "target": "flag 30-60 days ahead with 80%+ precision",
        },
        "next_best_offer": {
            **nb["summary"],
            "per_product_auc": {p: v["auc"] for p, v in nb["per_product"].items()},
            "customers_with_offer": int(offers.loc[offers["rank"] == 1, "account_id"].nunique()),
            "target": "+20% cross-sell conversion",
        },
        "segmentation": {"k": sg["k"], "silhouette": sg["silhouette_by_k"][sg["k"]],
                         "segments": {p["segment"]: p["customers"] for p in sg["profiles"]}},
        "customers": int(len(c360)),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, default=str))
