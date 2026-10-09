"""FinSight decision API.

Run:  uvicorn finsight.api.main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import math
from contextlib import asynccontextmanager
from typing import Literal

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from finsight.config import DATA_PROCESSED, DUCKDB_PATH
from finsight.features.build import load_raw
from finsight.models.common import load_json, load_model
from finsight.models.credit_scoring import CreditScorer, applicant_features
from finsight.pipeline import LIVE_ASOF

STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["db"] = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    STATE["raw"] = load_raw()
    STATE["panel"] = pd.read_parquet(DATA_PROCESSED / "monthly_panel.parquet")
    STATE["profile"] = pd.read_parquet(DATA_PROCESSED / "account_profile.parquet")
    STATE["scorer"] = CreditScorer(load_model("credit_model.joblib"))
    yield
    STATE["db"].close()


app = FastAPI(title="FinSight Decision API", version="0.1.0", lifespan=lifespan,
              description="Alternative credit scoring, early-warning and next-best-offer on the Berka bank dataset.")


def _records(df: pd.DataFrame) -> list[dict]:
    out = df.astype(object).where(df.notna(), None).to_dict("records")
    for r in out:
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
            elif isinstance(v, pd.Timestamp):
                r[k] = v.isoformat()
    return out


def _q(sql: str, params: list | None = None) -> pd.DataFrame:
    return STATE["db"].execute(sql, params or []).df()


class CreditRequest(BaseModel):
    account_id: int = Field(..., examples=[19])
    amount: float = Field(..., gt=0, examples=[120000])
    duration: Literal[12, 24, 36, 48, 60] = Field(..., examples=[36])


class Reason(BaseModel):
    feature: str
    label: str
    value: float | None
    impact: float
    direction: str


class CreditResponse(BaseModel):
    account_id: int
    pd: float
    score: int
    decision: str
    monthly_instalment: float
    reasons: list[Reason]


@app.get("/health")
def health():
    return {"status": "ok", "asof": LIVE_ASOF}


@app.get("/summary")
def summary():
    return load_json("summary.json")


@app.post("/credit/score", response_model=CreditResponse)
def credit_score(req: CreditRequest):
    if req.account_id not in set(STATE["profile"]["account_id"]):
        raise HTTPException(404, "unknown account_id")
    X = applicant_features(STATE["raw"], STATE["panel"], STATE["profile"], [req.account_id],
                           LIVE_ASOF, req.amount, req.duration)
    s = STATE["scorer"].score(X).iloc[0]
    return CreditResponse(account_id=req.account_id, pd=float(s["pd"]), score=int(s["score"]),
                          decision=s["decision"], monthly_instalment=req.amount / req.duration,
                          reasons=STATE["scorer"].reasons(X)[0])


@app.get("/customers/{account_id}")
def customer(account_id: int):
    c = _q("SELECT * FROM customer_360 WHERE account_id = ?", [account_id])
    if c.empty:
        raise HTTPException(404, "unknown account_id")
    offers = _q("SELECT product, propensity, expected_value, eligible, blocked_reason, rank "
                "FROM offers WHERE account_id = ? ORDER BY rank NULLS LAST", [account_id])
    return {"customer": _records(c)[0], "offers": _records(offers)}


@app.get("/ews/watchlist")
def watchlist(level: Literal["red", "amber", "all"] = "all", limit: int = Query(50, le=500)):
    where = "" if level == "all" else "WHERE alert_level = ?"
    params = [] if level == "all" else [level]
    df = _q(f"SELECT * FROM ews_watchlist {where} ORDER BY risk DESC LIMIT {int(limit)}", params)
    return _records(df)


@app.get("/segments")
def segments():
    return load_json("segmentation_report.json")


@app.get("/offers/campaign/{product}")
def campaign(product: Literal["credit_card", "loan", "insurance"], top: int = Query(100, le=2000)):
    """Campaign list: best eligible customers for one product, ranked by expected value."""
    df = _q("SELECT o.account_id, o.propensity, o.expected_value, c.segment "
            "FROM offers o JOIN customer_360 c USING (account_id) "
            "WHERE o.product = ? AND o.eligible ORDER BY o.expected_value DESC LIMIT ?", [product, top])
    return _records(df)
