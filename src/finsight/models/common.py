"""Shared model utilities: metrics, persistence, model factories."""
from __future__ import annotations

import json
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from finsight.config import ARTIFACTS, RANDOM_STATE


def ks_stat(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def binary_metrics(y, p) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    return {
        "auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_stat(y, p), 4),
        "avg_precision": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "base_rate": round(float(y.mean()), 4),
        "n": int(len(y)),
    }


def logit_model(C: float = 0.3):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
    )


def lgbm_model(**overrides):
    params = dict(
        n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3,
        min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        reg_lambda=2.0, random_state=RANDOM_STATE, verbose=-1,
    )
    params.update(overrides)
    return lgb.LGBMClassifier(**params)


def lift_at(y, p, frac: float) -> dict[str, float]:
    """Conversion/hit rate in the top ``frac`` of scores vs the overall base rate."""
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(len(y) * frac)))
    top = y[np.argsort(-p)[:k]]
    return {"hit_rate": float(top.mean()), "base_rate": float(y.mean()),
            "lift": float(top.mean() / y.mean()) if y.mean() > 0 else float("nan"),
            "captured": float(top.sum() / max(1, y.sum()))}


def save_json(obj: Any, name: str) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        raise TypeError(type(o))
    (ARTIFACTS / name).write_text(json.dumps(obj, indent=2, default=default))


def load_json(name: str) -> Any:
    return json.loads((ARTIFACTS / name).read_text())


def save_model(obj: Any, name: str) -> None:
    joblib.dump(obj, ARTIFACTS / name)


def load_model(name: str) -> Any:
    return joblib.load(ARTIFACTS / name)
