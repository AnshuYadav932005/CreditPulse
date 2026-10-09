import numpy as np
import pandas as pd

from finsight.models.common import binary_metrics, lift_at
from finsight.models.credit_scoring import approval_curve, decision_bands, pd_to_score
from finsight.models.ews import borrower_alerts


def test_score_is_monotone_and_anchored():
    s = pd_to_score(np.array([0.5, 0.1, 1 / 51, 0.001]))
    assert list(s) == sorted(s)
    assert s[2] == 600  # 50:1 good:bad odds anchors at 600 points


def test_approval_curve_perfect_model_has_zero_npa_until_bads():
    y = np.array([0] * 90 + [1] * 10)
    perfect = y.astype(float)
    rows = approval_curve(y, perfect, np.random.default_rng(0).random(100), rates=[0.9, 1.0])
    assert rows[0]["npa_model"] == 0.0
    assert rows[1]["npa_model"] == 0.1


def test_decision_bands_partition():
    y = np.array([0, 0, 1, 1])
    b = decision_bands(y, np.array([0.01, 0.2, 0.5, 0.9]), 0.1, 0.5)
    assert abs(b["auto_approve_share"] + b["manual_review_share"] + b["auto_decline_share"] - 1) < 1e-9
    assert b["bads_caught_by_decline"] == 1.0


def test_lift_and_metrics():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0, 10)
    assert lift_at(y, p, 0.2)["lift"] == 5.0
    assert binary_metrics(y, p)["auc"] == 1.0


def test_borrower_alert_requires_30_days_lead():
    df = pd.DataFrame({
        "loan_id": [1, 1, 2, 3],
        "asof_month": pd.to_datetime(["1998-01-01", "1998-02-01", "1998-03-01", "1998-03-01"]),
        "onset_date": pd.to_datetime(["1998-02-15", "1998-02-15", "1998-03-10", None]),
    })
    p = np.array([0.9, 0.9, 0.9, 0.9])
    r = borrower_alerts(df, p, 0.5)
    # loan 1: first alert 45 days ahead -> hit; loan 2: 9 days -> miss; loan 3 never delinquent -> miss
    assert r["borrowers_flagged"] == 3 and r["true_flags"] == 1
    assert r["recall"] == 0.5
