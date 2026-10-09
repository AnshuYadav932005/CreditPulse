"""Point-in-time guarantees of the feature store, on a tiny synthetic panel."""
import numpy as np
import pandas as pd

from finsight.features.build import months_between, window_features


def _panel(n_months=18):
    months = pd.date_range("1997-01-01", periods=n_months, freq="MS")
    p = pd.DataFrame({"account_id": 1, "month": months})
    for c in ["inflow", "outflow", "cash_wd", "card_wd", "transfer_out", "household",
              "insurance", "pension", "loan_payment", "penalty", "n_trans"]:
        p[c] = 1.0
    p["inflow"] = np.arange(n_months, dtype=float) * 100   # strictly increasing
    p["bal_end"] = p["bal_min"] = p["bal_mean"] = 10_000.0
    p["neg_bal"] = 0
    return p


def test_months_between():
    a = pd.Series(pd.to_datetime(["1998-03-01"]))
    b = pd.Series(pd.to_datetime(["1997-12-01"]))
    assert months_between(a, b).iloc[0] == 3


def test_window_features_ignore_asof_month_and_future():
    p = _panel()
    asof = pd.Timestamp("1997-07-01")  # index 6
    keys = pd.DataFrame({"key": [0], "account_id": [1], "asof_month": [asof]})
    before = window_features(p, keys)
    # Poison every month at/after the as-of month: features must not change.
    p2 = p.copy()
    p2.loc[p2["month"] >= asof, ["inflow", "bal_mean", "bal_min"]] = -1e9
    after = window_features(p2, keys)
    pd.testing.assert_frame_equal(before, after)
    # 3-month mean inflow = months 3,4,5 -> (300+400+500)/3
    assert before["inflow_mean_3m"].iloc[0] == 400
    assert before["history_months"].iloc[0] == 6


def test_window_caps_at_12_months():
    p = _panel(24)
    keys = pd.DataFrame({"key": [0], "account_id": [1], "asof_month": [pd.Timestamp("1998-12-01")]})
    f = window_features(p, keys)
    assert f["inflow_mean_12m"].iloc[0] == np.arange(11, 23).mean() * 100
