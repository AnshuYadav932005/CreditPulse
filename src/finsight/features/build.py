"""Point-in-time feature engineering shared by every model.

Everything is derived from a monthly account panel (one row per account per calendar month).
Features for any (entity, as-of month) pair use ONLY months strictly before the as-of month,
which is what prevents the temporal leakage that plagues naive work on this dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finsight.config import DATA_RAW, SNAPSHOT_DATE

WINDOWS = (3, 6, 12)


def load_raw() -> dict[str, pd.DataFrame]:
    names = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]
    return {n: pd.read_parquet(DATA_RAW / f"{n}.parquet") for n in names}


def month_start(s: pd.Series) -> pd.Series:
    return s.dt.to_period("M").dt.to_timestamp()


def months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)


# --------------------------------------------------------------------------- monthly panel
def monthly_panel(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Dense account x month panel of flows and balances, from account opening to snapshot."""
    t = raw["trans"].sort_values(["account_id", "date", "trans_id"]).copy()
    t["month"] = month_start(t["date"])
    amt = t["amount"].astype(float)
    is_in = t["type"].eq("credit")
    for col, mask in {
        "inflow": is_in,
        "outflow": ~is_in,
        "cash_wd": t["operation"].eq("cash_withdrawal"),
        "card_wd": t["operation"].eq("card_withdrawal"),
        "transfer_out": t["operation"].eq("transfer_out"),
        "household": t["k_symbol"].eq("household"),
        "insurance": t["k_symbol"].eq("insurance"),
        "pension": t["k_symbol"].eq("pension"),
        "loan_payment": t["k_symbol"].eq("loan_payment"),
    }.items():
        t[col] = np.where(mask, amt, 0.0)
    t["penalty"] = t["k_symbol"].eq("penalty_interest").astype(int)

    g = t.groupby(["account_id", "month"])
    p = g[["inflow", "outflow", "cash_wd", "card_wd", "transfer_out", "household",
           "insurance", "pension", "loan_payment", "penalty"]].sum()
    p["n_trans"] = g.size()
    p["bal_end"] = g["balance"].last()
    p["bal_min"] = g["balance"].min()
    p["bal_mean"] = g["balance"].mean()
    p = p.reset_index()

    # Densify: months without any transaction carry the last balance forward and zero flows.
    acc = raw["account"][["account_id", "date"]].copy()
    acc["open"] = month_start(acc["date"])
    end = pd.Timestamp(SNAPSHOT_DATE).to_period("M").to_timestamp()
    grid = pd.concat(
        [pd.DataFrame({"account_id": a, "month": pd.date_range(o, end, freq="MS")})
         for a, o in zip(acc["account_id"], acc["open"])],
        ignore_index=True,
    )
    p = grid.merge(p, on=["account_id", "month"], how="left")
    flow_cols = ["inflow", "outflow", "cash_wd", "card_wd", "transfer_out", "household",
                 "insurance", "pension", "loan_payment", "penalty", "n_trans"]
    p[flow_cols] = p[flow_cols].fillna(0)
    p["bal_end"] = p.groupby("account_id")["bal_end"].ffill()
    p["bal_min"] = p["bal_min"].fillna(p["bal_end"])
    p["bal_mean"] = p["bal_mean"].fillna(p["bal_end"])
    p[["bal_end", "bal_min", "bal_mean"]] = p[["bal_end", "bal_min", "bal_mean"]].fillna(0)
    p["neg_bal"] = (p["bal_min"] < 0).astype(int)
    return p


# --------------------------------------------------------------------------- static profile
def account_profile(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per account: owner demographics, district economics, account setup."""
    disp, client, district = raw["disp"], raw["client"], raw["district"]
    owner = disp[disp["type"] == "OWNER"].merge(client, on="client_id")
    prof = raw["account"].rename(columns={"date": "open_date"}).merge(
        owner[["account_id", "client_id", "gender", "birth_date", "disp_id"]], on="account_id")
    prof["has_disponent"] = prof["account_id"].isin(
        disp.loc[disp["type"] == "DISPONENT", "account_id"]).astype(int)
    d = district[["district_id", "region", "population", "urban_ratio", "avg_salary",
                  "unemployment_96", "entrepreneurs_per_1k", "crimes_96"]].copy()
    d["crime_rate"] = d["crimes_96"] / d["population"] * 1000
    prof = prof.merge(d.drop(columns="crimes_96"), on="district_id", how="left")
    prof["is_male"] = (prof["gender"] == "M").astype(int)
    prof["freq_monthly"] = (prof["frequency"] == "monthly").astype(int)
    return prof


def card_events(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    c = raw["card"].merge(raw["disp"][["disp_id", "account_id"]], on="disp_id")
    return c.groupby("account_id").agg(card_issued=("issued", "min"),
                                       card_type=("type", "first")).reset_index()


# --------------------------------------------------------------------------- windowed features
def window_features(panel: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Behavioural features for each row of ``keys`` (columns: key, account_id, asof_month).

    Uses panel months in [asof - 12, asof - 1]; nothing at or after the as-of month.
    """
    m = keys.merge(panel, on="account_id")
    m["lag"] = months_between(m["asof_month"], m["month"])
    hist = m[m["lag"] >= 1].groupby("key").size().rename("history_months")
    m = m[(m["lag"] >= 1) & (m["lag"] <= max(WINDOWS))]

    out = [hist]
    for w in WINDOWS:
        s = m[m["lag"] <= w]
        g = s.groupby("key")
        f = pd.DataFrame({
            f"inflow_mean_{w}m": g["inflow"].mean(),
            f"outflow_mean_{w}m": g["outflow"].mean(),
            f"netflow_mean_{w}m": g["inflow"].mean() - g["outflow"].mean(),
            f"bal_mean_{w}m": g["bal_mean"].mean(),
            f"bal_min_{w}m": g["bal_min"].min(),
            f"n_trans_mean_{w}m": g["n_trans"].mean(),
            f"neg_months_{w}m": g["neg_bal"].sum(),
            f"penalties_{w}m": g["penalty"].sum(),
        })
        if w == max(WINDOWS):
            outflow = g["outflow"].sum().replace(0, np.nan)
            f["inflow_cv_12m"] = g["inflow"].std() / g["inflow"].mean().replace(0, np.nan)
            f["bal_std_12m"] = g["bal_mean"].std()
            f["cash_share_12m"] = g["cash_wd"].sum() / outflow
            f["transfer_share_12m"] = g["transfer_out"].sum() / outflow
            f["card_wd_12m"] = g["card_wd"].sum()
            f["household_mean_12m"] = g["household"].mean()
            f["has_insurance_pmt"] = (g["insurance"].sum() > 0).astype(int)
            f["has_pension"] = (g["pension"].sum() > 0).astype(int)
            f["has_loan_pmt"] = (g["loan_payment"].sum() > 0).astype(int)
            f["low_bal_months_12m"] = s.assign(lb=s["bal_min"] < 5000).groupby("key")["lb"].sum()
        out.append(f)

    feats = pd.concat(out, axis=1)
    last = m[m["lag"] == 1].set_index("key")["bal_end"].rename("bal_end_last")
    feats = feats.join(last)
    # Momentum: recent quarter vs the full year.
    feats["inflow_trend"] = feats["inflow_mean_3m"] / feats["inflow_mean_12m"].replace(0, np.nan)
    feats["bal_trend"] = (feats["bal_mean_3m"] - feats["bal_mean_12m"]) / (feats["bal_mean_12m"].abs() + 1000)
    return feats.reset_index().rename(columns={"index": "key"})


def add_profile_features(df: pd.DataFrame, profile: pd.DataFrame, asof_col: str) -> pd.DataFrame:
    cols = ["account_id", "open_date", "birth_date", "is_male", "has_disponent", "freq_monthly",
            "urban_ratio", "avg_salary", "unemployment_96", "entrepreneurs_per_1k", "crime_rate"]
    df = df.merge(profile[cols], on="account_id", how="left")
    df["age"] = (df[asof_col] - df["birth_date"]).dt.days / 365.25
    df["tenure_months"] = months_between(df[asof_col], df["open_date"])
    return df.drop(columns=["open_date", "birth_date"])
