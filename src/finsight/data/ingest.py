"""Download the Berka financial dataset from the CTU public MariaDB and cache it as parquet.

Usage:  python -m finsight.data.ingest [--force]
"""
import argparse
import pandas as pd
import pymysql

from finsight.config import DATA_RAW, SOURCE_DB, TABLES

# Translate the Czech codes once, at the boundary, so the rest of the code reads in English.
TRANS_TYPE = {"PRIJEM": "credit", "VYDAJ": "debit", "VYBER": "withdrawal"}
TRANS_OPERATION = {
    "VYBER KARTOU": "card_withdrawal",
    "VKLAD": "cash_deposit",
    "PREVOD Z UCTU": "transfer_in",
    "VYBER": "cash_withdrawal",
    "PREVOD NA UCET": "transfer_out",
}
K_SYMBOL = {
    "POJISTNE": "insurance",
    "SLUZBY": "statement_fee",
    "UROK": "interest_credited",
    "SANKC. UROK": "penalty_interest",
    "SIPO": "household",
    "DUCHOD": "pension",
    "UVER": "loan_payment",
    "LEASING": "leasing",
}
FREQUENCY = {
    "POPLATEK MESICNE": "monthly",
    "POPLATEK TYDNE": "weekly",
    "POPLATEK PO OBRATU": "after_transaction",
}
DISTRICT_COLS = {
    "A2": "district_name", "A3": "region", "A4": "population",
    "A5": "munis_lt_500", "A6": "munis_500_2k", "A7": "munis_2k_10k", "A8": "munis_gt_10k",
    "A9": "n_cities", "A10": "urban_ratio", "A11": "avg_salary",
    "A12": "unemployment_95", "A13": "unemployment_96",
    "A14": "entrepreneurs_per_1k", "A15": "crimes_95", "A16": "crimes_96",
}


def _clean(name: str, df: pd.DataFrame) -> pd.DataFrame:
    for c in ("date", "issued", "birth_date"):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    for c in df.columns:
        if df[c].dtype != object:
            continue
        sample = df[c].dropna()
        if len(sample) and isinstance(sample.iloc[0], str):
            df[c] = df[c].str.strip().replace({"": None})
        else:  # Decimal columns
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if name == "trans":
        df["type"] = df["type"].map(TRANS_TYPE).fillna(df["type"])
        df["operation"] = df["operation"].map(TRANS_OPERATION)
        df["k_symbol"] = df["k_symbol"].map(K_SYMBOL)
    elif name == "order":
        df["k_symbol"] = df["k_symbol"].map(K_SYMBOL)
    elif name == "account":
        df["frequency"] = df["frequency"].map(FREQUENCY)
    elif name == "district":
        df = df.rename(columns=DISTRICT_COLS)
    return df


def ingest(force: bool = False) -> None:
    conn = None
    for t in TABLES:
        out = DATA_RAW / f"{t}.parquet"
        if out.exists() and not force:
            print(f"[skip] {t} (cached)")
            continue
        conn = conn or pymysql.connect(**SOURCE_DB, connect_timeout=30, read_timeout=600)
        df = pd.read_sql(f"SELECT * FROM `{t}`", conn)
        df = _clean(t, df)
        df.to_parquet(out, index=False)
        print(f"[ok]   {t}: {len(df):,} rows")
    if conn:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ingest(ap.parse_args().force)
