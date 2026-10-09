"""Paths and constants shared across the pipeline."""
from pathlib import Path
import os

ROOT = Path(os.environ.get("FINSIGHT_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
DUCKDB_PATH = ROOT / "data" / "finsight.duckdb"

# Public, read-only guest access published by the CTU Relational Learning Repository.
SOURCE_DB = dict(
    host="relational.fel.cvut.cz",
    port=3306,
    user="guest",
    password="ctu-relational",
    database="financial",
)
TABLES = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]

# The dataset's "today": the last day with transactions.
SNAPSHOT_DATE = "1998-12-31"
RANDOM_STATE = 42

for p in (DATA_RAW, DATA_PROCESSED, ARTIFACTS):
    p.mkdir(parents=True, exist_ok=True)
