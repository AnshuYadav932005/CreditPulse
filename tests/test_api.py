"""API smoke tests; skipped until the pipeline has produced the warehouse."""
import pytest

from finsight.config import DUCKDB_PATH

pytestmark = pytest.mark.skipif(not DUCKDB_PATH.exists(), reason="run the pipeline first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from finsight.api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_credit_score_and_reasons(client):
    r = client.post("/credit/score", json={"account_id": 19, "amount": 120000, "duration": 36})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["pd"] <= 1 and body["decision"] in {"auto_approve", "manual_review", "decline"}
    assert len(body["reasons"]) == 4


def test_bigger_loan_is_riskier(client):
    small = client.post("/credit/score", json={"account_id": 19, "amount": 20000, "duration": 36}).json()
    big = client.post("/credit/score", json={"account_id": 19, "amount": 500000, "duration": 12}).json()
    assert big["pd"] >= small["pd"]


def test_customer_and_watchlist(client):
    c = client.get("/customers/19").json()
    assert c["customer"]["account_id"] == 19 and len(c["offers"]) == 3
    assert isinstance(client.get("/ews/watchlist?level=all&limit=5").json(), list)
    assert client.get("/customers/999999").status_code == 404
