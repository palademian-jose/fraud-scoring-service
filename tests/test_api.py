from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fraudscore.api import create_app
from fraudscore.data import FEATURES, TARGET, Split
from fraudscore.models import ModelBundle
from fraudscore.ratelimit import TokenBucketLimiter
from fraudscore.settings import Settings


@pytest.fixture()
def client(champion: ModelBundle, settings: Settings):
    cfg = settings.model_copy(update={"rate_limit_burst": 1000, "max_batch": 5})
    with TestClient(create_app(champion, cfg)) as c:
        yield c


def _tx(split: Split, label: int, i: int = 0) -> dict:
    row = split.test[split.test[TARGET] == label].iloc[i]
    return {f: float(row[f]) for f in FEATURES} | {"id": f"t{label}{i}"}


def test_health_and_model(client: TestClient, champion: ModelBundle) -> None:
    assert client.get("/health").json() == {"status": "ok", "model_version": champion.version}
    info = client.get("/model").json()
    assert info["threshold"] == champion.threshold and info["features"] == FEATURES


def test_score_single(client: TestClient, split: Split, champion: ModelBundle) -> None:
    r = client.post("/score", json=_tx(split, 1))
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["score"] <= 1
    assert body["model_version"] == champion.version
    assert body["id"] == "t10"
    assert len(body["reasons"]) == 5
    assert body["decision"] in {"alert", "allow"}


def test_fraud_scores_higher_than_legit(client: TestClient, split: Split) -> None:
    fraud = [client.post("/score", json=_tx(split, 1, i)).json()["score"] for i in range(5)]
    legit = [client.post("/score", json=_tx(split, 0, i)).json()["score"] for i in range(5)]
    assert sum(fraud) / 5 > sum(legit) / 5


def test_score_batch_matches_single(client: TestClient, split: Split) -> None:
    txs = [_tx(split, 1, 0), _tx(split, 0, 0)]
    batch = client.post("/score", json={"transactions": txs}).json()
    singles = [client.post("/score", json=t).json() for t in txs]
    assert [r["score"] for r in batch["results"]] == pytest.approx([s["score"] for s in singles])


def test_decision_follows_threshold(client: TestClient, split: Split) -> None:
    for label in (0, 1):
        for i in range(3):
            b = client.post("/score", json=_tx(split, label, i)).json()
            assert (b["decision"] == "alert") == (b["score"] >= b["threshold"])


def test_explain_off(client: TestClient, split: Split) -> None:
    r = client.post("/score?explain=false", json=_tx(split, 0))
    assert r.json()["reasons"] == []


def test_validation_errors(client: TestClient, split: Split) -> None:
    bad = _tx(split, 0)
    del bad["V3"]
    assert client.post("/score", json=bad).status_code == 422
    extra = _tx(split, 0) | {"surprise": 1}
    assert client.post("/score", json=extra).status_code == 422
    neg = _tx(split, 0) | {"Amount": -1}
    assert client.post("/score", json=neg).status_code == 422
    assert client.post("/score", json={"transactions": []}).status_code == 422


def test_batch_too_large(client: TestClient, split: Split) -> None:
    r = client.post("/score", json={"transactions": [_tx(split, 0)] * 6})
    assert r.status_code == 413


def test_rate_limit_returns_429(champion: ModelBundle, settings: Settings, split: Split) -> None:
    cfg = settings.model_copy(update={"rate_limit_burst": 2, "rate_limit_per_minute": 1})
    with TestClient(create_app(champion, cfg)) as c:
        codes = [c.post("/score", json=_tx(split, 0)).status_code for _ in range(4)]
        assert codes[:2] == [200, 200] and codes[-1] == 429
        assert "retry-after" in c.post("/score", json=_tx(split, 0)).headers
        assert c.get("/health").status_code == 200


def test_token_bucket_refills() -> None:
    now = [0.0]
    lim = TokenBucketLimiter(rate_per_minute=60, burst=2, clock=lambda: now[0])
    assert lim.allow("a")[0] and lim.allow("a")[0]
    ok, wait = lim.allow("a")
    assert not ok and wait == pytest.approx(1.0)
    assert lim.allow("b")[0]
    now[0] += 1.0
    assert lim.allow("a")[0]
