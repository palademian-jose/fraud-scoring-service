"""Measure /score latency against a real uvicorn server on localhost.

Sequential single-transaction requests with SHAP reason codes on (the slower, default path),
and again with `explain=false`. Writes reports/bench.json. The latency target is
P95_LATENCY_BUDGET_MS; the script exits 1 when the measured p95 exceeds it.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import uvicorn

from fraudscore.api import create_app
from fraudscore.data import FEATURES, TARGET, load_frame, time_split
from fraudscore.models import load_bundle
from fraudscore.settings import get_settings


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def run(client: httpx.Client, payloads: list[dict], explain: bool, warmup: int) -> dict:
    url = f"/score?explain={'true' if explain else 'false'}"
    for p in payloads[:warmup]:
        client.post(url, json=p).raise_for_status()
    lat = []
    started = time.perf_counter()
    for p in payloads:
        t0 = time.perf_counter()
        client.post(url, json=p).raise_for_status()
        lat.append((time.perf_counter() - t0) * 1000)
    wall = time.perf_counter() - started
    a = np.array(lat)
    return {
        "explain": explain,
        "requests": len(a),
        "p50_ms": round(float(np.percentile(a, 50)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2),
        "p99_ms": round(float(np.percentile(a, 99)), 2),
        "max_ms": round(float(a.max()), 2),
        "throughput_rps": round(len(a) / wall, 1),
    }


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=s.model_path)
    p.add_argument("--data", type=Path, default=s.resolved_data_path())
    p.add_argument("--requests", type=int, default=500)
    p.add_argument("--out", type=Path, default=s.reports_dir / "bench.json")
    args = p.parse_args()

    bundle = load_bundle(args.model)
    full_test = time_split(load_frame(args.data)).test
    frauds = full_test[full_test[TARGET] == 1].head(args.requests // 10)
    legit = full_test[full_test[TARGET] == 0].head(args.requests - len(frauds))
    test = pd.concat([frauds, legit]).sample(frac=1.0, random_state=s.seed)
    payloads = [{f: float(row[f]) for f in FEATURES} for _, row in test.iterrows()]
    cfg = s.model_copy(update={"rate_limit_burst": 10_000, "rate_limit_per_minute": 600_000})
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(bundle, cfg), host="127.0.0.1", port=port, log_level="warning", workers=1
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
        for _ in range(100):
            try:
                if client.get("/health").status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.1)
        with_reasons = run(client, payloads, True, warmup=50)
        without = run(client, payloads, False, warmup=50)
    server.should_exit = True
    thread.join(timeout=5)

    out = {
        "model_version": bundle.version,
        "frauds_in_payloads": int(test[TARGET].sum()),
        "budget_p95_ms": s.p95_latency_budget_ms,
        "within_budget": with_reasons["p95_ms"] <= s.p95_latency_budget_ms,
        "host": "localhost, single uvicorn worker, sequential client, CPU only",
        "with_reason_codes": with_reasons,
        "without_reason_codes": without,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["within_budget"] else 1)


if __name__ == "__main__":
    main()
