"""Offline demo on the committed 5,000-row sample: train, score, explain, drift, compare.

Writes only under models/demo/ and reports/demo/. No network access.
"""

from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from fraudscore.api import create_app
from fraudscore.compare import eval_result, promotion_gate, write_eval
from fraudscore.data import FEATURES, TARGET, load_frame, time_split
from fraudscore.drift import detect_drift, load_monitoring_config, render_drift_report
from fraudscore.models import load_bundle
from fraudscore.settings import ROOT, get_settings
from fraudscore.synth import make_stream
from fraudscore.train import run_training

MODELS = ROOT / "models" / "demo"
REPORTS = ROOT / "reports" / "demo"


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    s = get_settings()
    sample = s.sample_path
    banner("1. train baseline, champion, challenger on the 5,000-row sample")
    out = run_training(sample, MODELS, REPORTS, s)
    for role, r in out["models"].items():
        t = r["test"]
        print(
            f"{role:<10} test PR-AUC {t['pr_auc']:.3f}  recall@1% {t['recall_at_1pct']:.3f}  "
            f"cost/10k {t['cost']['cost_per_10k']:.0f}"
        )

    banner("2. score one fraud and one legitimate transaction through the API")
    champion = load_bundle(MODELS / "champion.joblib")
    split = time_split(load_frame(sample))
    test = split.test
    best_fraud = test[test[TARGET] == 1].iloc[[champion.proba(test[test[TARGET] == 1]).argmax()]]
    legit = test[test[TARGET] == 0].head(1)
    with TestClient(create_app(champion, s)) as client:
        for name, row in (("fraud", best_fraud), ("legitimate", legit)):
            payload = {f: float(row.iloc[0][f]) for f in FEATURES}
            body = client.post("/score", json=payload).json()
            print(
                f"{name}: score {body['score']:.4f} -> {body['decision']} "
                f"(threshold {body['threshold']:.4f}, model {body['model_version']})"
            )
            for r in body["reasons"]:
                print(
                    f"    {r['feature']:<7} value {r['value']:+.2f}  "
                    f"SHAP {r['contribution']:+.2f}  {r['direction']}"
                )

    banner("3. drift: unshifted vs shifted synthetic stream")
    cfg = load_monitoring_config(s.monitoring_config)
    for label, shift in (("unshifted", False), ("shifted", True)):
        stream = make_stream(split.train, 3000, shift=shift)
        res = detect_drift(
            champion.reference,
            stream,
            FEATURES,
            cfg,
            reference_scores=champion.reference_scores,
            current_scores=champion.proba(stream),
        )
        path = REPORTS / f"drift-{label}.md"
        path.write_text(render_drift_report(res, f"synthetic {label} stream"), encoding="utf-8")
        top = ", ".join(f.name for f in res.features if f.status == "ALERT")
        print(f"{label}: {res.verdict} ({res.n_alert} features ALERT: {top or 'none'}) -> {path}")

    banner("4. champion vs challenger promotion gate")
    chall = load_bundle(MODELS / "challenger.joblib")
    write_eval(
        eval_result("champion", champion, split, s, sample.name), REPORTS / "eval-champion.json"
    )
    write_eval(
        eval_result("challenger", chall, split, s, sample.name), REPORTS / "eval-challenger.json"
    )
    ok, table = promotion_gate(
        REPORTS / "eval-champion.json", REPORTS / "eval-challenger.json", ROOT / "evals.yaml"
    )
    print(table)
    print("decision:", "PROMOTE challenger" if ok else "KEEP champion")
    print("\nDemo artifacts are in", REPORTS)


if __name__ == "__main__":
    main()
