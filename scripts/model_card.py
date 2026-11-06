"""Write reports/model_card.md from reports/metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraudscore.modelcard import render_model_card

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=ROOT / "reports" / "metrics.json")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "model_card.md")
    args = p.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    args.out.write_text(render_model_card(metrics), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
