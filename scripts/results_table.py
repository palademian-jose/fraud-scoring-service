"""Write reports/results.md and refresh the results table inside README.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraudscore.results import inject, render_results

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=ROOT / "reports" / "metrics.json")
    p.add_argument("--bench", type=Path, default=ROOT / "reports" / "bench.json")
    p.add_argument("--out", type=Path, default=ROOT / "reports" / "results.md")
    p.add_argument("--readme", type=Path, default=ROOT / "README.md")
    p.add_argument("--no-readme", action="store_true")
    args = p.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    bench = json.loads(args.bench.read_text(encoding="utf-8")) if args.bench.exists() else None
    table = render_results(metrics, bench)
    args.out.write_text(table, encoding="utf-8")
    print(table)
    if not args.no_readme and args.readme.exists():
        args.readme.write_text(
            inject(args.readme.read_text(encoding="utf-8"), table), encoding="utf-8"
        )
        print(f"updated {args.readme}")


if __name__ == "__main__":
    main()
