"""Generate a labelled synthetic stream for drift demos.

uv run python scripts/synth.py --out data/synthetic/stream.parquet
uv run python scripts/synth.py --shift --out data/synthetic/stream_shifted.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fraudscore.data import load_frame, time_split
from fraudscore.settings import get_settings
from fraudscore.synth import make_stream


def main() -> None:
    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference",
        type=Path,
        default=s.sample_path,
        help="parquet to resample; only its training (first 60%%) rows are used",
    )
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--shift", action="store_true", help="apply a covariate shift")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    stream = make_stream(
        time_split(load_frame(args.reference)).train, args.rows, shift=args.shift, seed=args.seed
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stream.to_parquet(args.out, index=False)
    print(f"wrote {len(stream)} rows (shift={args.shift}) -> {args.out}")


if __name__ == "__main__":
    main()
