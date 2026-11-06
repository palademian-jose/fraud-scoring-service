"""`fraudscore` command line: train, evaluate, score-file, drift, compare."""

from __future__ import annotations

from pathlib import Path

import typer

from fraudscore.settings import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Card-fraud scoring toolkit.")


@app.command()
def train(
    data: Path | None = typer.Option(None, help="Parquet file (default: full data, else sample)."),
    models_dir: Path | None = typer.Option(None, help="Where to write the joblib bundles."),
    reports_dir: Path | None = typer.Option(None, help="Where to write metrics and figures."),
) -> None:
    """Train baseline, champion and challenger; write bundles, metrics.json and figures."""
    from fraudscore.train import run_training

    s = get_settings()
    path = data or s.resolved_data_path()
    out = run_training(path, models_dir or s.models_dir, reports_dir or s.reports_dir, s)
    for role, r in out["models"].items():
        t = r["test"]
        typer.echo(
            f"{role:<10} PR-AUC {t['pr_auc']:.4f} "
            f"[{t['pr_auc_ci_low']:.3f}, {t['pr_auc_ci_high']:.3f}] "
            f"cost/10k {t['cost']['cost_per_10k']:.1f} threshold {r['threshold']:.4f}"
        )


if __name__ == "__main__":
    app()
