"""`fraudscore` command line: train, evaluate, score-file, drift, compare."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import typer

from fraudscore.settings import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Card-fraud scoring toolkit.")

DATA_HELP = "Parquet file (default: full data when downloaded, else the committed sample)."


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


@app.command()
def train(
    data: Path | None = typer.Option(None, help=DATA_HELP),
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


@app.command()
def evaluate(
    model: Path | None = typer.Option(None, help="Model bundle (default: models/champion)."),
    data: Path | None = typer.Option(None, help=DATA_HELP),
    split: str = typer.Option("test", help="train, valid or test."),
) -> None:
    """Print ranking metrics and expected cost for a saved model on one split."""
    from fraudscore.models import load_bundle
    from fraudscore.train import evaluate_bundle, load_split

    s = get_settings()
    bundle = load_bundle(model or s.model_path)
    sp = load_split(data or s.resolved_data_path())
    result = evaluate_bundle(bundle, sp, s, ci=True)
    if split not in result:
        raise typer.BadParameter("split must be train, valid or test")
    typer.echo(json.dumps({"version": bundle.version, split: result[split]}, indent=2))


@app.command("score-file")
def score_file(
    input_path: Path = typer.Argument(..., help="Parquet or CSV with V1..V28 and Amount."),
    output: Path = typer.Option(..., help="Output parquet or CSV."),
    model: Path | None = typer.Option(None, help="Model bundle (default: models/champion)."),
    reasons: bool = typer.Option(True, help="Add top-5 SHAP reason codes as a JSON column."),
) -> None:
    """Score every row of a file; adds score, decision and optional reason codes."""
    from fraudscore.explain import reason_codes
    from fraudscore.models import load_bundle

    s = get_settings()
    bundle = load_bundle(model or s.model_path)
    df = _read_table(input_path)
    missing = [f for f in bundle.features if f not in df.columns]
    if missing:
        raise typer.BadParameter(f"input is missing columns: {missing}")
    out = df.copy()
    out["score"] = bundle.proba(df)
    out["decision"] = (out["score"] >= bundle.threshold).map({True: "alert", False: "allow"})
    out["model_version"] = bundle.version
    if reasons:
        out["reasons"] = [json.dumps(c) for c in reason_codes(bundle, df)]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".csv":
        out.to_csv(output, index=False)
    else:
        out.to_parquet(output, index=False)
    typer.echo(
        f"scored {len(out)} rows, {int((out['decision'] == 'alert').sum())} alerts "
        f"(threshold {bundle.threshold:.4f}) -> {output}"
    )


@app.command()
def drift(
    batch: Path = typer.Argument(..., help="Parquet or CSV batch with the model features."),
    model: Path | None = typer.Option(None, help="Model bundle holding the reference sample."),
    config: Path | None = typer.Option(None, help="monitoring.yaml."),
    out_dir: Path | None = typer.Option(None, help="Directory for drift-<date>.md."),
    fail_on_alert: bool = typer.Option(False, help="Exit with code 2 when the verdict is ALERT."),
) -> None:
    """Compare a new batch with the training reference: KS and PSI per feature and on scores."""
    from fraudscore.drift import detect_drift, load_monitoring_config, render_drift_report
    from fraudscore.models import load_bundle

    s = get_settings()
    bundle = load_bundle(model or s.model_path)
    cfg = load_monitoring_config(config or s.monitoring_config)
    df = _read_table(batch)
    res = detect_drift(
        bundle.reference,
        df,
        bundle.features,
        cfg,
        reference_scores=bundle.reference_scores,
        current_scores=bundle.proba(df),
    )
    target_dir = out_dir or s.reports_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"drift-{res.date}.md"
    path.write_text(render_drift_report(res, batch.name), encoding="utf-8")
    typer.echo(f"{res.verdict}: {res.n_alert} features ALERT, {res.n_warn} WARN -> {path}")
    if fail_on_alert and res.verdict == "ALERT":
        raise typer.Exit(code=2)


@app.command()
def compare(
    champion: Path | None = typer.Option(None, help="Champion bundle."),
    challenger: Path | None = typer.Option(None, help="Challenger bundle."),
    data: Path | None = typer.Option(None, help=DATA_HELP),
    out_dir: Path | None = typer.Option(None, help="Where to write eval-*.json and compare.md."),
    config: Path = typer.Option(Path("evals.yaml"), help="Gate thresholds."),
    strict: bool = typer.Option(False, help="Exit 1 when the challenger fails the gate."),
) -> None:
    """Score champion and challenger on the test split; write eval JSON and a promotion verdict."""
    from fraudscore.compare import eval_result, promotion_gate, write_eval
    from fraudscore.models import load_bundle
    from fraudscore.train import load_split

    s = get_settings()
    data_path = data or s.resolved_data_path()
    target = out_dir or s.reports_dir
    sp = load_split(data_path)
    champ = load_bundle(champion or s.models_dir / "champion.joblib")
    chall = load_bundle(challenger or s.models_dir / "challenger.joblib")
    champ_json = target / "eval-champion.json"
    chall_json = target / "eval-challenger.json"
    write_eval(eval_result("champion", champ, sp, s, data_path.name), champ_json)
    write_eval(eval_result("challenger", chall, sp, s, data_path.name), chall_json)
    ok, table = promotion_gate(champ_json, chall_json, config)
    verdict = "PROMOTE" if ok else "KEEP CHAMPION"
    md = (
        f"# Champion vs challenger ({date.today().isoformat()})\n\n"
        f"Data: `{data_path.name}`, test split. Baseline = champion `{champ.version}`, "
        f"candidate = challenger `{chall.version}`. Thresholds from `{config.name}`.\n\n"
        f"**Decision: {verdict}**\n\n{table}\n"
    )
    (target / "compare.md").write_text(md, encoding="utf-8")
    typer.echo(md)
    if strict and not ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
