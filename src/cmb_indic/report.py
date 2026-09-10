"""Aggregate finished runs into a tidy table and a publishable Markdown report.

Reads every ``metrics.json`` under a run directory and emits:

    results/tidy.csv        one row per (model, split, metric) -- the machine-readable
                            artefact to commit alongside a paper or blog post
    results/summary.csv     one row per (model, split) with just the headline metric
    results/REPORT.md       leaderboards + per-task and per-language pivots

Design choices worth knowing:

*   **No cross-task averaging.** A single "CodeMixTax score" would mix
    BLEU with accuracy and average a 160-row split against a 3049-row one. The
    report shows per-task tables and a task-count-weighted mean rank instead.
*   **Baselines are shown inline.** Every classification table carries the split's
    majority-class baseline, because 68% accuracy on `sa_tameng` is *below* the
    majority baseline and a bare number hides that.
*   **Coverage is reported next to the score.** A model with a 40% unparsed rate
    and 45% accuracy is not comparable to one with 0% unparsed and 45%.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .registry import DATASETS, LANGUAGE_ORDER, LANGUAGES, TASK_ORDER, TASKS


@dataclass
class Aggregate:
    tidy: pd.DataFrame
    summary: pd.DataFrame


# --------------------------------------------------------------------------- #
def collect(run_dir: Path) -> Aggregate:
    """Walk a run directory and build the tidy + summary frames."""
    run_dir = Path(run_dir)
    metric_files = sorted(run_dir.glob("*/*/metrics.json"))
    if not metric_files:
        raise FileNotFoundError(
            f"No metrics.json found under {run_dir}. "
            "Point --run-dir at runs/<run_id> (or a parent of it) after a run completes."
        )

    tidy_rows: list[dict] = []
    summary_rows: list[dict] = []

    for path in metric_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_key = path.parent.parent.name
        dataset = payload.get("dataset", path.parent.name)
        spec = DATASETS.get(dataset)
        meta = _load_meta(path.parent / "run_meta.json")

        common = {
            "model": model_key,
            "model_display": meta.get("model", {}).get("display_name", model_key),
            "dataset": dataset,
            "task": payload.get("task", spec.task if spec else ""),
            "language": payload.get("language", spec.language.name if spec else ""),
            "lang_code": payload.get("lang_code", spec.lang if spec else ""),
            "n_rows": payload.get("n_rows", 0),
            "shots": payload.get("shots", 0),
            "extraction": payload.get("extraction", ""),
        }

        diagnostics = payload.get("diagnostics", {})
        for name, value in payload.get("scores", {}).items():
            tidy_rows.append({**common, "metric": name, "value": value})

        headline_key = payload.get("headline_metric") or (
            TASKS[common["task"]].headline if common["task"] in TASKS else "accuracy"
        )
        summary_rows.append(
            {
                **common,
                "metric": headline_key,
                "value": payload.get("scores", {}).get(headline_key),
                "majority_baseline": diagnostics.get("majority_baseline"),
                "unparsed_rate": diagnostics.get("unparsed_rate", diagnostics.get("format_error_rate")),
                "empty_hyp_rate": diagnostics.get("empty_hypothesis_rate"),
                "n_truncated": meta.get("generation", {}).get("n_truncated"),
                "backend": meta.get("backend", {}).get("backend"),
                "hf_id": meta.get("model", {}).get("hf_id"),
                "wall_s": meta.get("wall_s"),
            }
        )

    tidy = pd.DataFrame(tidy_rows)
    summary = pd.DataFrame(summary_rows)
    for frame in (tidy, summary):
        if not frame.empty:
            frame["task_order"] = frame["task"].map(
                {t: i for i, t in enumerate(TASK_ORDER)}
            )
            frame["lang_order"] = frame["lang_code"].map(
                {c: i for i, c in enumerate(LANGUAGE_ORDER)}
            )
    summary = summary.sort_values(["model", "task_order", "lang_order"]).reset_index(drop=True)
    tidy = tidy.sort_values(["model", "task_order", "lang_order", "metric"]).reset_index(drop=True)
    return Aggregate(tidy=tidy, summary=summary)


def _load_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------- #
def write(agg: Aggregate, out_dir: Path, *, run_dir: Path | None = None) -> dict[str, Path]:
    """Write tidy.csv, summary.csv and REPORT.md. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tidy_path = out_dir / "tidy.csv"
    summary_path = out_dir / "summary.csv"
    report_path = out_dir / "REPORT.md"

    agg.tidy.drop(columns=["task_order", "lang_order"], errors="ignore").to_csv(
        tidy_path, index=False
    )
    agg.summary.drop(columns=["task_order", "lang_order"], errors="ignore").to_csv(
        summary_path, index=False
    )
    report_path.write_text(render_markdown(agg, run_dir=run_dir), encoding="utf-8")

    return {"tidy": tidy_path, "summary": summary_path, "report": report_path}


def render_markdown(agg: Aggregate, *, run_dir: Path | None = None) -> str:
    """Build the human-readable report."""
    s = agg.summary
    if s.empty:
        return "# CodeMixTax results\n\nNo results found.\n"

    lines: list[str] = []
    lines.append("# CodeMixTax results")
    lines.append("")
    models = sorted(s["model"].unique())
    lines.append(
        f"{len(models)} model(s) x {s['dataset'].nunique()} split(s), "
        f"{int(s['n_rows'].sum()):,} scored rows."
    )
    if run_dir:
        lines.append(f"Source run directory: `{run_dir}`")
    modes = sorted(x for x in s["extraction"].dropna().unique())
    shots = sorted(int(x) for x in s["shots"].dropna().unique())
    lines.append(f"Extraction mode(s): {', '.join(modes) or 'n/a'} | shots: {shots}")
    lines.append("")
    lines.append(
        "> Scores are **not** averaged across tasks: accuracy and BLEU are different "
        "units, and split sizes range from 160 to 3,049 rows. Compare within a task."
    )
    lines.append("")

    # ---- per-task tables ---------------------------------------------- #
    lines.append("## Per-task results")
    lines.append("")
    for task in TASK_ORDER:
        block = s[s["task"] == task]
        if block.empty:
            continue
        metric = TASKS[task].headline
        unit = "BLEU" if metric == "bleu" else "accuracy %"
        lines.append(f"### {TASKS[task].pretty} ({unit})")
        lines.append("")
        pivot = block.pivot_table(
            index="model", columns="language", values="value", aggfunc="first"
        )
        langs = [
            LANGUAGES[c].name
            for c in LANGUAGE_ORDER
            if LANGUAGES[c].name in pivot.columns
        ]
        pivot = pivot[langs]
        lines.append(_md_table(pivot))
        lines.append("")

        base = block.dropna(subset=["majority_baseline"])
        if not base.empty:
            per_lang = base.groupby("language")["majority_baseline"].first()
            baseline_txt = ", ".join(f"{k} {v:.1f}%" for k, v in per_lang.items())
            lines.append(f"Majority-class baseline: {baseline_txt}")
            lines.append("")

    # ---- coverage / reliability --------------------------------------- #
    lines.append("## Extraction reliability")
    lines.append("")
    lines.append(
        "Share of rows where no answer could be parsed from the generation. "
        "A high value means the score reflects formatting failure, not capability -- "
        "check `max_tokens` and whether reasoning output was truncated."
    )
    lines.append("")
    rel = s.pivot_table(index="model", columns="task", values="unparsed_rate", aggfunc="mean")
    cols = [t for t in TASK_ORDER if t in rel.columns]
    lines.append(_md_table(rel[cols], fmt="{:.1f}"))
    lines.append("")

    trunc = s.groupby("model")["n_truncated"].sum(min_count=1).dropna()
    if not trunc.empty and trunc.sum() > 0:
        lines.append("Generations that hit the token ceiling (`finish_reason=length`):")
        lines.append("")
        for model, n in trunc[trunc > 0].sort_values(ascending=False).items():
            lines.append(f"- **{model}**: {int(n):,} rows")
        lines.append("")

    # ---- per-language rollup ------------------------------------------ #
    lines.append("## Per-language mean rank")
    lines.append("")
    lines.append(
        "Rank of each model within every split (1 = best), averaged per language. "
        "Rank-based so BLEU and accuracy can be combined without unit mixing."
    )
    lines.append("")
    ranked = s.dropna(subset=["value"]).copy()
    ranked["rank"] = ranked.groupby("dataset")["value"].rank(ascending=False, method="min")
    rank_pivot = ranked.pivot_table(index="model", columns="language", values="rank", aggfunc="mean")
    langs = [LANGUAGES[c].name for c in LANGUAGE_ORDER if LANGUAGES[c].name in rank_pivot.columns]
    rank_pivot = rank_pivot[langs]
    rank_pivot["overall"] = ranked.groupby("model")["rank"].mean()
    lines.append(_md_table(rank_pivot.sort_values("overall"), fmt="{:.2f}"))
    lines.append("")

    # ---- caveats ------------------------------------------------------- #
    lines.append("## Caveats carried from the benchmark")
    lines.append("")
    for name in sorted(s["dataset"].unique()):
        spec = DATASETS.get(name)
        if spec and spec.notes:
            lines.append(f"- **{name}** ({spec.n_rows:,} rows): {spec.notes}")
    lines.append("")
    lines.append("## Failed / skipped splits")
    lines.append("")
    missing = s[s["value"].isna()]
    if missing.empty:
        lines.append("None -- every selected split produced a score.")
    else:
        for _, row in missing.iterrows():
            lines.append(f"- `{row['model']}` / `{row['dataset']}`: no score recorded")
    lines.append("")
    return "\n".join(lines)


def _md_table(frame: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    """Render a pivot as a GitHub Markdown table, blanks for missing cells."""
    if frame.empty:
        return "_(no data)_"
    out = frame.copy()
    for col in out.columns:
        out[col] = out[col].map(lambda v: fmt.format(v) if pd.notna(v) else "--")
    header = "| model | " + " | ".join(str(c) for c in out.columns) + " |"
    divider = "|---|" + "---|" * len(out.columns)
    body = [
        "| " + str(idx) + " | " + " | ".join(row.tolist()) + " |"
        for idx, row in out.iterrows()
    ]
    return "\n".join([header, divider, *body])
