"""Orchestration: prompts -> generations -> predictions -> metrics, resumably.

Layout written per (run, model, split):

    runs/<run_id>/<model_key>/<dataset>/
        generations.jsonl   one JSON object per row: raw reply + token counts
        predictions.csv     gold vs extracted prediction, per row (auditable)
        metrics.json        scores + diagnostics
        run_meta.json       config, versions, checksums, timings

`generations.jsonl` is the expensive artefact and the unit of resume: it is
appended as rows complete and keyed by the split's own ``index``, so an
interrupted sweep restarts exactly where it stopped. Re-scoring with a different
``--extraction`` mode reuses the cached generations and costs no GPU time at all.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess  # nosec B404 -- only `git`, resolved via shutil.which(), argv list, no shell
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .backends import Backend, build_backend
from .config import ModelConfig, RunConfig
from .data import Split, load_split, read_manifest
from .extract import TokenExtraction, extract_prediction, strip_reasoning
from .metrics import UNK, MetricResult, score
from .prompts import build_prompts, prompt_fingerprint
from .registry import DatasetSpec, resolve


# --------------------------------------------------------------------------- #
@dataclass
class SplitResult:
    model_key: str
    dataset: str
    task: str
    lang: str
    metrics: MetricResult
    n_rows: int
    n_generated: int
    n_cached: int
    n_errors: int
    n_truncated: int
    wall_s: float
    out_dir: Path
    skipped: bool = False
    skip_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------- #
# Generation cache
# --------------------------------------------------------------------------- #
def read_generation_cache(path: Path) -> dict[int, dict]:
    """Load previously generated rows, keyed by dataset row index.

    A truncated final line (killed mid-write) is skipped rather than fatal, so an
    ungraceful interrupt never corrupts a resumable run.
    """
    if not path.exists():
        return {}
    cache: dict[int, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "index" in obj:
                cache[int(obj["index"])] = obj
    return cache


def append_generations(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# --------------------------------------------------------------------------- #
# Single split
# --------------------------------------------------------------------------- #
def run_split(
    spec: DatasetSpec,
    model_cfg: ModelConfig,
    run_cfg: RunConfig,
    *,
    backend: Backend | None = None,
    run_id: str = "",
    data_manifest: dict | None = None,
) -> SplitResult:
    """Evaluate one model on one split, reusing any cached generations."""
    started = time.perf_counter()
    task_cfg = model_cfg.for_task(spec.task)
    run_id = run_id or default_run_id()

    out_dir = Path(run_cfg.out_dir) / run_id / model_cfg.key / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / "generations.jsonl"

    split = load_split(
        spec,
        Path(run_cfg.data_dir),
        limit=run_cfg.limit,
        seed=run_cfg.seed,
        sample=run_cfg.sample,
        revision=run_cfg.dataset_revision,
    )
    prompts = build_prompts(
        spec,
        split.frame,
        prompt_file=run_cfg.prompt_file,
        shots=run_cfg.shots,
        seed=run_cfg.seed,
        style=task_cfg.prompt_style,
    )
    fingerprint = prompt_fingerprint(prompts)

    # ---- resume ---------------------------------------------------------- #
    cache = read_generation_cache(gen_path) if run_cfg.resume else {}
    if cache:
        stale = {k: v for k, v in cache.items() if v.get("prompt_fingerprint") not in (None, fingerprint)}
        if stale:
            raise RuntimeError(
                f"{spec.name}: {len(stale)} cached generations were produced with a "
                f"different prompt set (fingerprint mismatch). The prompt config "
                f"changed since that run. Delete {gen_path} to regenerate, or use a "
                f"new --run-id to keep both."
            )

    # Selective invalidation: drop cached rows the caller wants regenerated, so a
    # raised token budget re-runs only the rows that actually hit the old ceiling.
    # Safe because generations.jsonl is append-only and read_generation_cache keys
    # by index in file order -- a newly appended row supersedes the stale one.
    n_evicted = 0
    if cache and run_cfg.regenerate:
        want = set(run_cfg.regenerate)
        unparsed_idx: set[int] = set()
        if "unparsed" in want:
            prev = out_dir / "predictions.csv"
            if prev.exists():
                import pandas as _pd

                pdf = _pd.read_csv(prev, dtype=str)
                if "pred" in pdf.columns:
                    unparsed_idx = {int(i) for i, v in zip(pdf["index"], pdf["pred"], strict=True)
                                    if str(v).strip() == UNK}
        drop = set()
        for idx, row in cache.items():
            if "truncated" in want and row.get("finish_reason") == "length":
                drop.add(idx)
            if "errors" in want and row.get("error"):
                drop.add(idx)
            if idx in unparsed_idx:
                drop.add(idx)
        for idx in drop:
            cache.pop(idx, None)
        n_evicted = len(drop)

    todo = [i for i, idx in enumerate(split.indices) if idx not in cache]
    n_cached = len(split.indices) - len(todo)

    owns_backend = backend is None
    if todo:
        if backend is None:
            backend = build_backend(task_cfg)
        else:
            # Shared instance from the sweep: apply this task's sampling settings.
            backend.retarget(task_cfg)
    try:
        if todo:
            batch_prompts = [prompts[i] for i in todo]
            desc = f"{model_cfg.key}/{spec.name}"
            gens = backend.generate(batch_prompts, desc=desc)
            rows = []
            for pos, gen in zip(todo, gens, strict=True):
                rows.append(
                    {
                        "index": int(split.indices[pos]),
                        "row": int(pos),
                        "text": gen.text,
                        "finish_reason": gen.finish_reason,
                        "prompt_tokens": gen.prompt_tokens,
                        "completion_tokens": gen.completion_tokens,
                        "latency_s": round(gen.latency_s, 4),
                        "error": gen.error,
                        "prompt_fingerprint": fingerprint,
                        # Recorded per row so a split regenerated at a raised budget
                        # is auditable rather than a silent mix of two settings.
                        "max_tokens": task_cfg.max_tokens,
                    }
                )
            if run_cfg.save_generations:
                append_generations(gen_path, rows)
            for row in rows:
                cache[int(row["index"])] = row
    finally:
        if owns_backend and backend is not None:
            backend.close()

    # ---- extract --------------------------------------------------------- #
    result = _score_split(spec, split, cache, run_cfg, out_dir,
                          reasoning=bool(getattr(model_cfg, "reasoning", False)))

    n_errors = sum(1 for idx in split.indices if cache.get(idx, {}).get("error"))
    n_trunc = sum(1 for idx in split.indices if cache.get(idx, {}).get("finish_reason") == "length")

    meta = {
        "run_id": run_id,
        "created_utc": utc_now(),
        "cmb_indic_version": __version__,
        "dataset": {
            "name": spec.name,
            "task": spec.task,
            "language": spec.language.name,
            "hf_path": spec.hf_path,
            "n_rows_full": split.n_full,
            "n_rows_evaluated": len(split),
            "subsampled": split.subsampled,
            "csv_sha256": split.sha256,
            "expected_rows_in_registry": spec.n_rows,
        },
        "model": task_cfg.to_dict(),
        "run": run_cfg.to_dict(),
        "run_fingerprint": run_cfg.fingerprint(),
        "prompt_fingerprint": fingerprint,
        "backend": (backend.describe() if backend is not None else {"backend": "cache-only"}),
        "generation": {
            "n_generated_now": len(todo),
            "n_reused_from_cache": n_cached,
            "n_evicted_for_regeneration": n_evicted,
            "regenerate_requested": list(run_cfg.regenerate),
            "max_tokens_present": sorted({
                r.get("max_tokens") for r in cache.values() if r.get("max_tokens") is not None
            }),
            "n_errors": n_errors,
            "n_truncated": n_trunc,
        },
        "environment": environment_info(),
        "data_manifest": data_manifest or read_manifest(Path(run_cfg.data_dir)),
        "wall_s": round(time.perf_counter() - started, 2),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    return SplitResult(
        model_key=model_cfg.key,
        dataset=spec.name,
        task=spec.task,
        lang=spec.lang,
        metrics=result,
        n_rows=len(split),
        n_generated=len(todo),
        n_cached=n_cached,
        n_errors=n_errors,
        n_truncated=n_trunc,
        wall_s=meta["wall_s"],
        out_dir=out_dir,
        extra={"n_evicted": n_evicted},
    )


def _score_split(
    spec: DatasetSpec,
    split: Split,
    cache: dict[int, dict],
    run_cfg: RunConfig,
    out_dir: Path,
    reasoning: bool = False,
) -> MetricResult:
    """Extract predictions from cached generations, score, and persist both.

    ``reasoning`` marks models that emit a hidden chain-of-thought before the answer.
    For those, a generation cut off by the token budget contains no answer at all --
    the trace was still running -- so it must score as unparsed. Without that rule the
    extractor reads an option letter out of the reasoning (traces routinely enumerate
    the options), which manufactures a confident prediction the model never made and
    lands close to chance. Models that answer first and explain afterwards, such as
    Llama 3.1, are unaffected and must NOT get this treatment: their truncated rows
    do carry a real answer.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = run_cfg.extraction
    gold = split.gold
    tokens = split.tokens
    texts = [str(cache.get(idx, {}).get("text") or "") for idx in split.indices]

    records: list[dict] = []
    n_format_errors = 0

    if spec.metric == "token_label":
        assert tokens is not None, f"{spec.name} has no tokens column"  # nosec B101 -- internal invariant, not a security check
        preds: list[list[str]] = []
        for i, idx in enumerate(split.indices):
            res = extract_prediction(
                spec.task, texts[i], labels=spec.labels, gold_tokens=tokens[i], mode=mode
            )
            assert isinstance(res, TokenExtraction)  # nosec B101 -- internal invariant, not a security check
            if not res.ok:
                n_format_errors += 1
            preds.append(res.labels)
            records.append(
                {
                    "index": idx,
                    "n_tokens": len(tokens[i]),
                    "gold": json.dumps(gold[i], ensure_ascii=False),
                    "pred": json.dumps(res.labels, ensure_ascii=False),
                    "aligned_by": res.aligned_by,
                    "n_unk": res.n_missing,
                    "parse_ok": res.ok,
                    "raw": texts[i],
                }
            )
        # Gold and pred are equal length by construction (extractor pads to gold).
        scored = score(
            "token_label", gold, preds, labels=spec.labels, n_format_errors=n_format_errors
        )

    elif spec.metric == "sentence_label":
        preds_s: list[str] = []
        # Normalise gold up front so the per-row `correct` column written to
        # predictions.csv is decided by exactly the same comparison the metric
        # uses. Deriving it from the raw gold instead lets the audit trail
        # disagree with metrics.json wherever normalisation matters -- e.g. GSM8K
        # gold "1,250" vs an extracted "1250" reads as wrong in the CSV while
        # scoring as right, so a reviewer recomputing accuracy by hand gets a
        # different number than the pipeline reports.
        gold_s = [_norm_gold(spec, g) for g in gold]
        for i, idx in enumerate(split.indices):
            if reasoning and cache.get(idx, {}).get("finish_reason") == "length":
                # Truncated mid-trace: the model never emitted an answer.
                pred = UNK
            else:
                pred = extract_prediction(
                    spec.task,
                    texts[i],
                    labels=spec.labels,
                    gold=str(gold[i]) if spec.task == "gsm8k" else None,
                    mode=mode,
                )
            assert isinstance(pred, str)  # nosec B101 -- internal invariant, not a security check
            if pred == UNK:
                n_format_errors += 1
            preds_s.append(pred)
            records.append(
                {
                    "index": idx,
                    "gold": gold_s[i],
                    "gold_raw": gold[i],
                    "pred": pred,
                    "correct": str(pred).strip() == str(gold_s[i]).strip(),
                    "raw": texts[i],
                    "raw_stripped": strip_reasoning(texts[i])[:2000],
                }
            )
        scored = score(
            "sentence_label", gold_s, preds_s, labels=spec.labels, n_format_errors=n_format_errors
        )

    else:  # bleu
        hyps: list[str] = []
        for i, idx in enumerate(split.indices):
            pred = extract_prediction(spec.task, texts[i], mode=mode)
            assert isinstance(pred, str)  # nosec B101 -- internal invariant, not a security check
            hyps.append(pred)
            records.append({"index": idx, "gold": gold[i], "pred": pred, "raw": texts[i]})
        refs = [str(g) for g in gold]
        scored = score("bleu", refs, hyps, bleu_tokenize=spec.bleu_tokenize)

    pd.DataFrame.from_records(records).to_csv(
        out_dir / "predictions.csv", index=False, encoding="utf-8"
    )
    payload = scored.to_dict()
    payload.update(
        {
            "dataset": spec.name,
            "task": spec.task,
            "language": spec.language.name,
            "lang_code": spec.lang,
            "extraction": mode,
            "shots": run_cfg.shots,
            "n_rows": len(split),
            "headline_metric": _headline_key(spec),
            "headline_value": scored.scores.get(_headline_key(spec)),
        }
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return scored


def _norm_gold(spec: DatasetSpec, value) -> str:
    """Normalise a gold scalar so it compares against the extractor's output."""
    if spec.task == "gsm8k":
        from .extract import normalize_number

        return normalize_number(str(value))
    return str(value).strip()


def _headline_key(spec: DatasetSpec) -> str:
    from .registry import TASKS

    return TASKS[spec.task].headline


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def run_sweep(
    model_cfgs: list[ModelConfig],
    run_cfg: RunConfig,
    *,
    run_id: str = "",
    on_result=None,
) -> list[SplitResult]:
    """Evaluate every model on every selected split.

    Models are the outer loop so that a heavyweight backend (vLLM in-process, or
    a 27B model loaded via transformers) is built once and reused across all of
    that model's splits, instead of reloading weights 18 times.
    """
    run_id = run_id or run_cfg.run_id or default_run_id()
    specs = resolve(run_cfg.datasets)
    results: list[SplitResult] = []
    manifest = read_manifest(Path(run_cfg.data_dir))

    for model_cfg in model_cfgs:
        # Load weights once per model and reuse across splits. Safe as long as no
        # task override touches a load-time field; `Backend.retarget` enforces that.
        touched = {k for over in model_cfg.task_overrides.values() for k in over}
        share = not (touched & set(Backend.LOAD_TIME_FIELDS))
        backend: Backend | None = None
        try:
            if share and model_cfg.backend in ("vllm", "hf"):
                backend = build_backend(model_cfg)

            for spec in specs:
                try:
                    res = run_split(
                        spec,
                        model_cfg,
                        run_cfg,
                        backend=backend,
                        run_id=run_id,
                        data_manifest=manifest,
                    )
                except Exception as exc:  # noqa: BLE001 -- one split must not kill the sweep
                    if run_cfg.fail_fast:
                        raise
                    res = SplitResult(
                        model_key=model_cfg.key,
                        dataset=spec.name,
                        task=spec.task,
                        lang=spec.lang,
                        metrics=MetricResult(metric_kind=spec.metric),
                        n_rows=0,
                        n_generated=0,
                        n_cached=0,
                        n_errors=0,
                        n_truncated=0,
                        wall_s=0.0,
                        out_dir=Path(run_cfg.out_dir) / run_id / model_cfg.key / spec.name,
                        skipped=True,
                        skip_reason=f"{type(exc).__name__}: {exc}",
                    )
                results.append(res)
                if on_result:
                    on_result(res)
        finally:
            if backend is not None:
                backend.close()

    return results


# --------------------------------------------------------------------------- #
def environment_info() -> dict[str, Any]:
    """Capture what a reader needs to reproduce or explain a number."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    for mod in ("torch", "transformers", "vllm", "sacrebleu", "sklearn", "pandas", "openai"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001 # nosec B110 -- optional dependency, not security-relevant
            pass
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001 # nosec B110 -- GPU info is best effort, not security-relevant
        pass
    return info


def _git(*args: str) -> str:
    git = shutil.which("git")
    if not git:
        return ""
    try:
        return subprocess.run(  # nosec B603 -- absolute path from shutil.which(), argv list, no shell
            [git, *args],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
            timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- not a git checkout
        return ""
