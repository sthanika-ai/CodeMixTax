"""Command-line interface: ``cmb-indic <command>``.

    cmb-indic datasets                     what will be evaluated, and the coverage gaps
    cmb-indic models                       registered model cards, VRAM estimates, status
    cmb-indic download                     fetch the 18 splits into data/raw/
    cmb-indic validate                     preflight: configs, data, server reachability
    cmb-indic run --models a,b --datasets sa,mmlu
    cmb-indic rescore --run-dir runs/<id> --extraction paper
    cmb-indic report --run-dir runs/<id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .config import (
    ModelConfig,
    available_models,
    load_defaults,
    load_model_config,
    load_run_config,
    load_suite,
)
from .registry import (
    DATASETS,
    MISSING_UPSTREAM,
    TASK_ORDER,
    TASKS,
    TOTAL_ROWS,
    coverage_matrix,
    resolve,
)


# --------------------------------------------------------------------------- #
def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader so CMB_BASE_URL / HF_TOKEN work without extra deps."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def cmd_datasets(args: argparse.Namespace) -> int:
    specs = resolve(_split_csv(args.datasets))
    print(f"CodeMixTax: {len(DATASETS)} evaluable splits, {TOTAL_ROWS:,} rows total")
    print(f"Selected: {len(specs)} split(s), {sum(s.n_rows for s in specs):,} rows\n")

    print(f"{'split':<22}{'task':<12}{'language':<12}{'rows':>7}  {'metric':<16}labels")
    print("-" * 100)
    for spec in specs:
        labels = "" if spec.labels is None else f"{len(spec.labels)}: {','.join(spec.labels[:5])}"
        if spec.labels and len(spec.labels) > 5:
            labels += ",..."
        print(
            f"{spec.name:<22}{spec.task:<12}{spec.language.name:<12}"
            f"{spec.n_rows:>7}  {spec.metric:<16}{labels}"
        )

    print("\nCoverage (rows; -- = no data upstream)")
    names, rows = coverage_matrix()
    head = f"{'language':<12}" + "".join(f"{TASKS[t].pretty:>12}" for t in TASK_ORDER)
    print(head)
    print("-" * len(head))
    for name, row in zip(names, rows, strict=True):
        cells = "".join(f"{(str(v) if v is not None else '--'):>12}" for v in row)
        print(f"{name:<12}{cells}")

    print("\nNotes")
    for spec in specs:
        if spec.notes:
            print(f"  {spec.name}: {spec.notes}")

    if args.verbose:
        print("\nDefined upstream but not evaluable here")
        for name, why in MISSING_UPSTREAM.items():
            print(f"  {name}: {why}")
    return 0


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def cmd_models(args: argparse.Namespace) -> int:
    names = _split_csv(args.models) or available_models()
    if not names:
        print("No model configs found in configs/models/.", file=sys.stderr)
        return 1

    defaults = load_defaults()
    rows: list[tuple[str, ModelConfig]] = []
    for name in names:
        try:
            rows.append((name, load_model_config(name, defaults=defaults)))
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {name}: {exc}", file=sys.stderr)

    print(f"{'config':<26}{'params':>8}{'vram~':>8}  {'backend':<9}{'status':<13}hf_id")
    print("-" * 118)
    for name, cfg in rows:
        params = f"{cfg.params_b:g}B" if cfg.params_b else "?"
        vram = f"{cfg.est_vram_gb:g}G" if cfg.est_vram_gb else "?"
        flags = []
        if cfg.vision:
            flags.append("vision")
        if cfg.reasoning:
            flags.append("reasoning")
        if cfg.quantization:
            flags.append(cfg.quantization)
        if cfg.gated:
            flags.append("gated")
        suffix = f"  [{','.join(flags)}]" if flags else ""
        print(
            f"{name:<26}{params:>8}{vram:>8}  {cfg.backend:<9}{cfg.status:<13}{cfg.hf_id}{suffix}"
        )

    subs = [(n, c) for n, c in rows if c.status == "substituted"]
    if subs:
        print("\nSubstituted models -- the requested checkpoint does not exist on the Hub:")
        for name, cfg in subs:
            print(f"  {name}: {cfg.notes}")

    if args.verbose:
        print("\nEstimated VRAM is weight-only (params x bytes/param) and excludes KV cache.")
        print("Add roughly 15-30% headroom for activations and context.")
    return 0


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def cmd_download(args: argparse.Namespace) -> int:
    from .data import ensure_downloaded, sha256_of, write_manifest

    specs = resolve(_split_csv(args.datasets))
    data_dir = Path(args.data_dir)
    entries: dict[str, dict] = {}
    total = 0

    print(f"Downloading {len(specs)} split(s) into {data_dir}/")
    for spec in specs:
        path = ensure_downloaded(spec, data_dir, revision=args.revision, force=args.force)
        digest = sha256_of(path)
        size = path.stat().st_size
        total += size
        entries[spec.name] = {
            "hf_path": spec.hf_path,
            "local_path": str(path),
            "sha256": digest,
            "bytes": size,
            "expected_rows": spec.n_rows,
            "revision": args.revision or "main",
        }
        print(f"  {spec.name:<22} {size/1024:>9.0f} KiB  {digest[:12]}")

    manifest = write_manifest(data_dir, entries)
    print(f"\n{total/1024/1024:.1f} MiB total. Manifest: {manifest}")
    return 0


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
def cmd_validate(args: argparse.Namespace) -> int:
    from .data import local_path
    from .prompts import get_template

    ok = True
    print("== prompt templates ==")
    for spec in resolve(None):
        try:
            get_template(spec.name, args.prompt_file)
            print(f"  ok    {spec.name}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  FAIL  {spec.name}: {exc}")

    print("\n== local data ==")
    data_dir = Path(args.data_dir)
    for spec in resolve(None):
        path = local_path(spec, data_dir)
        if not path.exists():
            print(f"  miss  {spec.name} ({path}) -- run `cmb-indic download`")
            ok = False
            continue
        if args.check_rows:
            import pandas as pd

            n = len(pd.read_csv(path))
            flag = "ok  " if n == spec.n_rows else "DIFF"
            if n != spec.n_rows:
                ok = False
            print(f"  {flag}  {spec.name}: {n} rows (registry expects {spec.n_rows})")
        else:
            print(f"  ok    {spec.name}")

    print("\n== model configs ==")
    defaults = load_defaults()
    names = _split_csv(args.models) or available_models()
    cfgs: list[ModelConfig] = []
    for name in names:
        try:
            cfg = load_model_config(name, defaults=defaults)
            cfgs.append(cfg)
            print(f"  ok    {name} -> {cfg.hf_id} ({cfg.backend})")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  FAIL  {name}: {exc}")

    if args.check_server:
        print("\n== server reachability ==")
        from .backends.openai_compat import OpenAICompatBackend

        seen: set[str] = set()
        for cfg in cfgs:
            if cfg.backend != "openai":
                continue
            url = cfg.resolved_base_url()
            if url in seen:
                continue
            seen.add(url)
            probe = OpenAICompatBackend(cfg).probe()
            if probe.get("server_reachable"):
                print(f"  ok    {url} serving: {probe.get('server_models')}")
            else:
                ok = False
                print(f"  FAIL  {url}: {probe.get('server_probe_error')}")

    if args.check_hub:
        print("\n== hugging face model ids ==")
        import httpx

        for cfg in cfgs:
            # httpx speaks only http/https, so an hf_id read out of a config
            # file cannot turn this existence probe into a file:/ fetch.
            url = f"https://huggingface.co/api/models/{cfg.hf_id}"
            try:
                resp = httpx.get(url, timeout=20, follow_redirects=True)
                # 401/403 on a gated repo is expected without a token.
                verdict = "gated" if resp.status_code in (401, 403) else ""
                if verdict:
                    print(f"  {verdict:<5} {cfg.hf_id} (HTTP {resp.status_code})")
                    continue
                resp.raise_for_status()
                print(f"  ok    {cfg.hf_id}")
            except httpx.HTTPStatusError as exc:
                ok = False
                print(f"  {'FAIL':<5} {cfg.hf_id} (HTTP {exc.response.status_code})")
            except Exception as exc:  # noqa: BLE001
                ok = False
                print(f"  FAIL  {cfg.hf_id}: {exc}")

    print("\n" + ("All checks passed." if ok else "Some checks FAILED (see above)."))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    from .runner import SplitResult, default_run_id, run_sweep

    defaults = load_defaults()
    suite = load_suite(args.suite) if args.suite else {}

    model_names = _split_csv(args.models) or suite.get("models") or []
    if not model_names:
        print(
            "No models selected. Pass --models a,b or --suite <name>. "
            f"Available: {available_models()}",
            file=sys.stderr,
        )
        return 2

    model_overrides: dict = {}
    for field_name in (
        "backend", "base_url", "concurrency", "max_tokens", "temperature", "prompt_style",
        # vLLM / transformers load-time knobs. Overridable from the CLI because on a
        # shared GPU the right values depend on what other jobs currently hold, which
        # a checked-in config file cannot know.
        "gpu_memory_utilization", "tensor_parallel_size", "max_model_len", "quantization",
        "dtype",
    ):
        value = getattr(args, field_name, None)
        if value is not None:
            model_overrides[field_name] = value

    cfgs = [
        load_model_config(name, defaults=defaults, overrides=model_overrides)
        for name in model_names
    ]

    # An explicit --max-tokens must beat the per-task budgets in default.yaml.
    # Without this it is silently ignored for every task (they all have overrides),
    # so `--max-tokens 2048` would appear to do nothing.
    if args.max_tokens is not None:
        for cfg in cfgs:
            for task, over in list(cfg.task_overrides.items()):
                if "max_tokens" in over:
                    over = {k: v for k, v in over.items() if k != "max_tokens"}
                    cfg.task_overrides[task] = over
        print(f"note: --max-tokens {args.max_tokens} overrides all per-task budgets")

    run_cfg = load_run_config(
        defaults=defaults,
        suite=suite,
        overrides={
            "datasets": _split_csv(args.datasets),
            "shots": args.shots,
            "limit": args.limit,
            "sample": args.sample,
            "seed": args.seed,
            "extraction": args.extraction,
            "prompt_file": args.prompt_file,
            "data_dir": args.data_dir,
            "out_dir": args.out_dir,
            "resume": False if args.no_resume else None,
            "regenerate": _split_csv(args.regenerate),
            "dataset_revision": args.dataset_revision,
            "fail_fast": True if args.fail_fast else None,
        },
    )
    run_id = args.run_id or run_cfg.run_id or default_run_id()
    specs = resolve(run_cfg.datasets)

    n_rows = sum(min(s.n_rows, run_cfg.limit or s.n_rows) for s in specs)
    print(f"run_id       : {run_id}")
    print(f"models       : {', '.join(c.key for c in cfgs)}")
    print(f"splits       : {len(specs)} ({', '.join(s.name for s in specs)})")
    print(f"rows/model   : {n_rows:,}   total generations: {n_rows * len(cfgs):,}")
    print(f"shots        : {run_cfg.shots}   extraction: {run_cfg.extraction}   seed: {run_cfg.seed}")
    print(f"output       : {Path(run_cfg.out_dir) / run_id}")
    print()

    if args.dry_run:
        print("--dry-run: nothing generated. Prompt preview for the first split:\n")
        from .data import load_split
        from .prompts import build_prompts

        spec = specs[0]
        split = load_split(spec, Path(run_cfg.data_dir), limit=1, seed=run_cfg.seed)
        prompts = build_prompts(
            spec, split.frame, prompt_file=run_cfg.prompt_file,
            shots=run_cfg.shots, seed=run_cfg.seed, style=cfgs[0].prompt_style,
        )
        print(json.dumps(prompts[0], indent=2, ensure_ascii=False))
        return 0

    def on_result(res: SplitResult) -> None:
        if res.skipped:
            print(f"  SKIP {res.model_key}/{res.dataset}: {res.skip_reason}")
            return
        headline = TASKS[res.task].headline
        value = res.metrics.scores.get(headline)
        diag = res.metrics.diagnostics
        unparsed = diag.get("unparsed_rate", diag.get("format_error_rate", 0.0)) or 0.0
        ev = res.extra.get("n_evicted", 0)
        ev_s = f" evicted={ev}" if ev else ""
        print(
            f"  {res.model_key}/{res.dataset:<22} {headline}={value:6.2f}  "
            f"n={res.n_rows:<5} unparsed={unparsed:4.1f}%  "
            f"gen={res.n_generated} cached={res.n_cached}{ev_s} err={res.n_errors} "
            f"trunc={res.n_truncated}  {res.wall_s:.0f}s"
        )

    results = run_sweep(cfgs, run_cfg, run_id=run_id, on_result=on_result)

    failed = [r for r in results if r.skipped]
    print(f"\nDone: {len(results) - len(failed)} scored, {len(failed)} failed.")
    if failed:
        print("Failures:")
        for r in failed:
            print(f"  {r.model_key}/{r.dataset}: {r.skip_reason}")

    if not args.no_report:
        from .report import collect, write

        try:
            agg = collect(Path(run_cfg.out_dir) / run_id)
            paths = write(agg, Path(args.results_dir), run_dir=Path(run_cfg.out_dir) / run_id)
            print(f"\nReport: {paths['report']}")
        except Exception as exc:  # noqa: BLE001
            print(f"\nCould not build report: {exc}", file=sys.stderr)

    return 1 if failed and args.fail_fast else 0


# --------------------------------------------------------------------------- #
# rescore
# --------------------------------------------------------------------------- #
def cmd_rescore(args: argparse.Namespace) -> int:
    """Re-extract and re-score cached generations without touching a model.

    This is what makes the ``robust`` vs ``paper`` comparison cheap: the
    generations are already on disk, so switching extraction mode is a
    CPU-only pass over ``generations.jsonl``.
    """
    from .data import load_split
    # ALL_SPECS, not DATASETS: rescoring must also cover the derived condition
    # splits (mmlu_hineng_en, gsm8k_hineng_hirom, truthfulqa_hineng_rom, ...),
    # which `run` scores happily but which this path used to skip as "not in
    # registry" -- silently leaving stale metrics behind after a scorer fix.
    from .registry import ALL_SPECS as ALL
    from .runner import _score_split, read_generation_cache

    run_dir = Path(args.run_dir)
    gen_files = sorted(run_dir.glob("*/*/generations.jsonl"))
    if not gen_files:
        print(f"No generations.jsonl under {run_dir}", file=sys.stderr)
        return 1

    run_cfg = load_run_config(
        defaults=load_defaults(),
        overrides={
            "extraction": args.extraction,
            "data_dir": args.data_dir,
            "limit": args.limit,
            "shots": args.shots,
        },
    )

    print(f"Re-scoring {len(gen_files)} split(s) with extraction={args.extraction}")
    for gen_path in gen_files:
        split_dir = gen_path.parent
        name = split_dir.name
        spec = ALL.get(name)
        if spec is None:
            print(f"  skip {name}: not in registry")
            continue

        meta_path = split_dir / "run_meta.json"
        limit = run_cfg.limit
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            evaluated = meta.get("dataset", {}).get("n_rows_evaluated")
            full = meta.get("dataset", {}).get("n_rows_full")
            if evaluated and full and evaluated < full:
                limit = evaluated
            run_cfg.shots = meta.get("run", {}).get("shots", run_cfg.shots)
            run_cfg.sample = meta.get("run", {}).get("sample", run_cfg.sample)
            run_cfg.seed = meta.get("run", {}).get("seed", run_cfg.seed)

        split = load_split(
            spec, Path(run_cfg.data_dir), limit=limit,
            seed=run_cfg.seed, sample=run_cfg.sample,
        )
        cache = read_generation_cache(gen_path)
        target = split_dir if args.in_place else Path(args.out_dir) / split_dir.relative_to(run_dir.parent)
        target.mkdir(parents=True, exist_ok=True)

        result = _score_split(spec, split, cache, run_cfg, target)
        headline = TASKS[spec.task].headline
        print(f"  {split_dir.parent.name}/{name:<22} {headline}={result.scores.get(headline):6.2f}")

    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def cmd_report(args: argparse.Namespace) -> int:
    from .report import collect, write

    run_dir = Path(args.run_dir)
    agg = collect(run_dir)
    paths = write(agg, Path(args.out_dir), run_dir=run_dir)
    print(f"tidy    : {paths['tidy']}  ({len(agg.tidy)} rows)")
    print(f"summary : {paths['summary']}  ({len(agg.summary)} rows)")
    print(f"report  : {paths['report']}")
    if args.show:
        print()
        print(paths["report"].read_text(encoding="utf-8"))
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cmb-indic",
        description="Evaluate local LLMs on the CodeMixBench Indic subset "
                    "(Hindi, Bengali, Marathi, Tamil, Malayalam).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"cmb-indic {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_data_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", default="data/raw", help="local dataset cache")
        p.add_argument("--prompt-file", default="prompt.json", help="upstream prompt templates")

    # datasets
    p = sub.add_parser("datasets", help="list the evaluable splits and coverage gaps")
    p.add_argument("--datasets", help="filter: split/task/language names, comma-separated")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_datasets)

    # models
    p = sub.add_parser("models", help="list registered model cards")
    p.add_argument("--models", help="comma-separated config names")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_models)

    # download
    p = sub.add_parser("download", help="fetch splits from the Hugging Face Hub")
    p.add_argument("--datasets", help="default: all 18 Indic splits")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--revision", help="pin a dataset revision (commit sha or tag)")
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    p.set_defaults(func=cmd_download)

    # validate
    p = sub.add_parser("validate", help="preflight checks before a long sweep")
    add_data_args(p)
    p.add_argument("--models", help="comma-separated config names (default: all)")
    p.add_argument("--check-rows", action="store_true", help="verify row counts against the registry")
    p.add_argument("--check-server", action="store_true", help="probe /v1/models on each endpoint")
    p.add_argument("--check-hub", action="store_true", help="verify each hf_id resolves")
    p.set_defaults(func=cmd_validate)

    # run
    p = sub.add_parser("run", help="generate + score")
    add_data_args(p)
    p.add_argument("--models", help="comma-separated model config names")
    p.add_argument("--suite", help="suite config name (configs/suites/*.yaml)")
    p.add_argument("--datasets", help="splits/tasks/languages; default all 18")
    p.add_argument("--shots", type=int, default=None, help="in-context exemplars (MMLU/GSM8K/TruthfulQA)")
    p.add_argument("--limit", type=int, default=None, help="max rows per split (smoke tests)")
    p.add_argument("--sample", choices=["head", "random"], default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--extraction", choices=["robust", "paper"], default=None)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--run-id", help="default: UTC timestamp")
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--no-resume", action="store_true", help="ignore cached generations")
    p.add_argument(
        "--regenerate", default=None,
        help="comma-separated: truncated,errors,unparsed. Invalidates ONLY those cached "
             "rows so they regenerate (e.g. after raising --max-tokens); every other row "
             "is reused. Cheaper and more targeted than --no-resume.",
    )
    p.add_argument("--fail-fast", action="store_true", help="abort on the first split error")
    p.add_argument("--dry-run", action="store_true", help="print plan + one prompt, generate nothing")
    p.add_argument("--no-report", action="store_true")
    # model-level overrides
    p.add_argument("--backend", choices=["openai", "vllm", "hf", "echo"], default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--prompt-style", choices=["chat", "completion"], default=None)
    p.add_argument(
        "--gpu-memory-utilization", type=float, default=None,
        help="vLLM: FRACTION OF TOTAL GPU MEMORY, not of free memory. On a shared "
             "card, set it to (free_bytes / total_bytes) minus headroom, or vLLM "
             "will try to claim memory other jobs already hold and OOM.",
    )
    p.add_argument("--tensor-parallel-size", type=int, default=None,
                   help="vLLM: split the model across this many GPUs.")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Cap the context window. Lower = much smaller KV cache. The "
                        "longest prompt in this benchmark is only a few hundred tokens.")
    p.add_argument("--quantization", default=None,
                   help="vLLM quantization kernel: awq | gptq | compressed-tensors | fp8 | bitsandbytes.")
    p.add_argument("--dtype", default=None,
                   help="Weight dtype: auto | bfloat16 | float16 | float32. Use float32 for "
                        "CPU runs -- bfloat16 lacks native CPU kernels for some ops and can be "
                        "slower despite using half the memory.")
    p.set_defaults(func=cmd_run)

    # rescore
    p = sub.add_parser("rescore", help="re-extract + re-score cached generations (no model needed)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--extraction", choices=["robust", "paper"], default="robust")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--in-place", action="store_true", help="overwrite metrics.json in the run dir")
    p.add_argument("--out-dir", default="runs/rescored")
    p.set_defaults(func=cmd_rescore)

    # report
    p = sub.add_parser("report", help="aggregate a run into tidy.csv + REPORT.md")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--show", action="store_true", help="print the report to stdout")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Cached generations are preserved; re-run to resume.", file=sys.stderr)
        return 130
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
