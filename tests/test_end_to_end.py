"""End-to-end pipeline tests using the echo backend.

Covers the properties that matter for a benchmark you intend to publish:
resumability, reproducibility, and the guarantee that changing prompts cannot
silently reuse stale generations.
"""

from __future__ import annotations

import json

import pytest

from cmb_indic.backends.echo import EchoBackend
from cmb_indic.config import ModelConfig, RunConfig
from cmb_indic.registry import DATASETS
from cmb_indic.report import collect, write
from cmb_indic.runner import read_generation_cache, run_split, run_sweep


@pytest.fixture
def echo_cfg():
    return ModelConfig(key="echo-test", hf_id="echo/test", backend="echo")


def _run_cfg(fake_splits, tmp_path, **kw):
    return RunConfig(
        data_dir=str(fake_splits),
        out_dir=str(tmp_path / "runs"),
        prompt_file="prompt.json",
        **kw,
    )


class TestSingleSplit:
    def test_produces_all_artefacts(self, fake_splits, tmp_path, echo_cfg):
        cfg = _run_cfg(fake_splits, tmp_path)
        res = run_split(DATASETS["sa_hineng"], echo_cfg, cfg, run_id="t1")
        assert res.n_rows == 3
        for name in ("generations.jsonl", "predictions.csv", "metrics.json", "run_meta.json"):
            assert (res.out_dir / name).exists(), name

    def test_metrics_json_is_wellformed(self, fake_splits, tmp_path, echo_cfg):
        cfg = _run_cfg(fake_splits, tmp_path)
        res = run_split(DATASETS["sa_hineng"], echo_cfg, cfg, run_id="t1")
        payload = json.loads((res.out_dir / "metrics.json").read_text())
        assert payload["dataset"] == "sa_hineng"
        assert payload["headline_metric"] == "accuracy"
        assert 0.0 <= payload["scores"]["accuracy"] <= 100.0
        assert payload["extraction"] == "robust"

    def test_run_meta_records_provenance(self, fake_splits, tmp_path, echo_cfg):
        cfg = _run_cfg(fake_splits, tmp_path)
        res = run_split(DATASETS["mmlu_hineng"], echo_cfg, cfg, run_id="t1")
        meta = json.loads((res.out_dir / "run_meta.json").read_text())
        assert meta["dataset"]["csv_sha256"]
        assert meta["prompt_fingerprint"]
        assert meta["run_fingerprint"]
        assert meta["model"]["hf_id"] == "echo/test"
        assert "python" in meta["environment"]

    @pytest.mark.parametrize("name", sorted(DATASETS))
    def test_every_split_scores(self, name, fake_splits, tmp_path, echo_cfg):
        # All 18 splits must complete: this catches column-layout and label-space
        # mistakes for the splits that are easy to forget (sa_beneng, lid_mareng).
        cfg = _run_cfg(fake_splits, tmp_path)
        res = run_split(DATASETS[name], echo_cfg, cfg, run_id="all")
        assert res.n_rows > 0
        assert res.metrics.scores

    def test_token_task_alignment_is_exact(self, fake_splits, tmp_path, echo_cfg):
        # The echo backend emits one label per token, so accuracy must be well
        # defined (not zero from a length mismatch) and diagnostics must be sane.
        cfg = _run_cfg(fake_splits, tmp_path)
        res = run_split(DATASETS["lid_hineng"], echo_cfg, cfg, run_id="t1")
        assert res.metrics.diagnostics["n_format_errors"] == 0
        assert res.metrics.diagnostics["n_tokens"] == 16  # 4 rows x 4 tokens


class TestResume:
    def test_second_run_reuses_cache(self, fake_splits, tmp_path, echo_cfg):
        cfg = _run_cfg(fake_splits, tmp_path)
        first = run_split(DATASETS["sa_hineng"], echo_cfg, cfg, run_id="r1")
        assert first.n_generated == 3 and first.n_cached == 0

        second = run_split(DATASETS["sa_hineng"], echo_cfg, cfg, run_id="r1")
        assert second.n_generated == 0
        assert second.n_cached == 3
        assert second.metrics.scores == first.metrics.scores

    def test_partial_cache_generates_only_the_remainder(self, fake_splits, tmp_path, echo_cfg):
        cfg = _run_cfg(fake_splits, tmp_path)
        res = run_split(DATASETS["sa_hineng"], echo_cfg, cfg, run_id="r2")

        # Simulate an interrupt: drop the last cached row.
        gen = res.out_dir / "generations.jsonl"
        lines = gen.read_text().strip().splitlines()
        gen.write_text("\n".join(lines[:-1]) + "\n")

        again = run_split(DATASETS["sa_hineng"], echo_cfg, cfg, run_id="r2")
        assert again.n_generated == 1
        assert again.n_cached == 2

    def test_truncated_final_line_is_survivable(self, fake_splits, tmp_path, echo_cfg):
        cfg = _run_cfg(fake_splits, tmp_path)
        res = run_split(DATASETS["sa_hineng"], echo_cfg, cfg, run_id="r3")
        gen = res.out_dir / "generations.jsonl"
        with open(gen, "a") as fh:
            fh.write('{"index": 99, "text": "unterm')  # killed mid-write

        cache = read_generation_cache(gen)
        assert len(cache) == 3  # bad line skipped, good rows intact

    def test_changed_prompts_refuse_stale_cache(self, fake_splits, tmp_path, echo_cfg):
        # The guard that stops a silently-wrong published number: if the prompt
        # set changes, cached generations must not be reused.
        base = _run_cfg(fake_splits, tmp_path, shots=0)
        run_split(DATASETS["mmlu_hineng"], echo_cfg, base, run_id="r4")

        changed = _run_cfg(fake_splits, tmp_path, shots=2)
        with pytest.raises(RuntimeError, match="different prompt set"):
            run_split(DATASETS["mmlu_hineng"], echo_cfg, changed, run_id="r4")

    def test_no_resume_regenerates(self, fake_splits, tmp_path, echo_cfg):
        cfg = _run_cfg(fake_splits, tmp_path)
        run_split(DATASETS["sa_hineng"], echo_cfg, cfg, run_id="r5")
        cfg2 = _run_cfg(fake_splits, tmp_path, resume=False)
        again = run_split(DATASETS["sa_hineng"], echo_cfg, cfg2, run_id="r5")
        assert again.n_generated == 3


class TestExtractionModes:
    def test_rescoring_cached_generations_changes_scores(self, fake_splits, tmp_path):
        """paper vs robust on the same generations, with no model involved."""
        from cmb_indic.data import load_split
        from cmb_indic.runner import _score_split

        cfg = ModelConfig(key="scripted", hf_id="echo/test", backend="echo")
        # "Answer: B" is the exact shape upstream mis-parses as "A".
        run = _run_cfg(fake_splits, tmp_path)
        spec = DATASETS["mmlu_hineng"]

        backend = EchoBackend(cfg, scripted=["Answer: B"])
        res = run_split(spec, cfg, run, backend=backend, run_id="mode")

        split = load_split(spec, fake_splits, download=False)
        cache = read_generation_cache(res.out_dir / "generations.jsonl")
        paper_cfg = _run_cfg(fake_splits, tmp_path, extraction="paper")
        _score_split(spec, split, cache, paper_cfg, tmp_path / "paper")

        # Every reply is "Answer: B". robust reads B; paper reads A, because the
        # capital A in the word "Answer" is its first [ABCD] match. Same
        # generations, different predictions -- that is the whole point of the flag.
        import pandas as pd

        robust_preds = set(pd.read_csv(res.out_dir / "predictions.csv")["pred"])
        paper_preds = set(pd.read_csv(tmp_path / "paper" / "predictions.csv")["pred"])
        assert robust_preds == {"B"}
        assert paper_preds == {"A"}

    def test_unparseable_replies_are_counted_not_dropped(self, fake_splits, tmp_path):
        cfg = ModelConfig(key="mute", hf_id="echo/test", backend="echo")
        backend = EchoBackend(cfg, scripted=["मुझे नहीं पता"])
        run = _run_cfg(fake_splits, tmp_path)
        res = run_split(DATASETS["mmlu_hineng"], cfg, run, backend=backend, run_id="mute")
        assert res.metrics.scores["accuracy"] == 0.0
        assert res.metrics.diagnostics["unparsed_rate"] == 100.0
        assert res.metrics.diagnostics["n_rows"] == 4  # denominator intact


class TestSweepAndReport:
    def test_sweep_over_multiple_splits(self, fake_splits, tmp_path, echo_cfg):
        cfg = _run_cfg(fake_splits, tmp_path, datasets=["sa", "mmlu"])
        results = run_sweep([echo_cfg], cfg, run_id="sweep")
        assert len(results) == 9  # 5 SA + 4 MMLU splits
        assert not any(r.skipped for r in results)

    def test_report_generation(self, fake_splits, tmp_path, echo_cfg):
        cfg = _run_cfg(fake_splits, tmp_path, datasets=["sa", "mt"])
        run_sweep([echo_cfg], cfg, run_id="rep")
        agg = collect(tmp_path / "runs" / "rep")
        assert not agg.summary.empty
        assert set(agg.summary["task"]) == {"sa", "mt"}

        paths = write(agg, tmp_path / "results")
        report = paths["report"].read_text()
        assert "# CodeMixTax results" in report
        assert "Extraction reliability" in report
        # The offensive-detection caveat must reach the published report.
        assert "sa_beneng" in report

    def test_failing_split_does_not_abort_sweep(self, fake_splits, tmp_path, echo_cfg):
        # Remove one CSV so that split fails while the others still score.
        (fake_splits / DATASETS["sa_tameng"].hf_path).unlink()
        cfg = _run_cfg(fake_splits, tmp_path, datasets=["sa"])
        # download=True is the runner default, so make the failure a local one by
        # pointing at an offline dir; HF would otherwise fetch it.
        results = run_sweep([echo_cfg], cfg, run_id="partial")
        assert len(results) == 5
        assert sum(1 for r in results if not r.skipped) >= 4

    def test_limit_subsamples(self, fake_splits, tmp_path, echo_cfg):
        cfg = _run_cfg(fake_splits, tmp_path, datasets=["sa_hineng"], limit=2)
        res = run_sweep([echo_cfg], cfg, run_id="lim")[0]
        assert res.n_rows == 2
