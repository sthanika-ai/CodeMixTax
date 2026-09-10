"""Registry invariants and (optionally) live agreement with the Hub."""

from __future__ import annotations

import pytest

from cmb_indic.config import available_models, load_defaults, load_model_config
from cmb_indic.registry import (
    DATASETS,
    LANGUAGE_ORDER,
    LANGUAGES,
    TASK_ORDER,
    TASKS,
    TOTAL_ROWS,
    coverage_matrix,
    resolve,
)

EXPECTED_TOTAL = 21_256
EXPECTED_SPLITS = 18


class TestShape:
    def test_split_count_and_total_rows(self):
        assert len(DATASETS) == EXPECTED_SPLITS
        assert TOTAL_ROWS == EXPECTED_TOTAL

    def test_per_language_totals(self):
        # Measured directly from the CSVs; these are the numbers the report quotes.
        expected = {"hin": 6218, "ben": 4114, "mar": 5657, "tam": 4096, "mal": 1171}
        actual: dict[str, int] = {}
        for spec in DATASETS.values():
            actual[spec.lang] = actual.get(spec.lang, 0) + spec.n_rows
        assert actual == expected
        assert sum(expected.values()) == EXPECTED_TOTAL

    def test_per_task_totals(self):
        expected = {
            "lid": 2084, "pos": 160, "ner": 314, "sa": 7731,
            "mt": 4942, "mmlu": 4252, "gsm8k": 1016, "truthfulqa": 757,
        }
        actual: dict[str, int] = {}
        for spec in DATASETS.values():
            actual[spec.task] = actual.get(spec.task, 0) + spec.n_rows
        assert actual == expected

    def test_nepali_excluded(self):
        assert "nep" not in LANGUAGES
        assert not any("nep" in name for name in DATASETS)

    def test_every_language_and_task_known(self):
        for spec in DATASETS.values():
            assert spec.lang in LANGUAGES
            assert spec.task in TASKS

    def test_name_matches_key(self):
        for key, spec in DATASETS.items():
            assert key == spec.name

    def test_hf_paths_well_formed(self):
        for spec in DATASETS.values():
            assert spec.hf_path == f"{spec.task}/{spec.name}.csv"

    def test_label_sets_present_where_required(self):
        for spec in DATASETS.values():
            if spec.metric == "sentence_label" and spec.task != "gsm8k":
                assert spec.labels, f"{spec.name} needs a label set"
            if spec.metric == "token_label":
                assert spec.labels, f"{spec.name} needs a label set"
            if spec.metric == "bleu":
                assert spec.bleu_tokenize

    def test_label_sets_have_no_duplicates(self):
        for spec in DATASETS.values():
            if spec.labels:
                assert len(set(spec.labels)) == len(spec.labels), spec.name

    def test_non_uniform_sa_label_spaces_are_documented(self):
        # The trap this guards: sa_beneng is offensive-language detection, and the
        # Tamil/Malayalam splits are 4-class with different casing.
        assert DATASETS["sa_beneng"].labels == ("O", "N")
        assert "offensive" in DATASETS["sa_beneng"].notes.lower()
        assert DATASETS["sa_hineng"].labels[0] == "positive"
        assert DATASETS["sa_tameng"].labels[0] == "Positive"
        assert DATASETS["lid_mareng"].labels == ("ENG", "MAR", "OTH")


class TestCoverage:
    def test_matrix_dimensions(self):
        names, rows = coverage_matrix()
        assert len(names) == len(LANGUAGE_ORDER)
        assert all(len(r) == len(TASK_ORDER) for r in rows)

    def test_hindi_is_the_only_complete_language(self):
        names, rows = coverage_matrix()
        complete = [n for n, r in zip(names, rows, strict=True) if all(v is not None for v in r)]
        assert complete == ["Hindi"]

    def test_matrix_sums_to_total(self):
        _, rows = coverage_matrix()
        assert sum(v for r in rows for v in r if v is not None) == EXPECTED_TOTAL

    def test_gap_count(self):
        _, rows = coverage_matrix()
        gaps = sum(1 for r in rows for v in r if v is None)
        assert gaps == 40 - EXPECTED_SPLITS  # 22 empty cells


class TestResolve:
    def test_none_gives_everything(self):
        assert len(resolve(None)) == EXPECTED_SPLITS
        assert len(resolve(["all"])) == EXPECTED_SPLITS

    def test_by_task(self):
        assert {s.name for s in resolve(["mmlu"])} == {
            "mmlu_hineng", "mmlu_beneng", "mmlu_mareng", "mmlu_tameng"
        }

    def test_by_language(self):
        tam = resolve(["tam"])
        assert {s.name for s in tam} == {"sa_tameng", "mmlu_tameng"}

    def test_mixed_selectors_dedupe(self):
        got = resolve(["tam", "sa_tameng", "mmlu"])
        assert len(got) == len({s.name for s in got})
        assert "sa_tameng" in {s.name for s in got}

    def test_canonical_ordering(self):
        specs = resolve(None)
        order = [(TASK_ORDER.index(s.task), LANGUAGE_ORDER.index(s.lang)) for s in specs]
        assert order == sorted(order)

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown dataset"):
            resolve(["nepali"])


class TestModelConfigs:
    def test_all_configs_load(self):
        defaults = load_defaults()
        names = available_models()
        assert names, "no model cards found in configs/models/"
        for name in names:
            cfg = load_model_config(name, defaults=defaults)
            assert cfg.hf_id
            assert cfg.backend in ("openai", "vllm", "hf", "echo")

    def test_task_overrides_only_touch_sampling_fields(self):
        from cmb_indic.backends.base import Backend

        defaults = load_defaults()
        for name in available_models():
            cfg = load_model_config(name, defaults=defaults)
            touched = {k for over in cfg.task_overrides.values() for k in over}
            clash = touched & set(Backend.LOAD_TIME_FIELDS)
            assert not clash, f"{name}: task_overrides may not set load-time fields {clash}"

    def test_every_task_has_a_token_budget(self):
        defaults = load_defaults()
        cfg = load_model_config("echo-debug", defaults=defaults)
        for task in TASK_ORDER:
            assert cfg.for_task(task).max_tokens > 0

    def test_substituted_models_explain_themselves(self):
        defaults = load_defaults()
        for name in available_models():
            cfg = load_model_config(name, defaults=defaults)
            if cfg.status == "substituted":
                assert "SUBSTITUTION" in cfg.notes


@pytest.mark.network
class TestAgainstHub:
    """Verify the registry still matches the live dataset. Run with `-m network`."""

    def test_row_counts_match_hub(self, tmp_path):
        import pandas as pd

        from cmb_indic.data import ensure_downloaded

        mismatches = []
        for spec in DATASETS.values():
            path = ensure_downloaded(spec, tmp_path)
            n = len(pd.read_csv(path))
            if n != spec.n_rows:
                mismatches.append(f"{spec.name}: hub={n} registry={spec.n_rows}")
        assert not mismatches, "registry is stale: " + "; ".join(mismatches)
