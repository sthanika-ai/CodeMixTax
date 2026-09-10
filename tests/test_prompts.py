"""Prompt construction: shape, determinism, and fidelity to the upstream wording."""

from __future__ import annotations

import pytest

from cmb_indic.data import load_split
from cmb_indic.prompts import build_prompts, prompt_fingerprint, system_text
from cmb_indic.registry import DATASETS


def _load(name, fake_splits, **kw):
    return load_split(DATASETS[name], fake_splits, download=False, **kw)


class TestShape:
    def test_chat_prompt_has_system_and_user(self, fake_splits, prompt_file):
        spec = DATASETS["lid_hineng"]
        split = _load("lid_hineng", fake_splits)
        prompts = build_prompts(spec, split.frame, prompt_file=prompt_file)
        assert len(prompts) == len(split)
        first = prompts[0]
        assert [m["role"] for m in first] == ["system", "user"]
        assert "Tokenized sentence:" in first[1]["content"]

    def test_completion_style_is_a_string(self, fake_splits, prompt_file):
        spec = DATASETS["sa_hineng"]
        split = _load("sa_hineng", fake_splits)
        prompts = build_prompts(spec, split.frame, prompt_file=prompt_file, style="completion")
        assert all(isinstance(p, str) for p in prompts)
        assert "Sentence:" in prompts[0]

    def test_system_text_comes_from_upstream_file(self, prompt_file):
        # Fidelity check against the published prompt, not a paraphrase of it.
        text = system_text("lid_hineng", prompt_file)
        assert "language identification" in text
        assert "lang1" in text and "lang2" in text

    def test_marathi_lid_uses_its_own_label_names(self, prompt_file):
        text = system_text("lid_mareng", prompt_file)
        assert "MAR" in text and "ENG" in text

    def test_bengali_sa_prompt_is_offensive_detection(self, prompt_file):
        text = system_text("sa_beneng", prompt_file)
        assert "Offensive" in text

    def test_invalid_style_rejected(self, fake_splits, prompt_file):
        with pytest.raises(ValueError, match="style must be"):
            build_prompts(
                DATASETS["sa_hineng"],
                _load("sa_hineng", fake_splits).frame,
                prompt_file=prompt_file,
                style="banana",
            )


class TestFewShot:
    def test_zero_shot_has_one_user_turn(self, fake_splits, prompt_file):
        spec = DATASETS["mmlu_hineng"]
        split = _load("mmlu_hineng", fake_splits)
        prompts = build_prompts(spec, split.frame, prompt_file=prompt_file, shots=0)
        assert len(prompts[0]) == 2  # system + question

    def test_shots_add_user_assistant_pairs(self, fake_splits, prompt_file):
        spec = DATASETS["mmlu_hineng"]
        split = _load("mmlu_hineng", fake_splits)
        prompts = build_prompts(spec, split.frame, prompt_file=prompt_file, shots=2)
        roles = [m["role"] for m in prompts[0]]
        assert roles == ["system", "user", "assistant", "user", "assistant", "user"]

    def test_question_is_never_its_own_shot(self, fake_splits, prompt_file):
        spec = DATASETS["mmlu_hineng"]
        split = _load("mmlu_hineng", fake_splits)
        prompts = build_prompts(spec, split.frame, prompt_file=prompt_file, shots=2)
        for prompt, (_, row) in zip(prompts, split.frame.iterrows(), strict=True):
            shots = [m["content"] for m in prompt[1:-1] if m["role"] == "user"]
            # The final turn is the question; earlier user turns are exemplars.
            assert prompt[-1]["content"].startswith(str(row["sentence"])[:20])
            for shot in shots:
                assert shot != prompt[-1]["content"]

    def test_gsm8k_shots_include_chain_of_thought(self, fake_splits, prompt_file):
        spec = DATASETS["gsm8k_hineng"]
        split = _load("gsm8k_hineng", fake_splits)
        prompts = build_prompts(spec, split.frame, prompt_file=prompt_file, shots=1)
        assistant = [m for m in prompts[0] if m["role"] == "assistant"]
        assert assistant and "Solution:" in assistant[0]["content"]
        assert "Final Answer:" in assistant[0]["content"]

    def test_gsm8k_system_prompt_carries_answer_format(self, fake_splits, prompt_file):
        # The extractor's "Final answer:" anchor depends on this instruction.
        spec = DATASETS["gsm8k_hineng"]
        split = _load("gsm8k_hineng", fake_splits)
        prompts = build_prompts(spec, split.frame, prompt_file=prompt_file, shots=0)
        assert "Final answer:" in prompts[0][0]["content"]


class TestDeterminism:
    def test_same_seed_gives_identical_prompts(self, fake_splits, prompt_file):
        spec = DATASETS["mmlu_hineng"]
        split = _load("mmlu_hineng", fake_splits)
        a = build_prompts(spec, split.frame, prompt_file=prompt_file, shots=2, seed=7)
        b = build_prompts(spec, split.frame, prompt_file=prompt_file, shots=2, seed=7)
        assert a == b
        assert prompt_fingerprint(a) == prompt_fingerprint(b)

    def test_different_seed_changes_shots(self, fake_splits, prompt_file):
        spec = DATASETS["mmlu_hineng"]
        split = _load("mmlu_hineng", fake_splits)
        a = build_prompts(spec, split.frame, prompt_file=prompt_file, shots=2, seed=1)
        b = build_prompts(spec, split.frame, prompt_file=prompt_file, shots=2, seed=999)
        # Zero-shot prompts would be identical; with shots they must differ.
        assert prompt_fingerprint(a) != prompt_fingerprint(b)

    def test_fingerprint_is_stable_across_processes(self, fake_splits, prompt_file):
        # Hash of content only -- no ids, no addresses, no time.
        spec = DATASETS["sa_hineng"]
        split = _load("sa_hineng", fake_splits)
        prompts = build_prompts(spec, split.frame, prompt_file=prompt_file)
        assert prompt_fingerprint(prompts) == prompt_fingerprint(list(prompts))

    def test_shots_do_not_hang_when_filter_starves(self, fake_splits, prompt_file):
        # Asking for more shots than the pool can supply must terminate.
        spec = DATASETS["mmlu_hineng"]
        split = _load("mmlu_hineng", fake_splits)
        prompts = build_prompts(spec, split.frame, prompt_file=prompt_file, shots=10)
        assert len(prompts) == len(split)
