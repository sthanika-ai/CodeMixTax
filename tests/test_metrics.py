"""Metric tests, including a check that the port matches upstream semantics."""

from __future__ import annotations

import pytest

from cmb_indic.metrics import (
    UNK,
    bleu_metrics,
    score,
    sentence_label_metrics,
    token_label_metrics,
)


class TestTokenLabel:
    def test_perfect(self):
        gold = [["A", "B"], ["A", "A", "B"]]
        res = token_label_metrics(gold, [list(x) for x in gold])
        assert res.scores["accuracy"] == pytest.approx(100.0)
        assert res.scores["corpus_accuracy"] == pytest.approx(100.0)

    def test_upstream_averages_per_sentence_not_per_token(self):
        # Sentence 1: 1/1 correct. Sentence 2: 1/3 correct.
        # Upstream mean-over-sentences = (100 + 33.33)/2 = 66.67
        # Token-pooled                 = 2/4 = 50.0
        gold = [["A"], ["A", "A", "A"]]
        pred = [["A"], ["A", "B", "B"]]
        res = token_label_metrics(gold, pred)
        assert res.scores["accuracy"] == pytest.approx(66.667, abs=1e-2)
        assert res.scores["corpus_accuracy"] == pytest.approx(50.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="tokens but pred"):
            token_label_metrics([["A", "B"]], [["A"]])

    def test_row_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="sentence count"):
            token_label_metrics([["A"]], [["A"], ["B"]])

    def test_unk_and_off_label_diagnostics(self):
        gold = [["A", "B"]]
        pred = [[UNK, "ZZZ"]]
        res = token_label_metrics(gold, pred, labels=("A", "B"), n_format_errors=1)
        d = res.diagnostics
        assert d["n_unk_tokens"] == 1
        assert d["unk_token_rate"] == pytest.approx(50.0)
        assert d["n_off_label_tokens"] == 2  # UNK and ZZZ are both outside the set
        assert d["format_error_rate"] == pytest.approx(100.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no rows"):
            token_label_metrics([], [])


class TestSentenceLabel:
    def test_accuracy_and_baseline(self):
        gold = ["A", "A", "A", "B"]
        pred = ["A", "A", "B", "B"]
        res = sentence_label_metrics(gold, pred, labels=("A", "B"))
        assert res.scores["accuracy"] == pytest.approx(75.0)
        assert res.diagnostics["majority_baseline"] == pytest.approx(75.0)

    def test_unparsed_counts_as_wrong_never_dropped(self):
        gold = ["A", "B"]
        pred = ["A", UNK]
        res = sentence_label_metrics(gold, pred, labels=("A", "B"))
        assert res.scores["accuracy"] == pytest.approx(50.0)
        assert res.diagnostics["n_rows"] == 2  # denominator keeps the failed row
        assert res.diagnostics["unparsed_rate"] == pytest.approx(50.0)

    def test_whitespace_insensitive(self):
        res = sentence_label_metrics(["A "], [" A"], labels=("A",))
        assert res.scores["accuracy"] == pytest.approx(100.0)

    def test_unk_does_not_create_extra_macro_class(self):
        # With labels given, macro-F1 averages over the gold label space only.
        gold = ["A", "B"]
        res = sentence_label_metrics(gold, [UNK, UNK], labels=("A", "B"))
        assert res.scores["macro_f1"] == pytest.approx(0.0)


class TestBLEU:
    def test_identical_is_100(self):
        refs = ["I have the course overview.", "Observe the diagram."]
        res = bleu_metrics(refs, list(refs))
        assert res.scores["bleu"] == pytest.approx(100.0, abs=1e-6)
        assert res.scores["chrf2"] == pytest.approx(100.0, abs=1e-6)

    def test_signature_recorded(self):
        res = bleu_metrics(["hello world"], ["hello world"])
        assert "13a" in res.diagnostics["bleu_signature"]

    def test_empty_hypotheses_tracked(self):
        res = bleu_metrics(["a b c", "d e f"], ["a b c", ""])
        assert res.diagnostics["n_empty_hypotheses"] == 1
        assert res.diagnostics["empty_hypothesis_rate"] == pytest.approx(50.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="ref/hyp count"):
            bleu_metrics(["a"], ["a", "b"])


class TestDispatch:
    def test_dispatch_matches_direct_calls(self):
        assert score("sentence_label", ["A"], ["A"]).scores["accuracy"] == 100.0
        assert score("token_label", [["A"]], [["A"]]).scores["accuracy"] == 100.0
        # Needs >= 4 tokens: corpus BLEU uses up to 4-grams with no smoothing, so a
        # shorter sentence scores 0 even when it matches the reference exactly.
        assert score("bleu", ["a b c d"], ["a b c d"]).scores["bleu"] == pytest.approx(100.0)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown metric kind"):
            score("nope", [], [])
