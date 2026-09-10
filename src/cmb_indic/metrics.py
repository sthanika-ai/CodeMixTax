"""Metrics -- faithful ports of the upstream CodeMixBench implementations.

Provenance. These are the same computations as `compute_token_label_Metric`,
`compute_sentence_label_Metric` and `compute_BLEU` in the upstream `utils.py`,
refactored so they:

  * return plain dicts instead of writing files and mutating DataFrames,
  * raise instead of calling `exit()` on malformed input,
  * add corpus-level aggregates alongside the upstream per-sentence averages,
  * report coverage diagnostics (format-error and fallback rates) so a high
    score cannot silently hide a large fraction of unparseable generations.

The upstream headline numbers are preserved verbatim under the same keys, so
results from this pipeline stay comparable to the published paper:

  token-level tasks   accuracy, micro_f1, macro_f1, weighted_f1
                      = mean over sentences of the per-sentence sklearn score
  sentence-level      accuracy, micro_f1, macro_f1, weighted_f1
                      = corpus-level sklearn score over all rows
  MT                  bleu = sacreBLEU corpus score

See docs/DEVIATIONS.md for the full list of differences from upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sacrebleu.metrics import BLEU, CHRF
from sklearn.metrics import accuracy_score, f1_score

#: Label assigned when a token could not be recovered from the generation.
#: Matches upstream's use of the literal string 'unk' as a fallback tag.
UNK = "unk"


@dataclass
class MetricResult:
    """Scores plus the diagnostics needed to interpret them."""

    metric_kind: str
    scores: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_kind": self.metric_kind,
            "scores": self.scores,
            "diagnostics": self.diagnostics,
        }


# --------------------------------------------------------------------------- #
# Token-level tasks: LID, POS, NER
# --------------------------------------------------------------------------- #
def token_label_metrics(
    gold: list[list[str]],
    pred: list[list[str]],
    *,
    n_format_errors: int = 0,
    labels: tuple[str, ...] | None = None,
) -> MetricResult:
    """Score per-token sequence labelling.

    Upstream semantics: compute accuracy / micro / macro / weighted F1 *per
    sentence* with sklearn, then take the unweighted mean over sentences. That
    weights a 3-token sentence as heavily as a 40-token one, which is unusual
    but is what the published numbers use -- so it is what `accuracy` and the
    `*_f1` keys report here.

    `corpus_*` keys additionally flatten every token into one pool, which is the
    conventional token-level score and is the fairer number for tasks with very
    uneven sentence lengths. Both are written to the report.

    Args:
        gold: gold label sequences, one list per sentence.
        pred: predicted label sequences, aligned 1:1 and length-matched to gold.
        n_format_errors: how many generations could not be parsed at all (these
            are expected to appear in `pred` as all-UNK sequences).
        labels: the split's closed label set, used to count hallucinated tags.

    Raises:
        ValueError: on any length mismatch. Callers must align before scoring;
            `extract.extract_token_labels` guarantees this by padding with UNK.
    """
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred sentence count differs: {len(gold)} vs {len(pred)}")
    if not gold:
        raise ValueError("no rows to score")

    acc, micro, macro, weighted = [], [], [], []
    flat_gold: list[str] = []
    flat_pred: list[str] = []

    for i, (g, p) in enumerate(zip(gold, pred, strict=True)):
        if len(g) != len(p):
            raise ValueError(
                f"row {i}: gold has {len(g)} tokens but pred has {len(p)}. "
                "Predictions must be aligned/padded before scoring."
            )
        if not g:
            continue
        acc.append(accuracy_score(g, p) * 100)
        micro.append(f1_score(g, p, average="micro", zero_division=0) * 100)
        macro.append(f1_score(g, p, average="macro", zero_division=0) * 100)
        weighted.append(f1_score(g, p, average="weighted", zero_division=0) * 100)
        flat_gold.extend(g)
        flat_pred.extend(p)

    if not acc:
        raise ValueError("every row was empty; nothing to score")

    n_tok = len(flat_gold)
    unk_tokens = sum(1 for t in flat_pred if t == UNK)
    valid = set(labels or ())
    off_label = sum(1 for t in flat_pred if valid and t not in valid) if valid else 0

    return MetricResult(
        metric_kind="token_label",
        scores={
            # Upstream headline numbers (mean over sentences).
            "accuracy": _mean(acc),
            "micro_f1": _mean(micro),
            "macro_f1": _mean(macro),
            "weighted_f1": _mean(weighted),
            # Conventional token-pooled numbers.
            "corpus_accuracy": accuracy_score(flat_gold, flat_pred) * 100,
            "corpus_micro_f1": f1_score(flat_gold, flat_pred, average="micro", zero_division=0) * 100,
            "corpus_macro_f1": f1_score(flat_gold, flat_pred, average="macro", zero_division=0) * 100,
            "corpus_weighted_f1": f1_score(flat_gold, flat_pred, average="weighted", zero_division=0) * 100,
        },
        diagnostics={
            "n_rows": len(acc),
            "n_tokens": n_tok,
            "n_format_errors": n_format_errors,
            "format_error_rate": 100.0 * n_format_errors / len(gold),
            "n_unk_tokens": unk_tokens,
            "unk_token_rate": 100.0 * unk_tokens / n_tok if n_tok else 0.0,
            "n_off_label_tokens": off_label,
            "off_label_token_rate": 100.0 * off_label / n_tok if n_tok else 0.0,
        },
    )


# --------------------------------------------------------------------------- #
# Sentence-level tasks: SA, MMLU, GSM8K, TruthfulQA
# --------------------------------------------------------------------------- #
def sentence_label_metrics(
    gold: list[str],
    pred: list[str],
    *,
    labels: tuple[str, ...] | None = None,
    n_format_errors: int = 0,
) -> MetricResult:
    """Score single-label-per-row tasks, corpus-level (upstream semantics).

    Predictions that could not be parsed should arrive as UNK; they count as
    wrong (never dropped), which keeps accuracy an honest denominator. The
    `unparsed_rate` diagnostic tells you how much of the error is extraction
    failure rather than genuine mistakes -- important for base models and for
    reasoning models that overrun `max_tokens`.
    """
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred row count differs: {len(gold)} vs {len(pred)}")
    if not gold:
        raise ValueError("no rows to score")

    g = [str(x).strip() for x in gold]
    p = [str(x).strip() for x in pred]

    # Restrict F1 averaging to the gold label space where it is known, so that
    # UNK predictions do not create a spurious extra class that drags macro-F1.
    f1_labels = sorted(set(g) | (set(labels) if labels else set()))

    unparsed = sum(1 for x in p if x == UNK)
    majority = max((g.count(c) for c in set(g)), default=0)

    return MetricResult(
        metric_kind="sentence_label",
        scores={
            "accuracy": accuracy_score(g, p) * 100,
            "micro_f1": f1_score(g, p, average="micro", labels=f1_labels, zero_division=0) * 100,
            "macro_f1": f1_score(g, p, average="macro", labels=f1_labels, zero_division=0) * 100,
            "weighted_f1": f1_score(g, p, average="weighted", labels=f1_labels, zero_division=0) * 100,
        },
        diagnostics={
            "n_rows": len(g),
            "n_unparsed": unparsed,
            "unparsed_rate": 100.0 * unparsed / len(g),
            "n_format_errors": n_format_errors,
            "majority_baseline": 100.0 * majority / len(g),
            "n_gold_classes": len(set(g)),
            "pred_class_distribution": _distribution(p),
        },
    )


# --------------------------------------------------------------------------- #
# MT
# --------------------------------------------------------------------------- #
def bleu_metrics(
    refs: list[str],
    hyps: list[str],
    *,
    tokenize: str | None = "13a",
) -> MetricResult:
    """Corpus sacreBLEU (upstream metric) plus chrF++ as a secondary signal.

    Upstream selects `tokenize='zh'` for Chinese/Mandarin targets and the
    default otherwise. Every Indic MT split targets English, so `13a` is used
    throughout; the parameter is kept so the same code can score the non-Indic
    splits if the suite is ever widened.

    chrF++ is added because BLEU is brittle on short, noisy, transliterated
    references -- which describes much of the code-mixed MT data. BLEU stays the
    headline number for comparability.
    """
    if len(refs) != len(hyps):
        raise ValueError(f"ref/hyp count differs: {len(refs)} vs {len(hyps)}")
    if not refs:
        raise ValueError("no rows to score")

    r = [str(x) if x is not None else "" for x in refs]
    h = [str(x) if x is not None else "" for x in hyps]

    bleu = BLEU(tokenize=tokenize) if tokenize else BLEU()
    bleu_score = bleu.corpus_score(h, [r])
    chrf_score = CHRF(word_order=2).corpus_score(h, [r])  # chrF++

    empty = sum(1 for x in h if not x.strip())
    return MetricResult(
        metric_kind="bleu",
        scores={
            "bleu": bleu_score.score,
            "chrf2": chrf_score.score,
        },
        diagnostics={
            "n_rows": len(r),
            "bleu_signature": str(bleu.get_signature()),
            "bleu_detail": bleu_score.format(),
            "tokenize": tokenize or "default",
            "n_empty_hypotheses": empty,
            "empty_hypothesis_rate": 100.0 * empty / len(h),
            "mean_hyp_chars": _mean([len(x) for x in h]),
            "mean_ref_chars": _mean([len(x) for x in r]),
        },
    )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def score(
    metric_kind: str,
    gold: list,
    pred: list,
    *,
    labels: tuple[str, ...] | None = None,
    bleu_tokenize: str | None = "13a",
    n_format_errors: int = 0,
) -> MetricResult:
    """Score by metric kind. Called by the runner; see `registry.DatasetSpec.metric`."""
    if metric_kind == "token_label":
        return token_label_metrics(
            gold, pred, n_format_errors=n_format_errors, labels=labels
        )
    if metric_kind == "sentence_label":
        return sentence_label_metrics(
            gold, pred, labels=labels, n_format_errors=n_format_errors
        )
    if metric_kind == "bleu":
        return bleu_metrics(gold, pred, tokenize=bleu_tokenize)
    raise ValueError(f"unknown metric kind: {metric_kind!r}")


# --------------------------------------------------------------------------- #
def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _distribution(xs: list[str], top: int = 8) -> dict[str, int]:
    counts: dict[str, int] = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:top])
