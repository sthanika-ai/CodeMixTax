"""gpt-oss harmony parsing, both the pipe-delimited and bare-word renderings.

vLLM 0.10.1 emits the bare form (`analysis... assistantfinal...`) with no pipe
markers. Handling only the pipe form meant the reasoning trace survived into the
extractor, which then read an option letter out of the model's thinking -- and a
trace cut off by the token budget produced a confident answer the model never gave.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmb_indic.extract import extract_prediction, strip_reasoning


def test_bare_harmony_keeps_only_the_final_channel():
    txt = "analysisOptions: (A): foo. (B): bar. So answer: (B).assistantfinal(B) bar"
    assert strip_reasoning(txt) == "(B) bar"
    assert extract_prediction("mmlu", txt) == "B"


def test_truncated_reasoning_is_unparsed_not_a_guess():
    """The trace enumerates (A) but never concludes; that must not become an answer."""
    txt = "analysisOptions: (A): foo. (B): bar. Let us consider (A) first and"
    assert strip_reasoning(txt) == ""
    assert extract_prediction("mmlu", txt) == "unk"


def test_pipe_harmony_still_works():
    txt = "<|channel|>analysis<|message|>hmm (A)<|channel|>final<|message|>(C)"
    assert extract_prediction("mmlu", txt) == "C"


def test_plain_and_think_tag_replies_unaffected():
    assert extract_prediction("mmlu", "(D)") == "D"
    assert extract_prediction("mmlu", "<think>maybe (A)</think>(B)") == "B"


def test_gsm8k_final_channel_number():
    txt = "analysisLet me compute 2+2 which is 4 maybe 5.assistantfinalFinal answer: 4"
    assert extract_prediction("gsm8k", txt) == "4"


def test_unmatched_closing_think_tag():
    """Sarvam-M reasons in bare prose and closes with `</think>`, no opening tag."""
    txt = "Okay, the options are (A) digital, (B) print. Print is clearer.</think>\n\n(B)"
    assert strip_reasoning(txt) == "(B)"
    assert extract_prediction("mmlu", txt) == "B"


def test_real_think_pair_still_wins():
    txt = "<think>considering (A)</think>(C)"
    assert extract_prediction("mmlu", txt) == "C"


def test_close_tag_mentioned_inside_reasoning():
    """Only the LAST closing tag splits, so a mention mid-trace cannot cut it early."""
    txt = "I should emit </think> when done. Options (A),(B).</think>\n\n(A)"
    assert extract_prediction("mmlu", txt) == "A"
