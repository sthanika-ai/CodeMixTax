"""Answer position in a reply is not fixed, so extraction must key on reply shape.

Two shapes occur in this study and need opposite handling:

  answer-FIRST   " C\\n\\nD\\n\\nD\\n\\n..."                Sarvam-1 (base), Llama 3.1,
                 "(C): प्रसारण।"                          Mistral, Phi-4, Gemma 3
  answer-LAST    "## Step 1: option (A)... best answer is B."   Llama 4 Scout, Sarvam-M,
                                                               gpt-oss

Taking the first anchored match everywhere mis-scored 185 of 513 verbose Llama 4
English replies (14.65 accuracy points): the reasoning enumerates options with the
same keywords the anchor looks for, so `Option (A) is incorrect` won over
`The best answer is B` 1,140 characters later. Taking the last match everywhere
instead cost Sarvam-1 8.69 points by reading its repetition tail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmb_indic.extract import extract_prediction

MCQ = dict(labels=tuple("ABCD"))

ANSWER_FIRST = [
    ("bare letter", "B", "B"),
    ("parenthesised", "(B)", "B"),
    ("gemma with newline", "(A)\n", "A"),
    ("letter plus option text", "(C): प्रसारण।", "C"),
    ("letter then english gloss", "(B): Print.", "B"),
    # Sarvam-1 base: answers, then degenerates into repetition. The tail must not win.
    ("repetition tail", " C\n\nD\n\nD\n\nD\n\nD\n\nD", "C"),
    ("repetition with prose", " C\n\nD\n\nA group of people who are trying", "C"),
]

ANSWER_LAST = [
    ("llama4 enumerate then conclude",
     "## Step 1: Analyze\n- (A) is abiotic.\n- (D) involves animals.\n\nThe best answer is D.",
     "D"),
    # The exact failure: four keyword-anchored letters before the verdict.
    ("llama4 option-by-option",
     "Option (A) is incorrect because x. Option (B) suggests y. "
     "Option (C) is correct in stating z. Option (D) mentions w.\n\nThe best answer is B",
     "B"),
    ("sarvam-m unmatched close tag",
     "Okay, the options are (A) digital, (B) print. Print is clearer.</think>\n\n(B)", "B"),
    ("gpt-oss bare harmony",
     "analysisOptions: (A): foo. (B): bar. So answer: (B).assistantfinal(B) bar", "B"),
    ("verdict then trailing aside", "The answer is B. Note (A) was tempting.", "B"),
]


@pytest.mark.parametrize("name,text,expected", ANSWER_FIRST)
def test_answer_first_shapes(name, text, expected):
    assert extract_prediction("mmlu", text, **MCQ) == expected, name


@pytest.mark.parametrize("name,text,expected", ANSWER_LAST)
def test_answer_last_shapes(name, text, expected):
    assert extract_prediction("mmlu", text, **MCQ) == expected, name


def test_enumeration_without_a_verdict_falls_back_to_last():
    """No verdict at all: the last mentioned option is the best available guess."""
    assert extract_prediction("mmlu", "Options are (A) x and (C) y", **MCQ) == "C"


def test_truthfulqa_letters_past_d():
    """TruthfulQA keeps every mc1 choice, so option letters run past D."""
    labels = tuple("ABCDEFGHIJKLMN")
    assert extract_prediction("truthfulqa", "(G)", labels=labels) == "G"
    assert extract_prediction("truthfulqa", "The best answer is J.", labels=labels) == "J"


def test_paper_mode_keeps_the_upstream_defect():
    """`paper` reproduces upstream: first bare A-D anywhere, however wrong."""
    txt = "## Step 1: Digital is one option.\n\nThe best answer is B."
    assert extract_prediction("mmlu", txt, mode="paper", **MCQ) == "D"  # from "Digital"
    assert extract_prediction("mmlu", txt, **MCQ) == "B"                # robust gets it right
