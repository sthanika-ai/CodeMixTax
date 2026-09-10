"""Extraction tests.

These are the highest-value tests in the repo: an extraction bug does not crash,
it just silently produces a wrong benchmark number. Each case below is a reply
shape that a real local model actually emits.
"""

from __future__ import annotations

import pytest

from cmb_indic.extract import (
    extract_gsm8k,
    extract_label,
    extract_mcq,
    extract_token_labels,
    extract_translation,
    normalize_number,
    parse_token_list,
    strip_reasoning,
)
from cmb_indic.metrics import UNK

ABCD = ("A", "B", "C", "D")


# --------------------------------------------------------------------------- #
# Reasoning stripping
# --------------------------------------------------------------------------- #
class TestStripReasoning:
    def test_removes_think_block(self):
        text = "<think>The options are A, B, C. A looks wrong.</think>\nAnswer: D"
        assert strip_reasoning(text) == "Answer: D"

    def test_unclosed_think_leaves_nothing(self):
        # Ran out of tokens mid-thought: must not leak a letter from the trace.
        text = "<think>Let me consider option A first, then B"
        assert strip_reasoning(text) == ""

    def test_harmony_final_channel_only(self):
        text = (
            "<|channel|>analysis<|message|>Could be A or C.<|end|>"
            "<|channel|>final<|message|>B<|return|>"
        )
        assert strip_reasoning(text).strip() == "B"

    def test_plain_text_untouched(self):
        assert strip_reasoning("positive") == "positive"

    # -- Gemma 4 channel scheme -------------------------------------------- #
    # Single-sided pipes (`<|channel>` ... `<channel|>`), unlike gpt-oss harmony.
    # Verified against the `strip_thinking` macro in Google's canonical template.
    def test_gemma4_thought_channel_removed(self):
        text = "<|channel>thought\nOption A seems plausible, but B fits better.\n<channel|>B"
        assert strip_reasoning(text) == "B"
        assert extract_mcq(text, ABCD) == "B"

    def test_gemma4_empty_preclosed_thought_is_harmless(self):
        # What the template emits by default (enable_thinking=false): an empty,
        # already-closed thought block followed by the real answer.
        assert strip_reasoning("<|channel>thought\n<channel|>Positive") == "Positive"

    def test_gemma4_unclosed_thought_yields_nothing(self):
        # Truncated mid-thought: must not leak a letter from the scratchpad.
        assert strip_reasoning("<|channel>thought\nMaybe A, maybe C") == ""
        assert extract_mcq("<|channel>thought\nMaybe A, maybe C", ABCD) == UNK

    def test_gemma4_turn_scaffolding_stripped(self):
        assert strip_reasoning("<|turn>model\nneutral<turn|>") == "model\nneutral"

    def test_gemma3_style_plain_answer(self):
        # Gemma 3 has no system role and no channels; output is plain text.
        assert extract_mcq("<end_of_turn>", ABCD) == UNK
        assert strip_reasoning("Positive") == "Positive"

    def test_reasoning_letters_do_not_win(self):
        # The regression this exists for: without stripping, "A" inside the
        # scratchpad would be picked instead of the real answer.
        text = "<think>A is tempting but wrong.</think>C"
        assert extract_mcq(text, ABCD) == "C"


# --------------------------------------------------------------------------- #
# MCQ
# --------------------------------------------------------------------------- #
class TestMCQ:
    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("B", "B"),
            ("(C)", "C"),
            ("Answer: B", "B"),
            ("answer: b", "B"),
            ("The answer is (D).", "D"),
            ("**A**", "A"),
            ("D) Outdoor", "D"),
            ("C. Broadcast", "C"),
            ("The correct option is C because it is cheaper.", "C"),
            ("Answer:\nB", "B"),
            ("```\nA\n```", "A"),
            ("I think the answer is B", "B"),
        ],
    )
    def test_robust_shapes(self, reply, expected):
        assert extract_mcq(reply, ABCD) == expected

    def test_unparseable_is_unk(self):
        assert extract_mcq("मुझे नहीं पता।", ABCD) == UNK
        assert extract_mcq("", ABCD) == UNK

    def test_paper_mode_reproduces_upstream_bug(self):
        # The documented upstream defect: "Answer" starts with a capital A, so the
        # first [ABCD] match is A rather than the model's actual choice B.
        assert extract_mcq("Answer: B", ABCD, mode="paper") == "A"
        # robust mode must not share the defect.
        assert extract_mcq("Answer: B", ABCD, mode="robust") == "B"

    def test_truthfulqa_extended_letters(self):
        letters = tuple("ABCDEFGHIJKLMN")
        assert extract_mcq("Answer: K", letters) == "K"
        assert extract_mcq("(H)", letters) == "H"

    def test_letter_inside_word_is_not_matched(self):
        # "Delhi" contains no standalone letter; the answer marker must win.
        assert extract_mcq("Delhi is the capital. Answer: C", ABCD) == "C"


# --------------------------------------------------------------------------- #
# GSM8K
# --------------------------------------------------------------------------- #
class TestGSM8K:
    def test_final_answer_marker(self):
        text = "Solution:\n5 apples times 3 = 15.\nFinal answer: 15"
        assert extract_gsm8k(text) == "15"

    def test_last_number_without_marker(self):
        assert extract_gsm8k("He buys 3 and 4 more, so 7 total.") == "7"

    def test_thousands_separator_and_currency(self):
        assert extract_gsm8k("Final answer: $1,250") == "1250"

    def test_trailing_decimal_normalised(self):
        assert extract_gsm8k("Final answer: 12.0") == "12"

    def test_hash_marker(self):
        assert extract_gsm8k("reasoning ...\n#### 42") == "42"

    def test_no_number_is_unk(self):
        assert extract_gsm8k("I cannot solve this.") == UNK

    def test_robust_mode_ignores_gold(self):
        # Label-leakage guard: passing gold must not change the answer.
        text = "Maybe 10, maybe 20, so 30."
        assert extract_gsm8k(text, gold="10") == extract_gsm8k(text) == "30"

    def test_paper_mode_leaks_gold(self):
        # Documented upstream behaviour, reproduced only in paper mode.
        text = "Final Answer: 10 or 20"
        assert extract_gsm8k(text, mode="paper", gold="20") == "20"
        assert extract_gsm8k(text, mode="paper", gold="99") == "10"

    def test_reasoning_stripped_before_numbers(self):
        text = "<think>Try 5. No, 6.</think>Final answer: 7"
        assert extract_gsm8k(text) == "7"

    @pytest.mark.parametrize(
        "raw,expected",
        [("1,234.0", "1234"), ("-0.50", "-0.5"), ("12.", "12"), ("007", "7"), ("", UNK)],
    )
    def test_normalize_number(self, raw, expected):
        assert normalize_number(raw) == expected


# --------------------------------------------------------------------------- #
# Sentence labels (SA)
# --------------------------------------------------------------------------- #
class TestLabels:
    THREE = ("positive", "negative", "neutral")
    FOUR = ("Positive", "Negative", "Neutral", "Mixed_feelings")
    OFFENSIVE = ("O", "N")

    def test_exact_match(self):
        assert extract_label("positive", self.THREE) == "positive"

    def test_case_normalised_to_gold_spelling(self):
        # Gold is lowercase here, so the returned value must be too.
        assert extract_label("Positive", self.THREE) == "positive"
        # And capitalised where gold is capitalised.
        assert extract_label("positive", self.FOUR) == "Positive"

    def test_mixed_feelings_not_shadowed(self):
        assert extract_label("Mixed_feelings", self.FOUR) == "Mixed_feelings"
        assert extract_label("mixed feelings", self.FOUR) == "Mixed_feelings"

    def test_answer_prefix(self):
        assert extract_label("Your answer: negative", self.THREE) == "negative"

    def test_offensive_single_letter_labels(self):
        assert extract_label("O", self.OFFENSIVE) == "O"
        assert extract_label("N", self.OFFENSIVE) == "N"
        assert extract_label("Answer: N", self.OFFENSIVE) == "N"

    def test_single_letter_labels_ignore_prose_letters(self):
        # A bare "o" inside a word must not count as label O.
        assert extract_label("I do not know", self.OFFENSIVE) == UNK

    def test_unknown_is_unk(self):
        assert extract_label("no idea", self.THREE) == UNK

    def test_sentiment_with_explanation_takes_conclusion(self):
        text = "The sentence is not negative, it is actually positive"
        assert extract_label(text, self.THREE) == "positive"


# --------------------------------------------------------------------------- #
# MT
# --------------------------------------------------------------------------- #
class TestTranslation:
    def test_plain(self):
        assert extract_translation("I have the course overview.") == "I have the course overview."

    def test_strips_preamble(self):
        text = "Sure! Here is the translation: I love you."
        assert extract_translation(text) == "I love you."

    def test_strips_translation_label(self):
        assert extract_translation("Translation: Observe the diagram.") == "Observe the diagram."

    def test_strips_quotes(self):
        assert extract_translation('"Did people know that?"') == "Did people know that?"

    def test_reasoning_stripped(self):
        assert extract_translation("<think>hmm</think>Hello there") == "Hello there"

    def test_empty_stays_empty_not_unk(self):
        # BLEU has no notion of UNK; an empty hypothesis is the honest encoding.
        assert extract_translation("") == ""


# --------------------------------------------------------------------------- #
# Token labels
# --------------------------------------------------------------------------- #
class TestTokenLabels:
    GOLD = ["Ladke", "Ne", "Call", "Ki"]
    LABELS = ("lang1", "lang2", "other")

    def test_json_list_of_dicts(self):
        reply = '[{"Ladke":"lang2"}, {"Ne":"lang2"}, {"Call":"lang1"}, {"Ki":"lang2"}]'
        res = extract_token_labels(reply, self.GOLD, valid_labels=self.LABELS)
        assert res.ok
        assert res.labels == ["lang2", "lang2", "lang1", "lang2"]
        assert res.aligned_by == "position"

    def test_python_literal_with_single_quotes(self):
        reply = "[{'Ladke': 'lang2'}, {'Ne': 'lang2'}, {'Call': 'lang1'}, {'Ki': 'lang2'}]"
        res = extract_token_labels(reply, self.GOLD, valid_labels=self.LABELS)
        assert res.labels == ["lang2", "lang2", "lang1", "lang2"]

    def test_wrapped_in_prose_and_fence(self):
        reply = (
            "Here you go:\n```json\n"
            '[{"Ladke":"lang2"},{"Ne":"lang2"},{"Call":"lang1"},{"Ki":"lang2"}]\n```\nHope that helps!'
        )
        res = extract_token_labels(reply, self.GOLD, valid_labels=self.LABELS)
        assert res.ok
        assert res.labels == ["lang2", "lang2", "lang1", "lang2"]

    def test_flat_dict(self):
        reply = '{"Ladke":"lang2","Ne":"lang2","Call":"lang1","Ki":"lang2"}'
        res = extract_token_labels(reply, self.GOLD, valid_labels=self.LABELS)
        assert res.labels == ["lang2", "lang2", "lang1", "lang2"]

    def test_always_padded_to_gold_length(self):
        reply = '[{"Ladke":"lang2"}]'
        res = extract_token_labels(reply, self.GOLD, valid_labels=self.LABELS)
        assert len(res.labels) == len(self.GOLD)
        assert res.labels[1:] == [UNK, UNK, UNK]

    def test_unparseable_gives_all_unk(self):
        res = extract_token_labels("I don't know how to do this", self.GOLD)
        assert not res.ok
        assert res.labels == [UNK] * 4
        assert res.n_missing == 4

    def test_repeated_tokens_positional(self):
        # The upstream by-name bug: with a repeated token, dict(zip(...)) keeps only
        # the last tag and both positions inherit it. Positional alignment is correct.
        gold = ["the", "cat", "the", "dog"]
        reply = '[{"the":"DET"},{"cat":"NOUN"},{"the":"X"},{"dog":"NOUN"}]'
        res = extract_token_labels(reply, gold, valid_labels=("DET", "NOUN", "X"))
        assert res.labels == ["DET", "NOUN", "X", "NOUN"]

        paper = extract_token_labels(reply, gold, mode="paper")
        assert paper.labels == ["X", "NOUN", "X", "NOUN"]  # first "the" mislabelled

    def test_case_snapped_to_label_set(self):
        reply = '[{"Ladke":"LANG2"},{"Ne":"lang2"},{"Call":"Lang1"},{"Ki":"lang2"}]'
        res = extract_token_labels(reply, self.GOLD, valid_labels=self.LABELS)
        assert res.labels == ["lang2", "lang2", "lang1", "lang2"]

    def test_bare_label_list_positional(self):
        reply = '["lang2","lang2","lang1","lang2"]'
        res = extract_token_labels(reply, self.GOLD, valid_labels=self.LABELS)
        assert res.labels == ["lang2", "lang2", "lang1", "lang2"]

    def test_parse_token_list_returns_none_on_garbage(self):
        assert parse_token_list("no structure at all here") is None
