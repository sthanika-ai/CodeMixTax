"""Shared fixtures.

The synthetic split fixtures let the end-to-end tests run with no network and no
model, while still exercising the real prompt templates from `prompt.json`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cmb_indic.registry import DATASETS


@pytest.fixture
def repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


@pytest.fixture
def prompt_file(repo_root):
    path = repo_root / "prompt.json"
    if not path.exists():
        pytest.skip("prompt.json not present")
    return str(path)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.fixture
def fake_splits(tmp_path):
    """Write tiny CSVs that mimic the real column layouts, and return the data dir.

    Row counts deliberately differ from the registry, so anything that loads these
    also proves the row-count warning is non-fatal.
    """
    data = {
        "lid_hineng": _frame(
            [
                {
                    "index": i,
                    "sentence": "Ladke Ne Call Ki",
                    "tokens": "['Ladke', 'Ne', 'Call', 'Ki']",
                    "answer": "['lang2', 'lang2', 'lang1', 'lang2']",
                }
                for i in range(4)
            ]
        ),
        "pos_hineng": _frame(
            [
                {
                    "index": i,
                    "sentence": "mre Bharat k",
                    "tokens": "['mre', 'Bharat', 'k']",
                    "answer": "['ADJ', 'PROPN', 'ADP']",
                }
                for i in range(3)
            ]
        ),
        "ner_hineng": _frame(
            [
                {
                    "index": i,
                    "sentence": "Arvind Delhi se",
                    "tokens": "['Arvind', 'Delhi', 'se']",
                    "answer": "['B-PERSON', 'B-PLACE', 'O']",
                }
                for i in range(3)
            ]
        ),
        "sa_hineng": _frame(
            [{"index": i, "sentence": "kya baat hai", "answer": "positive"} for i in range(3)]
        ),
        "sa_beneng": _frame(
            [{"index": i, "sentence": "eta kharap", "answer": "N", "cm": 1} for i in range(3)]
        ),
        "sa_tameng": _frame(
            [{"index": i, "sentence": "super trailer", "answer": "Positive"} for i in range(3)]
        ),
        "sa_maleng": _frame(
            [{"index": i, "sentence": "Love u ikka", "answer": "Positive"} for i in range(3)]
        ),
        "sa_mareng": _frame(
            [{"index": i, "sentence": "kya pahije", "answer": "positive"} for i in range(3)]
        ),
        "mt_hineng_eng": _frame(
            [{"index": i, "sentence": "Kya logon ko pata tha", "answer": "Did people know"} for i in range(3)]
        ),
        "mt_beneng_eng": _frame(
            [{"index": i, "sentence": "amar kache course", "answer": "I have the course"} for i in range(3)]
        ),
        "mt_mareng_eng": _frame(
            [{"index": i, "sentence": "diagram madhil", "answer": "Observe the diagram"} for i in range(3)]
        ),
        "gsm8k_hineng": _frame(
            [
                {
                    "index": i,
                    "sentence": "Ram ke paas 5 apples hain, 3 kharide. Total?",
                    "cot": "5 + 3 = 8",
                    "answer": "8",
                    "tokens": "['Ram']",
                    "lids": "['lang2']",
                }
                for i in range(3)
            ]
        ),
    }
    # MCQ splits share a layout.
    for name, letters in (
        ("mmlu_hineng", "ABCD"), ("mmlu_beneng", "ABCD"),
        ("mmlu_mareng", "ABCD"), ("mmlu_tameng", "ABCD"),
        ("truthfulqa_hineng", "ABCD"),
    ):
        data[name] = _frame(
            [
                {
                    "index": i,
                    "id": f"q{i}",
                    # Distinct per row: identical sentences would collapse the
                    # few-shot dedupe and mask shot-count bugs.
                    "sentence": f"prashna {i}?\n(A): one\n(B): two\n(C): three\n(D): four",
                    "answer": letters[i % len(letters)],
                    "tokens": "['prashna']",
                    "lids": "['lang2']",
                }
                for i in range(4)
            ]
        )
    data["lid_mareng"] = _frame(
        [
            {
                "index": i,
                "sentence": "aamche hey fan",
                "tokens": "['aamche', 'hey', 'fan']",
                "answer": "['MAR', 'ENG', 'ENG']",
            }
            for i in range(3)
        ]
    )

    for name, frame in data.items():
        spec = DATASETS[name]
        dest = tmp_path / spec.hf_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dest, index=False)
    return tmp_path
