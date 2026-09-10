"""Static registry of the CodeMixBench Indic evaluation surface.

Everything downstream (data loading, prompting, extraction, scoring, reporting)
keys off the tables in this module, so this is the single place to look to know
exactly what is being evaluated.

Scope: the five Indian languages of CodeMixBench -- Hindi, Bengali, Marathi,
Tamil, Malayalam. Nepali is excluded by design (it is Indo-Aryan but not an
Indian language); the remaining 13 non-Indian language pairs of the full
benchmark are out of scope.

Row counts were measured against the Hugging Face revision recorded in
`DATA_REVISION_NOTE` by reading each CSV with pandas (not `wc -l`: several
splits contain embedded newlines inside quoted fields). `tests/test_registry.py`
re-verifies them against the live Hub when run with `-m network`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

HF_DATASET_ID = "CodeMixBench/CodeMixBench"

DATA_REVISION_NOTE = (
    "Row counts verified against the Hub state of 2025-10-11 "
    "(lastModified on the dataset card). Re-run `cmb-indic validate --check-rows` "
    "if the dataset is updated upstream."
)

TaskName = Literal["lid", "pos", "ner", "sa", "mt", "mmlu", "gsm8k", "truthfulqa"]

#: How a task's predictions are scored. Drives dispatch in `metrics.py`.
MetricKind = Literal["token_label", "sentence_label", "bleu"]


# --------------------------------------------------------------------------- #
# Languages
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Language:
    code: str  # CodeMixBench 3-letter code as used in split names
    iso639_1: str  # for reporting / sacrebleu
    name: str
    family: str
    script: str


LANGUAGES: dict[str, Language] = {
    "hin": Language("hin", "hi", "Hindi", "Indo-Aryan", "Devanagari"),
    "ben": Language("ben", "bn", "Bengali", "Indo-Aryan", "Bengali"),
    "mar": Language("mar", "mr", "Marathi", "Indo-Aryan", "Devanagari"),
    "tam": Language("tam", "ta", "Tamil", "Dravidian", "Tamil"),
    "mal": Language("mal", "ml", "Malayalam", "Dravidian", "Malayalam"),
}

#: Canonical display order: Indo-Aryan then Dravidian, most-covered first.
LANGUAGE_ORDER: tuple[str, ...] = ("hin", "ben", "mar", "tam", "mal")


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    name: str
    pretty: str
    metric: MetricKind
    #: Column holding the model input.
    input_col: str
    #: Primary metric key to headline in reports.
    headline: str
    description: str


TASKS: dict[str, Task] = {
    "lid": Task(
        "lid", "LID", "token_label", "tokens", "accuracy",
        "Per-token language identification on a tokenised code-mixed sentence.",
    ),
    "pos": Task(
        "pos", "POS", "token_label", "tokens", "accuracy",
        "Per-token universal part-of-speech tagging.",
    ),
    "ner": Task(
        "ner", "NER", "token_label", "tokens", "accuracy",
        "Per-token named-entity recognition in BIO scheme (PERSON/PLACE/ORGANISATION).",
    ),
    "sa": Task(
        "sa", "SA", "sentence_label", "sentence", "accuracy",
        "Sentence-level classification. NOTE: the label space is not uniform across "
        "languages -- see DatasetSpec.labels and the sa_beneng caveat.",
    ),
    "mt": Task(
        "mt", "MT", "bleu", "sentence", "bleu",
        "Translation of a code-mixed sentence into English, scored with sacreBLEU.",
    ),
    "mmlu": Task(
        "mmlu", "MMLU", "sentence_label", "sentence", "accuracy",
        "4-way multiple-choice knowledge reasoning (code-mixed MMLU test set).",
    ),
    "gsm8k": Task(
        "gsm8k", "GSM8K", "sentence_label", "sentence", "accuracy",
        "Grade-school math word problems; exact match on the final numeric answer.",
    ),
    "truthfulqa": Task(
        "truthfulqa", "TruthfulQA", "sentence_label", "sentence", "accuracy",
        "Multiple-choice truthfulness, up to 14 options (A-N).",
    ),
}

TASK_ORDER: tuple[str, ...] = ("lid", "pos", "ner", "sa", "mt", "mmlu", "gsm8k", "truthfulqa")


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSpec:
    """One evaluable (task, language) cell of the benchmark."""

    name: str  # e.g. "sa_tameng" -- matches both the CSV stem and the prompt.json key
    task: str
    lang: str  # key into LANGUAGES
    n_rows: int  # measured, see DATA_REVISION_NOTE
    #: Closed label set, in canonical order. None for open-ended output (MT) or
    #: unbounded numeric output (GSM8K).
    labels: tuple[str, ...] | None = None
    #: sacreBLEU tokenizer for MT; all Indic MT splits target English -> "13a".
    bleu_tokenize: str | None = None
    #: Free-text caveats surfaced in `cmb-indic datasets` and the report.
    notes: str = ""
    #: Extra columns present in the CSV beyond index/input/answer.
    extra_cols: tuple[str, ...] = field(default_factory=tuple)
    #: Which prompt.json template to use. Derived splits reuse their parent's
    #: template so that a condition comparison varies ONLY the question text --
    #: giving each condition its own instruction would confound the experiment.
    prompt_key: str = ""
    #: True for splits we construct locally rather than download from the Hub.
    derived: bool = False
    #: Free-text note on how a derived split was produced (goes into reports).
    provenance: str = ""

    @property
    def template_key(self) -> str:
        return self.prompt_key or self.name

    @property
    def hf_path(self) -> str:
        """Path of the CSV inside the Hugging Face dataset repo."""
        return f"{self.task}/{self.name}.csv"

    @property
    def metric(self) -> MetricKind:
        return TASKS[self.task].metric

    @property
    def input_col(self) -> str:
        return TASKS[self.task].input_col

    @property
    def language(self) -> Language:
        return LANGUAGES[self.lang]

    @property
    def pretty(self) -> str:
        return f"{TASKS[self.task].pretty}/{self.language.name}"


_SENT3_LOWER = ("positive", "negative", "neutral")
_SENT4_UPPER = ("Positive", "Negative", "Neutral", "Mixed_feelings")

#: The 18 evaluable cells. 5 languages x 8 tasks = 40 possible; 22 have no data
#: upstream. `coverage_matrix()` renders the gaps.
DATASETS: dict[str, DatasetSpec] = {
    # ---- LID (2) ----
    "lid_hineng": DatasetSpec(
        "lid_hineng", "lid", "hin", 744,
        labels=("lang1", "lang2", "mixed", "ambiguous", "fw", "ne", "unk", "other"),
        notes="LinCE-style label set: lang1=English, lang2=Hindi.",
    ),
    "lid_mareng": DatasetSpec(
        "lid_mareng", "lid", "mar", 1340,
        labels=("ENG", "MAR", "OTH"),
        notes="Different label scheme from lid_hineng (ENG/MAR/OTH, not lang1/lang2). "
              "Do not average LID accuracy across the two without noting this.",
    ),
    # ---- POS (1) ----
    "pos_hineng": DatasetSpec(
        "pos_hineng", "pos", "hin", 160,
        labels=("ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
                "PRON", "PROPN", "PUNCT", "SCONJ", "VERB", "X", "PART_NEG", "PRON_WH", "UNK"),
        notes="Smallest split in the suite (160 rows) -- expect wide confidence intervals.",
    ),
    # ---- NER (1) ----
    "ner_hineng": DatasetSpec(
        "ner_hineng", "ner", "hin", 314,
        labels=("O", "B-PERSON", "I-PERSON", "B-PLACE", "I-PLACE",
                "B-ORGANISATION", "I-ORGANISATION"),
        notes="BIO scheme. Heavily O-dominated, so macro-F1 is the informative metric.",
    ),
    # ---- SA (5) ----
    "sa_hineng": DatasetSpec(
        "sa_hineng", "sa", "hin", 1261, labels=_SENT3_LOWER,
        notes="3-class sentiment, lowercase labels.",
    ),
    "sa_beneng": DatasetSpec(
        "sa_beneng", "sa", "ben", 1000, labels=("O", "N"),
        notes="NOT sentiment: this split is binary OFFENSIVE-language detection "
              "('O'=offensive, 'N'=non-offensive) filed under the SA task upstream. "
              "Report it separately from the 3/4-class sentiment splits.",
        extra_cols=("cm",),
    ),
    "sa_mareng": DatasetSpec(
        "sa_mareng", "sa", "mar", 1250, labels=_SENT3_LOWER,
        notes="3-class sentiment, lowercase labels, near-perfectly balanced.",
    ),
    "sa_tameng": DatasetSpec(
        "sa_tameng", "sa", "tam", 3049, labels=_SENT4_UPPER,
        notes="4-class incl. Mixed_feelings; capitalised labels. Skewed to Positive (68%), "
              "so compare accuracy against a 68.1% majority baseline.",
    ),
    "sa_maleng": DatasetSpec(
        "sa_maleng", "sa", "mal", 1171, labels=_SENT4_UPPER,
        notes="4-class incl. Mixed_feelings; capitalised labels. Majority baseline 48.2%. "
              "The only Malayalam split in the benchmark.",
    ),
    # ---- MT (3), all -> English ----
    "mt_hineng_eng": DatasetSpec(
        "mt_hineng_eng", "mt", "hin", 942, bleu_tokenize="13a",
        notes="Code-mixed Hindi-English -> English.",
    ),
    "mt_beneng_eng": DatasetSpec(
        "mt_beneng_eng", "mt", "ben", 2000, bleu_tokenize="13a",
        notes="Code-mixed Bengali-English -> English.",
    ),
    "mt_mareng_eng": DatasetSpec(
        "mt_mareng_eng", "mt", "mar", 2000, bleu_tokenize="13a",
        notes="Code-mixed Marathi-English -> English.",
    ),
    # ---- MMLU (4) ----
    "mmlu_hineng": DatasetSpec(
        "mmlu_hineng", "mmlu", "hin", 1024, labels=("A", "B", "C", "D"),
        extra_cols=("id", "tokens", "lids"),
    ),
    "mmlu_beneng": DatasetSpec(
        "mmlu_beneng", "mmlu", "ben", 1114, labels=("A", "B", "C", "D"),
        extra_cols=("id", "tokens", "lids"),
    ),
    "mmlu_mareng": DatasetSpec(
        "mmlu_mareng", "mmlu", "mar", 1067, labels=("A", "B", "C", "D"),
        extra_cols=("id", "tokens", "lids"),
    ),
    "mmlu_tameng": DatasetSpec(
        "mmlu_tameng", "mmlu", "tam", 1047, labels=("A", "B", "C", "D"),
        extra_cols=("id", "tokens", "lids"),
    ),
    # ---- GSM8K (1) ----
    "gsm8k_hineng": DatasetSpec(
        "gsm8k_hineng", "gsm8k", "hin", 1016, labels=None,
        notes="Free-form numeric answer, exact match after normalisation. The CSV "
              "carries a `cot` column used to build few-shot chain-of-thought shots.",
        extra_cols=("cot", "tokens", "lids"),
    ),
    # ---- TruthfulQA (1) ----
    "truthfulqa_hineng": DatasetSpec(
        "truthfulqa_hineng", "truthfulqa", "hin", 757,
        labels=tuple("ABCDEFGHIJKLMN"),
        notes="Variable option count per question (observed gold labels span A-K).",
        extra_cols=("tokens", "lids"),
    ),
}

#: Total measured rows across the suite.
TOTAL_ROWS: int = sum(d.n_rows for d in DATASETS.values())  # 21_256

#: Splits that exist upstream in prompt.json but have no CSV published on the
#: Hub, so they cannot be evaluated. Kept here so the gap is documented rather
#: than silently absent.
MISSING_UPSTREAM: dict[str, str] = {
    "mmlu_maleng": "Prompt template exists in prompt.json but mmlu/mmlu_maleng.csv is "
                   "absent from the Hub. Would complete the SA+MMLU block for Malayalam.",
    "mt_msaea": "Non-Indic (Modern Standard Arabic / Egyptian Arabic); out of scope here.",
    "mt_spaeng": "Non-Indic (Spanish / English); out of scope here.",
}


#: Locally derived splits for the "code-mixing tax" study. NOT part of the
#: published benchmark -- kept out of DATASETS so the 18-split suite and its row
#: total stay exactly as upstream defines them. Build with
#: `python scripts/build_conditions.py`, which writes them into data/raw/.
DERIVED: dict[str, DatasetSpec] = {
    "mmlu_hineng_en": DatasetSpec(
        "mmlu_hineng_en", "mmlu", "hin", 1024, labels=("A", "B", "C", "D"),
        prompt_key="mmlu_hineng", derived=True,
        extra_cols=("id", "src"),
        provenance="English originals for the exact 1,024 items of mmlu_hineng, "
                   "recovered from cais/mmlu by the `id` column (subject/test/row). "
                   "Gold answers verified to agree with mmlu_hineng.",
        notes="Monolingual English reference condition. This is the ceiling against "
              "which the code-mixing tax is measured.",
    ),
    "mmlu_hineng_hi": DatasetSpec(
        "mmlu_hineng_hi", "mmlu", "hin", 1024, labels=("A", "B", "C", "D"),
        prompt_key="mmlu_hineng", derived=True,
        extra_cols=("id", "src", "annotated"),
        provenance="Monolingual Hindi (Devanagari) for the exact 1,024 items of "
                   "mmlu_hineng, retrieved from CohereLabs/Global-MMLU config 'hi' by "
                   "sample_id. Gold answers verified against mmlu_hineng (1024/1024). "
                   "21.6% of items carry Global-MMLU's is_annotated flag (human "
                   "verified); the rest are MT with lighter review.",
        notes="Native-script monolingual condition. 99.5% Devanagari -- technical "
              "loanwords are transliterated (न्यूरोट्रांसमीटर) rather than calqued, so "
              "this is 'native script', not literally 'zero English'.",
    ),
    "mmlu_hineng_hirom": DatasetSpec(
        "mmlu_hineng_hirom", "mmlu", "hin", 1024, labels=("A", "B", "C", "D"),
        prompt_key="mmlu_hineng", derived=True,
        extra_cols=("id", "src", "annotated"),
        provenance="mmlu_hineng_hi with Devanagari transliterated to informal romanized "
                   "Hindi by the same deterministic transform used for mmlu_hineng_rom.",
        notes="Romanized monolingual condition. Completes a 2x2 of script (Devanagari "
              "vs Latin) x mixing (monolingual vs code-mixed), which separates the "
              "script penalty from the code-mixing penalty.",
    ),
    "mmlu_hineng_rom": DatasetSpec(
        "mmlu_hineng_rom", "mmlu", "hin", 1024, labels=("A", "B", "C", "D"),
        prompt_key="mmlu_hineng", derived=True,
        extra_cols=("id", "src"),
        provenance="mmlu_hineng with Devanagari transliterated to informal romanized "
                   "Hindi (ITRANS + schwa deletion + anusvara->n + lowercasing). "
                   "English tokens untouched. Word-for-word faithful: no translation.",
        notes="Latin-script code-mixed condition. Isolates the SCRIPT component of "
              "the tax from the mixing component, since the words are identical to "
              "mmlu_hineng.",
    ),
}

# ---- GSM8K language-form conditions -------------------------------------- #
# Paired to gsm8k_hineng via a two-hop join: the `cot` field is verbatim English
# GSM8K reasoning, which fingerprints each item to (split, row) in openai/gsm8k
# (1016/1016 matched); bingbangboom/gsm8k-hindi preserves that row order.
#
# NOTE the row count: 1016 items reduce to the subset where the Hindi question's
# number set exactly matches the English one, dropping translations that garbled a
# quantity (such an item would be a DIFFERENT problem carrying the English gold
# answer, which would silently penalise the Hindi condition).
#
# ZERO-SHOT ONLY: the `cot` column on these derived splits carries the ENGLISH
# chain of thought regardless of condition, because the Hindi CoT field in the
# source is malformed for ~27% of rows. Running them with --shots > 0 would leak
# English reasoning into every condition's prompt.
_GSM8K_DERIVED_ROWS = 955

DERIVED_GSM8K: dict[str, DatasetSpec] = {
    "gsm8k_hineng_en": DatasetSpec(
        "gsm8k_hineng_en", "gsm8k", "hin", _GSM8K_DERIVED_ROWS, labels=None,
        prompt_key="gsm8k_hineng", derived=True, extra_cols=("cot", "src_split", "src_row"),
        provenance="English originals from openai/gsm8k, matched to gsm8k_hineng by "
                   "exact chain-of-thought fingerprint (1016/1016).",
        notes="Monolingual English reference. NOTE: 35.5% of the underlying items come "
              "from the GSM8K *train* split, not test as the CodeMixBench paper states; "
              "a contamination check found no memorisation effect for Gemma 3 "
              "(gaps < 2 pts, all CIs spanning zero).",
    ),
    "gsm8k_hineng_hi": DatasetSpec(
        "gsm8k_hineng_hi", "gsm8k", "hin", _GSM8K_DERIVED_ROWS, labels=None,
        prompt_key="gsm8k_hineng", derived=True, extra_cols=("cot", "src_split", "src_row"),
        provenance="Monolingual Hindi questions from bingbangboom/gsm8k-hindi (MIT), "
                   "aligned by row position; gold answers taken from openai/gsm8k, not "
                   "from that dataset's answer field (malformed for ~27% of rows).",
        notes="Native-script monolingual condition. Translation quality is visibly "
              "lower than Global-MMLU's -- lightly supervised MT, not professional.",
    ),
    "gsm8k_hineng_rom": DatasetSpec(
        "gsm8k_hineng_rom", "gsm8k", "hin", _GSM8K_DERIVED_ROWS, labels=None,
        prompt_key="gsm8k_hineng", derived=True, extra_cols=("cot", "src_split", "src_row"),
        provenance="gsm8k_hineng questions transliterated Devanagari -> romanized Hindi.",
        notes="Romanized code-mixed condition.",
    ),
    "gsm8k_hineng_hirom": DatasetSpec(
        "gsm8k_hineng_hirom", "gsm8k", "hin", _GSM8K_DERIVED_ROWS, labels=None,
        prompt_key="gsm8k_hineng", derived=True, extra_cols=("cot", "src_split", "src_row"),
        provenance="gsm8k_hineng_hi transliterated by the same deterministic transform.",
        notes="Romanized monolingual condition.",
    ),
}

#: TruthfulQA gets only the romanised condition, not the full five.
#: The English and Hindi originals both exist (truthfulqa/truthful_qa 817 rows;
#: alexandrainst/m_truthfulqa 'hi' 773 rows) but truthfulqa_hineng carries no `id`
#: column, and the benchmark authors sampled and shuffled 4 of the variable-length
#: mc1 choices per question. Recovering which distractors were used needs fuzzy
#: matching, and every route tried tops out near 50% recall: the code-mixed Hindi and
#: m_truthfulqa's Hindi are INDEPENDENT translations, so genuine matches only score
#: 0.8-0.9 on string similarity. With no id to verify against, a bad match would
#: silently put a different question in the reference condition. Romanisation needs
#: no join at all, so it is the one derived condition that is sound here.
DERIVED_TRUTHFULQA: dict[str, DatasetSpec] = {
    "truthfulqa_hineng_rom": DatasetSpec(
        "truthfulqa_hineng_rom", "truthfulqa", "hin", 757,
        # Inherit the label set instead of restating it. TruthfulQA keeps every mc1
        # choice, so option counts run from 2 to 13 and letters go past D -- an
        # A-D label tuple here silently rejected 122 valid E..N predictions as
        # unparsed and understated the romanised score.
        labels=DATASETS["truthfulqa_hineng"].labels,
        prompt_key="truthfulqa_hineng", derived=True,
        provenance="truthfulqa_hineng transliterated to Latin with the same ITRANS + "
                   "schwa-deletion pipeline used for the MMLU and GSM8K romanised "
                   "conditions. Option markers ((A): ... (D):) and gold letters are "
                   "untouched, so the pairing with truthfulqa_hineng is exact (757/757).",
        notes="Script-only manipulation on a third task: same items, same options, same "
              "gold, Devanagari replaced by Latin. Isolates the romanisation penalty "
              "without the translation provenance that EN/HI conditions introduce.",
    ),
}
DERIVED.update(DERIVED_GSM8K)
DERIVED.update(DERIVED_TRUTHFULQA)

#: Every spec the pipeline can run: the benchmark suite plus derived conditions.
ALL_SPECS: dict[str, DatasetSpec] = {**DATASETS, **DERIVED}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve(names: list[str] | None) -> list[DatasetSpec]:
    """Expand a list of dataset names, task names, or language codes into specs.

    Accepts, in any combination:
      * exact split names   -- ``sa_tameng``
      * task names          -- ``mmlu`` (all Indic MMLU splits)
      * language codes      -- ``tam`` (all Tamil splits)
      * ``all`` / ``None``  -- the whole 18-split suite

    Returns specs in canonical (task, language) order with duplicates removed.
    """
    if not names or "all" in names:
        # "all" means the published 18-split suite, never the derived conditions.
        selected = set(DATASETS)
    else:
        selected: set[str] = set()
        for raw in names:
            token = raw.strip()
            if token == "derived":  # nosec B105 -- keyword selecting the DERIVED split group, not a credential
                selected.update(DERIVED)
            elif token in ALL_SPECS:
                selected.add(token)
            elif token in TASKS:
                selected.update(n for n, d in DATASETS.items() if d.task == token)
            elif token in LANGUAGES:
                selected.update(n for n, d in DATASETS.items() if d.lang == token)
            else:
                raise KeyError(
                    f"Unknown dataset/task/language: {token!r}. "
                    f"Valid splits: {sorted(DATASETS)}. "
                    f"Valid tasks: {sorted(TASKS)}. Valid languages: {sorted(LANGUAGES)}."
                )
    return sort_specs([ALL_SPECS[n] for n in selected])


def sort_specs(specs: list[DatasetSpec]) -> list[DatasetSpec]:
    """Canonical ordering: task order, then language order."""
    return sorted(
        specs,
        key=lambda d: (TASK_ORDER.index(d.task), LANGUAGE_ORDER.index(d.lang)),
    )


def coverage_matrix() -> tuple[list[str], list[list[int | None]]]:
    """Return (row_labels, cells) for the language x task coverage grid.

    Cell values are row counts, or ``None`` where the benchmark has no data.
    """
    rows: list[list[int | None]] = []
    for lang in LANGUAGE_ORDER:
        row: list[int | None] = []
        for task in TASK_ORDER:
            hit = [d for d in DATASETS.values() if d.lang == lang and d.task == task]
            row.append(sum(d.n_rows for d in hit) if hit else None)
        rows.append(row)
    return [LANGUAGES[c].name for c in LANGUAGE_ORDER], rows


def majority_baseline(spec: DatasetSpec, labels: list[str]) -> float | None:
    """Accuracy a constant-prediction classifier would achieve, as a percentage.

    Needs the gold labels of the split, so callers pass them in after loading.
    Returns None for open-ended tasks where the notion does not apply.
    """
    if spec.metric != "sentence_label" or not labels:
        return None
    counts: dict[str, int] = {}
    for lab in labels:
        counts[str(lab)] = counts.get(str(lab), 0) + 1
    return 100.0 * max(counts.values()) / len(labels)
