# The Code-Mixing Tax

**Take the same question in English, Hinglish, Hindi, and their romanized forms.
Measure how much answer quality each model loses across them.**

Everyone claims their model handles Hinglish. Nobody publishes the degradation
curve. This repository is that curve, for **16 open-weight models** on **MMLU**
and **GSM8K**.

**This repository ships the code, not the numbers.** Everything below is
reproducible from a clean clone; `runs/` and `results/` are generated locally and
deliberately not committed.

---

## The headline

**Code-mixing is not the problem. Romanization is.**

Mean share of each model's own above-chance English headroom that survives:

| Task | Hinglish | Native Hindi | Romanized Hinglish | **Romanized Hindi** |
|---|---:|---:|---:|---:|
| MMLU | 86.5% | 74.2% | 70.3% | **41.4%** |
| GSM8K | 95.7% | 88.1% | 85.7% | **70.2%** |

Two findings follow from that table.

**1. Hinglish is the *easiest* non-English form — for every model, on both
tasks** (15/15 on MMLU, 13/13 on GSM8K, zero exceptions). A Hinglish item hands
the model two anchors: recognizable English content words *and* Latin script.
Either one alone is enough to locate the question. The advertised hard case is
the easy one.

**2. Romanized Hindi removes both anchors, and that is where models collapse** —
Hindi words, Latin script, no English to grip. It is also the form hundreds of
millions of people actually type.

**English capability does not transfer.** The strongest English model in the
study is mid-table on robustness:

| Model | English MMLU | Romanized Hindi retained |
|---|---:|---:|
| Sarvam-M 24B | 83.01 | **71.7%** |
| Sarvam-30B (reasoning capped) | 82.13 | **69.6%** |
| gpt-oss-20b | 82.13 | 61.9% |
| Gemma 3 27B | 80.27 | 52.3% |
| **Nemotron 3.5 Lightning** | **89.36** ← best English | 47.3% |

Nemotron is **6.3 points stronger in English** than Sarvam-M and retains
**24.4 points less**. Indic post-training, not scale or raw capability, is what
moves the number.

---

## The five language forms

Every model sees the **same items** in all five conditions, so a drop is
attributable to the language form and nothing else.

| Code | Form | Script | Lexicon |
|---|---|---|---|
| `EN` | English | Latin | English |
| `CM` | **Hinglish** | Latin + Devanagari | Hindi grammar, English content words |
| `HI` | Native Hindi | Devanagari | Hindi |
| `CM-ROM` | Romanized Hinglish | Latin | Hindi grammar, English content words |
| `HI-ROM` | **Romanized Hindi** | Latin | Hindi |

**Only `CM` ships with CodeMixBench.** The other four were assembled for this
study and aligned item-by-item onto it:

| Condition | MMLU source | GSM8K source |
|---|---|---|
| `CM` | **CodeMixBench** `mmlu_hineng` | **CodeMixBench** `gsm8k_hineng` |
| `EN` | `cais/mmlu`, rejoined by `id` | `openai/gsm8k`, matched by chain-of-thought fingerprint |
| `HI` | `CohereLabs/Global-MMLU` config `hi` | `bingbangboom/gsm8k-hindi` (MIT) |
| `CM-ROM` | `CM`, transliterated | `CM`, transliterated |
| `HI-ROM` | `HI`, transliterated | `HI`, transliterated |

Gold answers for both Hindi conditions come from the **English originals**, never
from the translated datasets, so a mistranslated question cannot score correct
off a mistranslated answer — it simply fails. Transliteration is deterministic
(ITRANS + schwa deletion + anusvara→n + lowercasing); English tokens are left
untouched.

**Row counts.** MMLU is 1,024 items in all five conditions. GSM8K's published
code-mixed split has 1,016, but the derived conditions have **955** — items whose
Hindi translation garbled a quantity are dropped, since such an item is a
*different problem* carrying the English gold answer. **All five GSM8K conditions
are reported on that paired 955**, so no column is scored on a different item set.

---

## The metric: floor-adjusted retention

Raw retention (`accuracy ÷ own English accuracy`) flatters weak models. A model
scoring 32% English and 25% Romanized Hindi on MMLU reports 78% raw retention
while retaining *none* of its actual signal — 25% is the 4-option guess floor.

```
retention_adj = (accuracy − floor) / (accuracy_EN − floor)
```

MMLU floor is 25%; GSM8K is free-form numeric, so its floor is 0. Models whose
English score leaves no headroom above the floor are marked **floored** and
excluded from retention rankings — a ratio built on noise is noise.

---

## Reproducing the study

### 1. Install

```bash
git clone <this-repo> && cd CodeMixTax
make install          # venv + package (CPU only)
make data             # download the splits, verify every row count
make smoke            # full pipeline on a fake backend, no GPU, ~10s
```

`make smoke` scores every split and writes `results/REPORT.md`. If that works,
the harness is sound and only the model remains.

### 2. Evaluate a model

```bash
make install-vllm
export CUDA_VISIBLE_DEVICES=0

cmb-indic run --backend vllm --models gemma3-12b-it \
  --datasets mmlu_hineng,mmlu_hineng_en,mmlu_hineng_hi,mmlu_hineng_rom,mmlu_hineng_hirom \
  --out-dir runs --run-id my-run --limit 50      # sanity check first
```

Drop `--limit` for the full sweep, and add the five `gsm8k_hineng*` splits for the
other task. Repeat per model; `configs/models/*.yaml` carries each model's dtype,
token budgets, prompt style and quirks, so the command above is the same for all
of them.

Every run writes `metrics.json`, `predictions.csv`, `generations.jsonl` and
`run_meta.json` per split. `run_meta.json` records the exact configuration —
dtype, budgets, vLLM version, seed, prompt fingerprint — so a run is auditable
after the fact.

### 3. Build the report from your runs

```bash
python scripts/slim_predictions.py     # compact audit trail
python scripts/build_tax_dataset.py    # runs/ -> results/tax_dataset.json
python scripts/plot_tax.py             # -> results/figures/*.svg (light + dark)
python scripts/build_tax_report.py     # -> results/METRICS_REPORT.md
python scripts/build_tax_html.py       # -> results/codemix-tax.html
python scripts/embed_figures.py        # inline the SVGs into the HTML
```

Every number in the generated prose is computed from the run data, not typed, so
the text cannot drift from what the models actually produced.

### 4. Notes on running

```bash
make install-vllm
export CUDA_VISIBLE_DEVICES=0

cmb-indic run --backend vllm --models gemma3-12b-it \
  --datasets mmlu_hineng,mmlu_hineng_en,mmlu_hineng_hi,mmlu_hineng_rom,mmlu_hineng_hirom \
  --out-dir runs --run-id my-run --limit 50      # sanity check first
```

Drop `--limit` for the full sweep.

**Per-model drivers are not shipped.** The 17 shell scripts used for this study
were written for one cluster's GPUs, queueing and quirks; they are not what
someone else should run. `configs/models/*.yaml` is the reproducible spec, and
the single `cmb-indic run` command above consumes it.

Drivers abort loudly if a split all-errors or if a run silently loses a setting —
worth copying if you write your own.

**Gated models** (Gemma, Llama) need access approval on the model page, then
`HF_TOKEN` in your environment — copy `.env.example` to `.env`.

### Hardware and stack

Everything ran on a **single A100 80GB**. Models 1–14 used **vLLM 0.10.1 /
torch 2.7.1+cu126**; Nemotron and Sarvam-30B needed **vLLM 0.27.1 /
torch 2.13.0+cu130** for architectures the older release did not implement. That
difference is recorded per model in the report and in `run_meta.json`.

---

## What's in here

```
src/cmb_indic/          the harness: registry, prompts, backends, extraction, scoring
  registry.py           split definitions + provenance for all 27 splits
  extract.py            shape-aware answer extraction (see "Pitfalls" below)
  backends/             vllm_offline · hf_local · openai_compat · echo
configs/models/         one YAML per model: dtype, budgets, quirks, caveats
scripts/                run_*.sh drivers · build_* report generators · plot_tax.py
runs/                   generated by a run; gitignored
results/                generated by the report scripts; gitignored
tests/                  190 tests, including regression tests for every parser bug
docs/                   compatibility notes, deviations from upstream, security posture
```

**What is committed.** Code, configs, tests and documentation — 135 files, 1.6 MB.
Nothing a run produces is committed: `runs/` and `results/` are gitignored in full,
as are the per-model drivers, smoke probes and one-off analyses used for this
study. The intent is that the repository is the method, and the numbers are
whatever you reproduce from it.

---

## Pitfalls this study hit, so you don't have to

These cost real time and several of them silently corrupted numbers before being
caught. All are now covered by regression tests.

**Answer extraction is the single biggest risk.** An extractor that takes the
first option letter appearing anywhere in a reply penalises models that explain
before answering. Fixing this moved Llama 4 Scout's English MMLU from **70.02 to
84.67** — a 14-point artifact that looked like a real result. Extraction is now
shape-aware: answer-first and answer-last replies are both handled, and a
truncated reasoning trace scores `UNK` rather than donating a letter lifted from
mid-reasoning.

**Reasoning models need budget, and sometimes a cap.** A trace cut off by the
token budget contains no answer. Sarvam-30B left 34–54% of GSM8K rows unfinished
at 8,192 tokens, and its author-recommended 65,536 collapsed concurrency to 7×
(2h22m for one 115-row split). Capping *reasoning* at 4,000 tokens via vLLM's
`thinking_token_budget` — which injects `</think>` at the cap and forces the model
to commit — took truncation from **2,543 rows to 15** and made the model
reportable. Note that caps change what you measure, so it is stated wherever those
numbers appear.

**Pairing is easy to break.** GSM8K's code-mixed split has 1,016 rows against the
derived conditions' 955. Scoring it on all 1,016 while its siblings used 955 broke
the within-item design the whole study rests on. The builder now raises rather
than emit a table whose paired claim would be false.

**Round once.** Rounding a percentage to 2dp in the data and again to 1dp for
display shifts values whose third decimal straddles a boundary (11/955 = 1.1518 →
1.15 → "1.1", but correctly "1.2").

**Base models carry no 0-shot signal.** OpenHathi 7B scores 32.13% on English
MMLU against a 25% floor. Raw retention would rank it above Gemma 3 12B;
floor-adjusted retention correctly reports ~0.

---

## Credit

**CodeMixBench** ([Yang & Chai, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.109/))
supplies the code-mixed Hinglish split, the anchor every other condition is
aligned to.

The English and Native Hindi conditions come from other public datasets:
`cais/mmlu` and `openai/gsm8k` for English, `CohereLabs/Global-MMLU` and
`bingbangboom/gsm8k-hindi` (MIT) for Hindi.

This study contributes the item-level alignment that makes all five conditions one
paired set, the two romanized forms, the floor-adjusted retention metric, and the
16-model sweep.

The harness is a fork of CodeMixBench's evaluation code, rewritten for local /
open-weight models — see [`docs/upstream_fork_README.md`](docs/upstream_fork_README.md)
for the full harness documentation and [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)
for where it departs from upstream.

## Licensing

This fork's contributions — the harness, configs, scripts, tests, and results —
are **MIT** (see [LICENSE](LICENSE)).

Files retained from upstream CodeMixBench (`utils.py`, `test_model.py`, `oaib/`,
`prompt.json`, `pics/`, `requirements.txt`) remain **Apache 2.0**, preserved in
[LICENSE.upstream-Apache-2.0](LICENSE.upstream-Apache-2.0). [NOTICE](NOTICE) has
the file-by-file breakdown.

Dataset licences belong to their sources — the CodeMixBench dataset is Apache 2.0,
`bingbangboom/gsm8k-hindi` is MIT. Model weights are governed by each model's own
licence, several of which are not OSI-approved (Gemma Terms of Use, Llama
Community License); no weights are included here.

## Citing

If you use the harness or the derived conditions, please cite the original
benchmark — see [CITATION.cff](CITATION.cff).
