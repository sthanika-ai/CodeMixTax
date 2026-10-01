# The Code-Mixing Tax

Measures how much answer quality 16 open-weight models lose when the same MMLU and GSM8K questions are asked in English, Hinglish, Hindi and their romanized forms.

[![License](https://img.shields.io/badge/license-MIT%20%2B%20Apache--2.0-56BF4F?style=flat-square&labelColor=1E281F)](LICENSE)
[![Upstream: CodeMixBench](https://img.shields.io/badge/built%20on-CodeMixBench-56BF4F?style=flat-square&labelColor=1E281F)](#related)
[![Report](https://img.shields.io/badge/report-sthanika.ai-56BF4F?style=flat-square&labelColor=1E281F&logo=firefox&logoColor=white)](https://sthanika.ai/research/codemix-tax-2026)

## What it measures

Everyone claims their model handles Hinglish, but nobody publishes the degradation curve. This repo is that curve. Every model sees the same items in five language forms, so a drop is attributable to the language form and nothing else.

| code | form | script | lexicon |
|---|---|---|---|
| EN | English | Latin | English |
| CM | Hinglish | Latin + Devanagari | Hindi grammar, English content words |
| HI | Native Hindi | Devanagari | Hindi |
| CM-ROM | Romanized Hinglish | Latin | Hindi grammar, English content words |
| HI-ROM | Romanized Hindi | Latin | Hindi |

Only CM ships with CodeMixBench; the other four were assembled for this study and aligned item by item onto it. Gold answers for both Hindi conditions come from the English originals, never the translated datasets, so a mistranslated question cannot score correct off a mistranslated answer. Transliteration is deterministic (ITRANS, schwa deletion, anusvara to n, lowercasing) and leaves English tokens untouched.

Scores use **floor-adjusted retention**, because raw retention (accuracy ÷ own English accuracy) flatters weak models:

```
retention_adj = (accuracy − floor) / (accuracy_EN − floor)
```

The MMLU floor is 25% (4-option guess); GSM8K is free-form numeric, so its floor is 0. Models whose English score leaves no headroom above the floor are marked floored and excluded from retention rankings.

Row counts: MMLU has 1,024 items in all five conditions. GSM8K's published code-mixed split has 1,016, but the derived conditions have 955, because items whose Hindi translation garbled a quantity are dropped. All five GSM8K conditions are reported on that paired 955. Report: [sthanika.ai](https://sthanika.ai/research/codemix-tax-2026)

## Quickstart

```bash
git clone https://github.com/sthanika-ai/CodeMixTax.git
cd CodeMixTax
make install          # venv + package (CPU only)
make data             # download the splits, verify every row count
make smoke            # full pipeline on a fake backend, no GPU, ~10s
```

`make smoke` scores every split and writes `results/REPORT.md`. If that works, the harness is sound and only the model remains.

Evaluate a model (one A100 80GB was used for everything):

```bash
make install-vllm
export CUDA_VISIBLE_DEVICES=0

cmb-indic run --backend vllm --models gemma3-12b-it \
  --datasets mmlu_hineng,mmlu_hineng_en,mmlu_hineng_hi,mmlu_hineng_rom,mmlu_hineng_hirom \
  --out-dir runs --run-id my-run --limit 50      # sanity check first
```

Drop `--limit` for the full sweep, add the five `gsm8k_hineng*` splits for the other task, and repeat per model. `configs/models/*.yaml` carries each model's dtype, token budgets, prompt style and quirks, so the command is the same for all of them. Then build the report:

```bash
python scripts/slim_predictions.py     # compact audit trail
python scripts/build_tax_dataset.py    # runs/ -> results/tax_dataset.json
python scripts/plot_tax.py             # -> results/figures/*.svg (light + dark)
python scripts/build_tax_report.py     # -> results/METRICS_REPORT.md
python scripts/build_tax_html.py       # -> results/codemix-tax.html
python scripts/embed_figures.py        # inline the SVGs into the HTML
```

Notes:

- **Outputs.** Every run writes `metrics.json`, `predictions.csv`, `generations.jsonl` and `run_meta.json` per split. `run_meta.json` records dtype, budgets, vLLM version, seed and prompt fingerprint, so a run is auditable afterwards. Every number in the generated prose is computed from the run data, not typed.
- **No per-model drivers.** The 17 shell scripts used for the study were written for one cluster and are not shipped. `configs/models/*.yaml` plus the single `cmb-indic run` command is the reproducible spec. If you write your own drivers, make them abort loudly if a split all-errors or a run silently loses a setting.
- **Gated models.** Gemma and Llama need access approval on the model page, then `HF_TOKEN` in your environment (copy `.env.example` to `.env`).
- **Stack.** Models 1–14 used vLLM 0.10.1 / torch 2.7.1+cu126. Nemotron and Sarvam-30B needed vLLM 0.27.1 / torch 2.13.0+cu130 for architectures the older release lacks. This is recorded per model in the report and in `run_meta.json`.
- **Layout.** The harness is in `src/cmb_indic/` (registry, prompts, backends, extraction, scoring). `runs/` and `results/` are generated and gitignored; the repo is the method, and the numbers are whatever you reproduce from it. 190 tests include a regression test for every parser bug.

## Results

**Code-mixing is not the problem. Romanization is.** Mean share of each model's own above-chance English headroom that survives:

| task | Hinglish | Native Hindi | Romanized Hinglish | Romanized Hindi |
|---|---|---|---|---|
| MMLU | 86.5% | 74.2% | 70.3% | 41.4% |
| GSM8K | 95.7% | 88.1% | 85.7% | 70.2% |

1. Hinglish is the easiest non-English form for every model on both tasks (15/15 on MMLU, 13/13 on GSM8K, zero exceptions). It gives the model two anchors, recognisable English content words and Latin script.
2. Romanized Hindi removes both anchors, and that is where models collapse. It is also the form hundreds of millions of people actually type.

English capability does not transfer. The strongest English model in the study is mid-table on robustness:

| model | English MMLU | Romanized Hindi retained |
|---|---|---|
| Sarvam-M 24B | 83.01 | 71.7% |
| Sarvam-30B (reasoning capped) | 82.13 | 69.6% |
| gpt-oss-20b | 82.13 | 61.9% |
| Gemma 3 27B | 80.27 | 52.3% |
| Nemotron 3.5 Lightning | 89.36 (best English) | 47.3% |

Nemotron is 6.3 points stronger in English than Sarvam-M and retains 24.4 points less, so Indic post-training, not scale or raw capability, is what moves the number.

Pitfalls this study hit, all now covered by regression tests:

- **Answer extraction is the biggest risk.** Taking the first option letter anywhere in a reply penalises models that explain before answering. Fixing it moved Llama 4 Scout's English MMLU from 70.02 to 84.67, a 14-point artifact. Extraction is now shape-aware, and a truncated reasoning trace scores UNK.
- **Reasoning models need budget, and sometimes a cap.** Sarvam-30B left 34–54% of GSM8K rows unfinished at 8,192 tokens. Capping reasoning at 4,000 tokens via vLLM's `thinking_token_budget` cut truncation from 2,543 rows to 15, but caps change what you measure, so it is stated wherever those numbers appear.
- **Pairing is easy to break.** Scoring GSM8K code-mixed on 1,016 rows while its siblings used 955 breaks the within-item design. The builder now raises rather than emit a false paired table.
- **Round once.** Rounding to 2dp in the data and again to 1dp for display shifts boundary values (11/955 is 1.1518, which should display as 1.2).
- **Base models carry no 0-shot signal.** OpenHathi 7B scores 32.13% on English MMLU against a 25% floor. Raw retention would rank it above Gemma 3 12B; floor-adjusted retention correctly reports about 0.

Full report: [sthanika.ai](https://sthanika.ai/research/codemix-tax-2026)

## Citation

If you use the harness or the derived conditions, please cite the original benchmark (CodeMixBench, Yang & Chai, EMNLP 2025); see [`CITATION.cff`](CITATION.cff).

```bibtex
@software{codemixtax2026,
  title  = {The Code-Mixing Tax},
  author = {{sthanika-ai}},
  year   = {2026},
  url    = {https://github.com/sthanika-ai/CodeMixTax}
}
```

## License

This fork's contributions (harness, configs, scripts, tests) are MIT, see [LICENSE](LICENSE). Files retained from upstream CodeMixBench (`utils.py`, `test_model.py`, `oaib/`, `prompt.json`, `pics/`, `requirements.txt`) remain Apache-2.0, see [LICENSE.upstream-Apache-2.0](LICENSE.upstream-Apache-2.0) and [NOTICE](NOTICE). Dataset licences belong to their sources (CodeMixBench is Apache-2.0, `bingbangboom/gsm8k-hindi` is MIT). Model weights follow each model's own licence, several of which are not OSI-approved (Gemma Terms of Use, Llama Community License). No weights are included.

## Related

- CodeMixBench (Yang & Chai, EMNLP 2025): supplies the code-mixed Hinglish split, the anchor every other condition is aligned to. This harness is a fork of its evaluation code, rewritten for local open-weight models; see `docs/upstream_fork_README.md` and `docs/DEVIATIONS.md`.
- Source datasets: `cais/mmlu`, `openai/gsm8k`, `CohereLabs/Global-MMLU` (config `hi`), `bingbangboom/gsm8k-hindi`
- Companion work from sthanika-ai: [token_fertility](https://github.com/sthanika-ai/token_fertility), [india-in-the-wild](https://github.com/sthanika-ai/india-in-the-wild)
- Site: [sthanika.ai](https://sthanika.ai)
