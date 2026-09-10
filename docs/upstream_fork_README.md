# CodeMixTax

**Reproducible evaluation of local / open-weight LLMs on code-mixed Indian languages.**

A fork of [CodeMixBench](https://aclanthology.org/2025.emnlp-main.109/) (EMNLP 2025)
that replaces its cloud-API harness with a resumable pipeline for models you run
yourself, scoped to the benchmark's five Indian languages.

<p align="left">
  <a href="https://huggingface.co/datasets/CodeMixBench/CodeMixBench"><img alt="Dataset" src="https://img.shields.io/badge/🤗-Dataset-blue" /></a>
  <a href="https://aclanthology.org/2025.emnlp-main.109/"><img alt="Paper" src="https://img.shields.io/badge/📜-EMNLP%202025-purple" /></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green" /></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue" />
</p>

**Scope: 5 languages × 8 tasks → 18 splits, 21,256 rows.**
Hindi, Bengali, Marathi (Indo-Aryan) and Tamil, Malayalam (Dravidian), each
code-mixed with English. Nepali is excluded, and the benchmark's other 13
language pairs are out of scope.

---

## Contents

- [Why this fork exists](#why-this-fork-exists)
- [Quickstart](#quickstart)
- [What gets evaluated](#what-gets-evaluated)
- [Models](#models)
- [Running a real sweep](#running-a-real-sweep)
- [Outputs](#outputs)
- [Reproducibility guarantees](#reproducibility-guarantees)
- [Interpreting results honestly](#interpreting-results-honestly)
- [Repository layout](#repository-layout)
- [Extending](#extending)
- [Citing](#citing)

---

## Why this fork exists

Upstream CodeMixBench is a good benchmark with a harness built for the OpenAI and
Replicate APIs. Pointing it at a local model surfaced four blocking problems:

1. **It does not run.** `test_model.py` raises `UnboundLocalError` on line 20 before
   reaching any model, and calls `asyncio.WindowsSelectorEventLoopPolicy()`, which
   does not exist off Windows.
2. **The answer extraction has scoring bugs.** MCQ extraction takes the first
   capital `A`–`D` anywhere in the reply, so `"Answer: B"` scores as **A** — the
   word "Answer" starts with a capital A. GSM8K extraction consults the *gold*
   answer to break ties among candidate numbers, which inflates accuracy for
   verbose models.
3. **No local-inference path.** Rate-limited cloud batching is the wrong shape for
   a GPU on your desk, and there was no resume: a crash 15,000 rows into a sweep
   meant starting over.
4. **Reasoning models did not exist yet.** Nothing strips `<think>` blocks, so a
   stray letter inside a scratchpad becomes the answer.

This fork keeps what is good — the prompt templates and the metric definitions,
both reused verbatim — and rebuilds the rest. Every difference is enumerated in
**[docs/DEVIATIONS.md](docs/DEVIATIONS.md)**, and the buggy behaviour is preserved
behind `--extraction paper` so you can quantify its effect on your own generations
rather than take our word for it.

The upstream code is retained in this repo for provenance
([`utils.py`](utils.py), [`test_model.py`](test_model.py), [`oaib/`](oaib/),
[`prompt.json`](prompt.json)), otherwise byte-identical to upstream except for
a narrow security patch (`eval()` -> `ast.literal_eval()`, no score impact --
see [docs/DEVIATIONS.md](docs/DEVIATIONS.md) item 11).

## Quickstart

```bash
git clone <your-fork-url> && cd CodeMixTax
make install          # venv + package (CPU only; no GPU deps yet)
make data             # download 18 splits (~14 MB), verify all row counts
make smoke            # full pipeline with a fake backend -- no GPU, ~10 seconds
```

`make smoke` should print 18 scored splits and write `results/REPORT.md`. If it
does, the harness works and only the model remains.

Then evaluate a real model. Serve it, and point the harness at it:

```bash
# terminal 1 -- any OpenAI-compatible server works
scripts/serve_vllm.sh google/gemma-3-12b-it

# terminal 2
export CMB_BASE_URL=http://localhost:8000/v1
cmb-indic run --models gemma3-12b-it --limit 50    # sanity check first
cmb-indic run --models gemma3-12b-it               # then the full 21,256 rows
```

Or skip the server and load the model in-process:

```bash
make install-vllm
cmb-indic run --models gemma3-12b-it --backend vllm
```

**Gated models** (all Gemma, all Llama) need access approval on the Hugging Face
model page, then `HF_TOKEN` in your environment — copy `.env.example` to `.env`.

## What gets evaluated

```
                     LID     POS     NER      SA      MT    MMLU   GSM8K  TruthfulQA
Hindi                744     160     314    1261     942    1024    1016     757     8/8
Bengali               --      --      --    1000    2000    1114      --      --     3/8
Marathi             1340      --      --    1250    2000    1067      --      --     4/8
Tamil                 --      --      --    3049      --    1047      --      --     2/8
Malayalam             --      --      --    1171      --      --      --      --     1/8
```

**Hindi is the only language with all eight tasks.** 22 of the 40 language×task
cells have no data upstream — that is a property of the benchmark, not of this
pipeline. The practical consequences for study design:

- **A uniform all-languages × all-tasks comparison is impossible.** Don't try.
- The only fully dense multi-language block is **SA + MMLU across all five
  languages** (13,798 rows). That is the right basis for a cross-language claim.
- POS, NER, GSM8K and TruthfulQA are **Hindi-only**. Treat them as a Hindi
  deep-dive, not as evidence about Indic languages generally.
- `mmlu_maleng` has a prompt template upstream but no published CSV. If it ever
  appears, Malayalam joins the dense block.

Run `cmb-indic datasets -v` for the full table with per-split notes.

### Task and label details that will bite you

| Split | Watch out for |
|---|---|
| `sa_beneng` | **Not sentiment.** Binary offensive-language detection (`O`/`N`). Never average it with the sentiment splits. |
| `sa_tameng`, `sa_maleng` | 4-class incl. `Mixed_feelings`, **capitalised** labels. Tamil is 68% Positive — compare against that baseline, not against 25%. |
| `sa_hineng`, `sa_mareng` | 3-class, **lowercase** labels. |
| `lid_hineng` vs `lid_mareng` | Different tag sets (`lang1`/`lang2`/… vs `ENG`/`MAR`/`OTH`). Not on the same scale. |
| `pos_hineng` | Only 160 rows. Confidence intervals are wide; don't over-read small gaps. |
| `ner_hineng` | Heavily `O`-dominated — macro-F1 is the informative metric, not accuracy. |
| All MT | Target is English; sacreBLEU `13a`. chrF++ is also reported because BLEU is brittle on short transliterated references. |

## Models

24 model cards ship in [`configs/models/`](configs/models/), covering every model
you asked for. Every `hf_id` was checked against the Hub.

| Config | Params | VRAM~ | Notes |
|---|---|---|---|
| `gemma3-4b-it` | 4B | 8 GB | vision, gated |
| `gemma3-12b-it` | 12B | 25 GB | vision, gated |
| `gemma3-12b-it-int4` | 12B | 8 GB | W4A16 INT4 |
| `gemma3-27b-it` | 27B | 57 GB | vision, gated |
| `gemma4-12b-it` | 12B | 25 GB | note the capital `12B` in the repo id |
| `qwen3.6-27b` | 27B | 57 GB | reasoning |
| `qwen3.6-27b-int4` | 27B | 18 GB | community AWQ |
| `qwen3.5-9b` | 9B | 19 GB | **substituted** — see below |
| `qwen3-8b` / `qwen3-8b-int4` | 8B | 17 / 5 GB | first-party AWQ INT4 pair |
| `qwen3-vl-8b` | 8B | 17 GB | VLM |
| `sarvam-1-2b` | 2B | 4 GB | **base model, not instruct** |
| `sarvam-m-24b` | 24B | 50 GB | Indic-tuned on Mistral Small |
| `sarvam-30b` | 30B | 63 GB | MoE, ~2.4B active |
| `llama3.1-8b-it` | 8B | 17 GB | gated |
| `llama4-scout` | 109B | 229 GB | MoE; multi-GPU only |
| `deepseek-v4-flash` | ? | ? | size unconfirmed — fill in before publishing |
| `mistral-small-3.1-24b` | 24B | 50 GB | control for `sarvam-m-24b` |
| `phi-4-14b` | 14B | 29 GB | English-centric |
| `gpt-oss-20b` | 21B | ~16 GB | MXFP4; harmony format |

VRAM is weight-only; add 15–30% for KV cache and activations. `cmb-indic models`
prints this live.

### Three model IDs need your decision

1. **"Qwen3.6 8B" does not exist.** The Qwen3.6 line ships only **27B** (dense) and
   **35B-A3B** (MoE). Shipped as `qwen3.5-9b` (`Qwen/Qwen3.5-9B`, nearest
   same-generation 8B-class), with `qwen3-8b` as the true-8B alternative. Pick one
   and name the actual checkpoint in your report — not "Qwen3.6 8B".
2. **"DeepSeek V4"** resolves to `DeepSeek-V4-Flash` and `DeepSeek-V4-Pro`. Flash is
   configured as the locally feasible one; Pro is very unlikely to fit on one node.
   `params_b` is unset — fill it from the model card.
3. **"Gemma 4 12B"** is `google/gemma-4-12B-it` (capital `B`). An official INT4 QAT
   checkpoint exists at `google/gemma-4-12B-it-qat-w4a16-ct` if you want a
   quantisation comparison there too.

### Comparisons worth designing around

The suite in [`configs/suites/single-gpu.yaml`](configs/suites/single-gpu.yaml) is
built for four controlled contrasts:

| Question | Compare |
|---|---|
| Does Indic post-training help on code-mixed input? | `sarvam-m-24b` vs `mistral-small-3.1-24b` — same size, same base family |
| What does INT4 cost? | `qwen3-8b-int4` vs `qwen3-8b` — first-party AWQ, so quantisation is the only variable |
| Does scale help within a family? | `gemma3-4b-it` vs `gemma3-12b-it` vs `gemma3-27b-it` |
| Base vs instruction-tuned | `sarvam-1-2b` vs everything else |

The first row is the single most informative result here for an Indic-focused
report: it isolates Indic tuning from size and base architecture.

## Running a real sweep

```bash
# One model, all 18 splits, in-process vLLM
cmb-indic run --models sarvam-m-24b --backend vllm

# Several models, serving each in turn and reclaiming GPU memory between them
RUN_ID=paper-v1 scripts/run_suite.sh sarvam-m-24b mistral-small-3.1-24b qwen3-8b

# A predefined suite
cmb-indic run --suite single-gpu

# Subsets: by task, by language, or by split
cmb-indic run --models phi-4-14b --datasets mmlu,sa      # tasks
cmb-indic run --models phi-4-14b --datasets tam,mal      # languages
cmb-indic run --models phi-4-14b --datasets sa_tameng    # one split

# 5-shot on the tasks that consume shots (MMLU/GSM8K/TruthfulQA only)
cmb-indic run --suite fewshot-5

# Inspect the exact prompt before spending GPU hours
cmb-indic run --models qwen3-8b --dry-run
```

**Cost.** ~21k rows per model. The token-level splits (LID/POS/NER) dominate
because they emit roughly one output token per input token. Budget 1–3 hours per
8B-class model on one modern GPU with vLLM. Interrupted sweeps resume for free.

**Preflight before a long run:**

```bash
cmb-indic validate --check-rows --check-server --check-hub
```

## Outputs

```
runs/<run_id>/<model>/<split>/
├── generations.jsonl   raw replies + token counts + latency   (gitignored, regenerable)
├── predictions.csv     gold vs prediction per row             (auditable)
├── metrics.json        scores + diagnostics
└── run_meta.json       config, versions, git sha, checksums, fingerprints

results/
├── tidy.csv            one row per (model, split, metric) -- commit this
├── summary.csv         headline metric per (model, split)
└── REPORT.md           leaderboards, per-task and per-language tables
```

Because generations are cached, re-scoring is free and needs no model:

```bash
# What did the upstream extraction bugs cost? Same generations, both modes.
cmb-indic rescore --run-dir runs/paper-v1 --extraction paper
```

## Reproducibility guarantees

Concrete properties, each enforced in code and covered by tests:

- **Greedy decoding by default** (`temperature=0`), seed recorded.
- **Seeded few-shot selection.** Upstream's unseeded `df.sample()` made scores move
  between identical runs; shots here are drawn from a seeded per-row RNG.
- **Prompt fingerprinting.** A hash of the exact prompt set is stored, and the
  runner **refuses to reuse cached generations** if prompts have changed since —
  the guard against publishing a number produced by a prompt set you have edited.
- **Dataset pinning.** Every split's CSV sha256 goes into `run_meta.json`;
  `--dataset-revision` pins a Hub commit. `make data` fails loudly if row counts
  drift from the registry.
- **Full environment capture.** git commit (and dirty flag), Python, torch, vLLM,
  transformers, sacrebleu versions, GPU names.
- **Crash-safe resume.** Generations are appended and fsynced; a line truncated by
  `kill -9` is skipped, not fatal.

## Interpreting results honestly

The report is built to make three failure modes visible rather than hide them.

**Unparsed rate belongs next to every score.** Rows whose answer can't be
extracted count as wrong — never dropped, because dropping them lets a model that
answers 30% of items in a parseable format post an inflated score on that 30%. The
cost is that a formatting failure looks like a capability failure, so the report has
a dedicated **Extraction reliability** table. A 45% accuracy with 40% unparsed is a
formatting result, not a capability result.

**Check `n_truncated` for reasoning models.** A model that spends its whole token
budget thinking produces an empty answer. The Qwen configs disable thinking via
`chat_template_kwargs` for exactly this reason; if you enable it, raise `max_tokens`
well above 2048 and report the two configurations separately.

**No cross-task average is reported, deliberately.** Averaging BLEU with accuracy is
meaningless, and averaging a 160-row split with a 3,049-row one is misleading. The
report gives per-task tables plus a rank-based rollup. If you need one number,
publish the per-task table too.

**Compare against the majority baseline.** 68% on `sa_tameng` is *below* the 68.1%
majority-class baseline. The report prints baselines inline for this reason.

## Repository layout

```
├── src/cmb_indic/          the pipeline
│   ├── registry.py         the 18 splits, label spaces, coverage matrix
│   ├── data.py             download, cache, checksum, subsample
│   ├── prompts.py          upstream templates + seeded few-shot
│   ├── backends/           openai-compatible | vllm | transformers | echo
│   ├── extract.py          generation -> prediction (robust | paper modes)
│   ├── metrics.py          ports of the upstream metrics
│   ├── runner.py           orchestration, caching, resume, provenance
│   ├── report.py           aggregation -> tidy.csv + REPORT.md
│   └── cli.py              cmb-indic
├── configs/
│   ├── default.yaml        baseline settings + per-task token budgets
│   ├── models/             24 model cards (+ _TEMPLATE.yaml)
│   └── suites/             smoke | single-gpu | indic5-full | fewshot-5
├── scripts/                serve_vllm.sh, run_suite.sh, download_data.py
├── tests/                  161 tests; no GPU or network needed
├── docs/DEVIATIONS.md      every difference from upstream, with rationale
├── docs/COMPATIBILITY.md   driver/vLLM/architecture support matrix
│
├── utils.py, test_model.py, oaib/, prompt.json    upstream (see DEVIATIONS #11)
└── docs/upstream_README.md                        the original README
```

## Extending

**Add a model:** copy [`configs/models/_TEMPLATE.yaml`](configs/models/_TEMPLATE.yaml),
set `hf_id`, run `cmb-indic validate --models <name> --check-hub`.

**Add a serving runtime:** subclass `Backend` and implement `generate`. Nothing else
in the pipeline knows how models are served.

**Widen to the other 13 language pairs:** add `DatasetSpec` entries to
`registry.py`. Everything else — prompts, extraction, metrics, reporting — already
handles all eight tasks generically. Note that MT to Chinese needs
`bleu_tokenize="zh"`, which `bleu_metrics` already supports.

**Check a model will actually load** before spending GPU time:
`python scripts/check_compat.py --models <name>` verifies the architecture against
vLLM's registry and renders a real prompt through the model's chat template, all on
CPU. See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) for the known driver/CUDA
pitfalls.

**Contributing:** `make lint test` must pass. `pre-commit install` is configured.

## Citing

Cite the upstream paper alongside this repository (see [CITATION.cff](CITATION.cff)):

```bibtex
@inproceedings{yang-chai-2025-codemixbench,
    title = "{C}ode{M}ix{B}ench: Evaluating Code-Mixing Capabilities of {LLM}s Across 18 Languages",
    author = "Yang, Yilun and Chai, Yekun",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov, year = "2025", address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.109/",
    doi = "10.18653/v1/2025.emnlp-main.109",
    pages = "2139--2169",
}
```

If you report numbers from this pipeline, please state the **extraction mode**, the
**shot count**, and the **exact model checkpoints** — the substituted IDs above make
that last one matter.

## Licensing

Pipeline code: Apache-2.0 ([LICENSE](LICENSE)). Dataset: Apache-2.0, not vendored
here. Upstream code retained under its original terms. **Model weights carry their
own licenses**, several of them restrictive (Gemma Terms of Use, Llama Community
License) — check each model card before publishing. See [NOTICE](NOTICE).
