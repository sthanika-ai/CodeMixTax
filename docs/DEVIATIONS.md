# Deviations from the upstream CodeMixBench implementation

Every difference between this pipeline and the upstream reference code, why it
exists, and whether it can move a score. Read this before comparing numbers from
this repo against the published CodeMixBench table.

Upstream reference: [`utils.py`](../utils.py), [`test_model.py`](../test_model.py) --
otherwise byte-identical to upstream except for the security patches in item 11,
none of which change any score (see that item for why).

## Summary

| # | Area | Score impact | Toggle |
|---|---|---|---|
| 1 | Metric formulas | None -- identical | n/a |
| 2 | MCQ answer extraction | **Large** | `--extraction paper` |
| 3 | GSM8K answer extraction | **Large** | `--extraction paper` |
| 4 | Token-label alignment | Small | `--extraction paper` |
| 5 | Reasoning/CoT stripping | Large for reasoning models | always on |
| 6 | Few-shot exemplar seeding | Small, removes run-to-run noise | `--seed` |
| 7 | SA label normalisation | Large | `--extraction paper` |
| 8 | Unparseable rows kept in denominator | Moderate | none (deliberate) |
| 9 | Inference layer rewritten | None | n/a |
| 10 | Upstream runtime bugs fixed | Upstream did not run at all | n/a |
| 11 | Security patches to the vendored files | None -- identical | n/a |

---

## 1. Metric formulas -- identical

`src/cmb_indic/metrics.py` is a direct port. Given the same predictions it returns
the same `accuracy`, `micro_f1`, `macro_f1`, `weighted_f1` and `bleu` as upstream:

* **Token tasks** compute the sklearn score *per sentence* and take the unweighted
  mean over sentences. This weights a 3-token sentence as heavily as a 40-token
  one. Unusual, but it is what the published numbers use, so it is preserved
  verbatim under the same keys.
* **Sentence tasks** compute corpus-level sklearn scores.
* **MT** uses corpus sacreBLEU.

Additions that do not alter the originals:

* `corpus_*` keys on token tasks: the same metrics computed over a flat pool of all
  tokens. This is the conventional token-level number and the fairer one when
  sentence lengths vary a lot. Both are reported.
* `chrf2` (chrF++) alongside BLEU. BLEU is brittle on short, transliterated,
  noisy references, which describes much of this MT data. BLEU stays the headline.
* Refactors: return dicts instead of writing files and mutating DataFrames; raise
  `ValueError` instead of calling `exit()`.

## 2. MCQ answer extraction -- the big one

Upstream (`utils.py`, `analyse_Sentence_Label_Result`):

```python
pattern = r"[ABCD]"
pred_re = re.findall(pattern, answer, re.MULTILINE)
pred = pred_re[0]     # first capital A-D anywhere in the reply
```

The reply `"Answer: B"` scores as **A**, because the word "Answer" starts with a
capital A. Any model that prefixes its choice — most instruction-tuned models —
is systematically mis-scored, and the error is not random: it biases predictions
toward A. The same applies to TruthfulQA over `[A-N]`, where far more English
words contain a matching capital.

`robust` mode anchors instead: explicit answer patterns first
(`answer is (C)`, `(B)`, `B)`, a bare letter on its own line), then a standalone
letter token as a last resort, preferring the last such token because a verbose
reply ends on its conclusion.

Reproduce the upstream behaviour with `--extraction paper`. Verified by
`tests/test_extract.py::TestMCQ::test_paper_mode_reproduces_upstream_bug`.

## 3. GSM8K answer extraction -- label leakage

Upstream (`utils.py` around line 810), when several numbers are found:

```python
if str(true_tag) in pred_re:
    pred = true_tag        # <-- the gold answer is consulted
else:
    pred = pred_re[0]
```

If the gold answer appears anywhere among the numbers in the reply, it is taken as
the prediction. A model that shows its working will usually mention the correct
intermediate or final value somewhere, so this inflates accuracy — and inflates it
more for more verbose models, which makes cross-model comparison unsound.

`robust` mode never reads the gold value. It looks for an explicit
`Final answer:` / `####` marker and otherwise takes the last number in the reply,
the standard GSM8K convention. `extract_gsm8k(..., gold=...)` is accepted but
ignored in `robust` mode by design, so the scoring path cannot leak the label.
Numbers are normalised (`1,234.0` → `1234`) before comparison.

## 4. Token-label alignment

Upstream aligns predicted tags to gold tokens by name:

```python
pred_dict = dict(zip(tokens, pred))
pred_lid = [pred_dict.get(tok, 'unk') for tok in true_tokens]
```

A sentence containing the same token twice collapses to one dictionary entry, so
every occurrence inherits the *last* tag the model gave. Repeated tokens are
common in this data (punctuation, `the`, `@USER`, hashtags).

`robust` mode aligns positionally when the model returned exactly one tag per gold
token — the common case, and the only correct handling of duplicates — and falls
back to by-name lookup (consuming duplicates left to right) otherwise. It also
accepts the reply shapes local models actually produce: JSON, Python literals,
fenced code blocks, a flat dict, a bare label list, and `token: LABEL` lines.
Tags are snapped case-insensitively onto the split's label set.

Both modes always return exactly `len(gold_tokens)` labels, padding with `unk`, so
a parse failure scores as wrong rather than crashing.

## 5. Reasoning / chain-of-thought stripping (new)

Upstream predates open reasoning models. Several models in this suite emit hidden
CoT — `<think>...</think>` (Qwen3.x, DeepSeek, Sarvam-M) or harmony
`analysis`/`final` channels (gpt-oss). Without stripping, a stray capital letter or
number inside the scratchpad is extracted as the answer, which produces near-random
scores for exactly the models most likely to be strong.

`strip_reasoning` runs in **both** extraction modes, because it is a transport
concern rather than a scoring choice. An *unterminated* `<think>` (the model ran out
of tokens mid-thought) discards everything from the tag onward, so the row is
correctly scored as unparsed instead of contributing a lucky guess. Truncated rows
are counted in `n_truncated` in `run_meta.json` — check it before trusting a
reasoning model's score.

## 6. Few-shot exemplar selection is seeded

Upstream calls `df.sample(n=1)` with no `random_state`, so every run draws
different in-context examples and re-running the identical command moves scores.
Here shots are drawn from an RNG seeded on `(seed, row position)`: reproducible
across runs, still row-dependent as upstream intended. Change the draw with
`--seed`.

Upstream's exemplar loop is also unbounded (`while` with `continue` on rejection)
and hangs when the length filter can never be satisfied. Here the filter relaxes
after a bounded number of draws.

`prompt_fingerprint` records a hash of the exact prompt set in `run_meta.json`, and
the runner **refuses to reuse cached generations** whose fingerprint differs — the
guard against publishing a number produced by a prompt set you have since changed.

## 7. SA label normalisation

The SA label space is not uniform across languages, which upstream's
`pred = answer.strip()` does not account for:

| Split | Labels | Note |
|---|---|---|
| `sa_hineng` | `positive` / `negative` / `neutral` | lowercase |
| `sa_mareng` | `positive` / `negative` / `neutral` | lowercase |
| `sa_tameng` | `Positive` / `Negative` / `Neutral` / `Mixed_feelings` | **capitalised**, 4-class |
| `sa_maleng` | `Positive` / `Negative` / `Neutral` / `Mixed_feelings` | **capitalised**, 4-class |
| `sa_beneng` | `O` / `N` | **offensive-language detection**, not sentiment |

A model replying `Positive` to `sa_hineng` is correct in substance but scores zero
under exact string match. `robust` mode matches case-insensitively and returns the
split's exact gold spelling. For the single-character `O`/`N` set it requires word
boundaries, so a stray "o" in prose cannot count as a label.

`sa_beneng` deserves emphasis: it is filed under SA upstream but is binary
offensive-language detection (`'O'`=offensive, `'N'`=non-offensive). **Do not average
it with the sentiment splits.** The registry carries this note and the generated
report reproduces it.

Similarly, `lid_hineng` uses LinCE tags (`lang1`/`lang2`/`mixed`/...) while
`lid_mareng` uses `ENG`/`MAR`/`OTH`. The two LID accuracies are not on the same
scale; note it if you average them.

## 8. Unparseable rows stay in the denominator

Rows whose answer cannot be extracted are scored as wrong, never dropped. Dropping
them would let a model that answers 30% of items in a parseable format post a
inflated accuracy on that 30%.

The cost of this honesty is that a formatting failure looks like a capability
failure, so every result carries `unparsed_rate` (and `format_error_rate` for token
tasks) next to the score, and the generated report has a dedicated **Extraction
reliability** table. A score with a high unparsed rate is a formatting result, not
a capability result — read them together. This matters most for `sarvam-1-2b`,
which is a base model.

## 9. Inference layer rewritten

Upstream drives the OpenAI cloud API through `oaib.Batch` with RPM/TPM rate limits
and a Replicate branch for Llama. Neither applies to local models. Replaced with a
`Backend` abstraction over four runtimes: an OpenAI-compatible HTTP client
(vLLM/SGLang/Ollama/llama.cpp/LM Studio/TGI), in-process vLLM, transformers
(including bitsandbytes INT4), and a fake `echo` backend for CI.

New operational properties, none of which change a score: resumable
`generations.jsonl` caching keyed by row index, per-row error isolation, bounded
concurrency with exponential backoff, and full provenance in `run_meta.json`
(git commit, library versions, GPU, dataset CSV sha256, prompt fingerprint).

Because generations are cached, re-scoring under a different `--extraction` mode
is a CPU-only pass: `cmb-indic rescore --run-dir runs/<id> --extraction paper`.

## 10. Upstream runtime bugs fixed

`test_model.py` as published does not execute. These were fixed in the new
pipeline (the upstream file is left untouched):

1. `test_model.py:20` — `if api:` reads `api` before assignment; since it is
   assigned at line 21 Python treats it as local, raising `UnboundLocalError` on
   every invocation. Same pattern for `url` at line 25.
2. `test_model.py:23` — `url = os.environ.get("OPENAI_API_KEY")` puts the API key
   into the URL variable.
3. `asyncio.WindowsSelectorEventLoopPolicy()` at lines 131, 151, 174, 192 — the
   attribute does not exist off Windows, so it raises `AttributeError` on
   Linux/macOS.
4. `test_model.py:44` — `os.mkdir("./result/<dataset>")` fails because `./result`
   does not exist; needs `makedirs(..., exist_ok=True)`.
5. Task dispatch uses `task in 'lid, pos, ner'` — a substring test on a string, not
   list membership. It happens to be correct for all eight current task names, but
   a task named `s` or `mm` would silently take the wrong branch. The new code
   dispatches on the registry.

## 11. Security patches to the vendored upstream files

### `eval()` replaced with `ast.literal_eval()`

`utils.py` (`result2pred`, `compute_token_label_Metric`'s label-parsing helper)
and `test_model.py` (`--indices`) called `eval()` on stringified Python literals
read back from a dataframe column or a CLI argument -- a list of strings like
`"['Ladke', 'Ne', 'ek']"`. `eval()` executes arbitrary Python, so anything that
can influence that string (a maliciously crafted results CSV, e.g.) gets code
execution. Every other call site in the same functions already parsed the
identical data shape with `ast.literal_eval()`, which accepts only literals
(strings, numbers, tuples, lists, dicts, booleans, `None`) and raises instead of
executing on anything else.

The three `eval()` call sites were switched to `ast.literal_eval()`. For every
row this pipeline actually produces, the two are behaviourally identical --
the fields being parsed are already Python-literal reprs, never expressions --
so this changes no metric and no published number.

### `load_dataset_from_hf` can pin a dataset revision

`utils.py`'s `load_dataset_from_hf` called `load_dataset(...)` with no
`revision`, so it always resolved the dataset repo's current `main` -- whatever
that happens to be at run time (CWE-494, download without an integrity check).
It now takes an optional `revision` argument and forwards it, matching what
`src/cmb_indic/data.py` already does for the maintained download path.

The default is `None`, which resolves `main` exactly as before, so no existing
caller changes behaviour. What the patch adds is the *ability* to pin: a caller
that passes a commit sha gets a byte-reproducible fetch. Note that the shipped
pipeline does not use this function at all -- it downloads through
`scripts/download_data.py` / `src/cmb_indic/data.py`. The function is reachable
only from the vendored `test_model.py`, which per item 10 does not execute as
published.

### `requirements.txt` pins raised to patched releases

`scikit_learn` 1.0.2 -> 1.5.0 (CVE-2024-5206, `TfidfVectorizer` leaked all
training tokens through `stop_words_`), `datasets` 2.18.0 -> 5.0.1
(CVE-2026-66007, path traversal in folder-based builders) and `tqdm` 4.66.2 ->
4.66.3 (CVE-2024-34062, CLI argument injection). None of these touch a metric:
the sklearn functions this pipeline uses (`accuracy_score`, `f1_score`) are
unchanged across 1.0.2 -> 1.5.0, and the other two are a loader and a progress
bar. `docs/SECURITY.md` has the whole dependency table, including the pins that
still carry open advisories and why.

## Reproducing published numbers

`--extraction paper` restores items 2, 3, 4 and 7. Items 5, 6 and 8 remain active
because they are either transport-level (5), strictly better with no scoring
semantics attached (6), or deliberate integrity choices (8).

The practical consequence: **`--extraction paper` will not exactly reproduce the
paper's table**, both because of those remaining items and because the paper
evaluated different models (GPT-3.5/4, Llama-2/3) through a cloud API. Use `paper`
mode to understand the size of the extraction effect on *your* generations — run
both over the same cached generations and report the pair — not to claim
replication.
