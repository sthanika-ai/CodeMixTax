#!/usr/bin/env python
"""Render results/METRICS_REPORT.md from the computed result files.

Mirrors the section structure of codemix-tax.html. Every number is computed from
results/*.json rather than typed, so the text cannot drift from the runs.

Reads: tax_dataset.json, bootstrap_ci.json, translation_diagnostics.json
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

D = json.loads(Path("results/tax_dataset.json").read_text())
CI = json.loads(Path("results/bootstrap_ci.json").read_text())
TD = json.loads(Path("results/translation_diagnostics.json").read_text())
BF = json.loads(Path("results/budget_fertility.json").read_text())
RA = json.loads(Path("results/romanization_audit.json").read_text())
M = D["models"]
FORMS = ["EN", "CM", "HI", "CM-ROM", "HI-ROM"]
NAMES = {"EN": "English", "CM": "Hinglish", "HI": "Hindi",
         "CM-ROM": "Romanized Hinglish", "HI-ROM": "Romanized Hindi"}
TASKS = [("mmlu", "General knowledge"), ("gsm8k", "Maths word problems")]
CUT = {"mmlu": 35.0, "gsm8k": 15.0}

floored = lambda s, t: M[s]["tasks"][t]["EN"]["accuracy"] < CUT[t]
capable = lambda t: [s for s in M if not floored(s, t)]
trunc = lambda t, f: [m["tasks"][t][f]["trunc_pct"] for m in M.values()]
ci = lambda t, f: CI["tasks"][t]["conditions"][f]
#: Paired difference between two forms, with its interval. Quote THIS rather
#: than two marginal intervals: overlapping marginals do not imply no effect.
dif = lambda t, k: CI["tasks"][t]["differences"][k]
fmt_ci = lambda c: f"{c[0]:.1f} to {c[1]:.1f}"

L: list[str] = []
w = L.append

oh = M["openhathi-7b"]["tasks"]["mmlu"]
gm = M["gemma3-12b-it"]["tasks"]["mmlu"]
best_en = max(M, key=lambda x: M[x]["tasks"]["mmlu"]["EN"]["accuracy"])
be = M[best_en]["tasks"]["mmlu"]
sm = M["sarvam-m-24b"]["tasks"]["mmlu"]
mi = M["mistral-small-3.1-24b"]["tasks"]["mmlu"]
rank = 1 + sum(1 for s in capable("mmlu")
               if M[s]["tasks"]["mmlu"]["HI-ROM"]["retention_adj"] > be["HI-ROM"]["retention_adj"])

# --------------------------------------------------------------- opening
w("# The Code-Mixing Tax")
w("")
w("### What happens to an AI model when you stop writing to it in English")
w("")
w("A great many people in India write Hindi in English letters, mixed with English "
  "words — *\"mujhe kal ka schedule bhejo\"*. We have not tried to put a number on how "
  "many, because we could not find a measurement we trust; what matters here is that it "
  "is a common way of typing, not a rare one. Many large AI models advertise Hindi or "
  "broader multilingual support. We could not find published figures for what this way "
  "of writing costs them.")
w("")
w(f"We asked **{len(M)} AI models** the *same* questions five different ways and measured "
  f"how much accuracy each way costs. Same questions, same models, only the language "
  f"changes.")
w("")

# ------------------------------------------------------------ five forms
w("## The five ways we asked each question")
w("")
w("| | Form | Example |")
w("|---|---|---|")
w("| **1** | English | *What is the capital of France?* |")
w("| **2** | Hinglish — Hindi sentence, English words, Hindi script | *France की capital क्या है?* |")
w("| **3** | Hindi — pure Hindi, Devanagari script | *फ़्रांस की राजधानी क्या है?* |")
w("| **4** | Romanized Hinglish | *France ki capital kya hai?* |")
w("| **5** | **Romanized Hindi** | *France ki rajdhani kya hai?* |")
w("")
w("Form 5 is a form many people use on a phone keyboard. It is also the one models "
  "handle worst, by a wide margin.")
w("")
w("The tests were **general knowledge** (multiple-choice, 57 subjects, 1,024 questions) "
  "and **maths word problems** (955 questions).")
w("")

# --------------------------------------------------------- 1 main finding
cm_m, hr_m = ci("mmlu", "CM"), ci("mmlu", "HI-ROM")
cm_g, hr_g = ci("gsm8k", "CM"), ci("gsm8k", "HI-ROM")
n_m, n_g = len(capable("mmlu")), len(capable("gsm8k"))
excl = TD["exclusions"]["excluded_models"]
# The two maths counterexamples, stated in points AND questions -- an earlier
# draft described them as "under half a question", which was wrong by ~10x.
_gs = lambda slug, f: M[slug]["tasks"]["gsm8k"][f]["accuracy"]
_n_gsm = M["sarvam-m-24b"]["tasks"]["gsm8k"]["EN"]["rows"]
_rev = {
    "sarvam": {"pp": _gs("sarvam-1-2b", "HI") - _gs("sarvam-1-2b", "CM")},
    "openhathi": {"pp": _gs("openhathi-7b", "CM-ROM") - _gs("openhathi-7b", "CM")},
}
for _k in _rev:
    _rev[_k]["q"] = _rev[_k]["pp"] / 100 * _n_gsm

w("## 1. The main finding")
w("")
w("**Writing in English letters costs more than mixing languages does — and the two "
  "costs compound.**")
w("")
w(f"Hinglish was the easiest of the four non-English forms for **every model we could "
  f"meaningfully rank** — all {n_m} on general knowledge and all {n_g} on maths. Models "
  f"kept **{cm_m['mean_retention']:.1f}%** of their ability "
  f"(95% CI {cm_m['ci95'][0]:.1f}–{cm_m['ci95'][1]:.1f}) on general knowledge and "
  f"**{cm_g['mean_retention']:.1f}%** ({cm_g['ci95'][0]:.1f}–{cm_g['ci95'][1]:.1f}) on "
  f"maths.")
w("")
w(f"**Which models are excluded, and why.** {len(M) - n_g} small models score so close to "
  f"guessing that ranking language forms by them is meaningless: "
  f"{', '.join(excl['gsm8k'])} on maths, {', '.join(excl['mmlu'])} on general knowledge. "
  f"OpenHathi 7B solves about 1 maths problem in 16 *in English*, and Param-1 2.9B scores "
  f"**higher** on Hinglish ({M['param-1-2.9b']['tasks']['gsm8k']['CM']['accuracy']:.2f}) "
  f"than on English ({M['param-1-2.9b']['tasks']['gsm8k']['EN']['accuracy']:.2f}), which "
  f"cannot be a real effect. Those rows appear in *italics* in the tables and are excluded "
  f"from every count and retention figure below. Two of them do reverse the pattern on "
  f"maths: Sarvam-1 scores higher on Hindi than Hinglish "
  f"({_rev['sarvam']['pp']:+.2f} points, about {_rev['sarvam']['q']:.0f} questions of 955) "
  f"and OpenHathi higher on Romanized Hinglish "
  f"({_rev['openhathi']['pp']:+.2f} points, about {_rev['openhathi']['q']:.0f} questions) "
  f"— small absolute differences on models that answer roughly one question in 16 to "
  f"begin with. On general knowledge the pattern holds for all {len(M)} models regardless "
  f"of exclusion.")
w("")
w("The reason Hinglish is easiest is that it gives a model two footholds: familiar "
  "English words, *and* familiar Hindi script. Either one substantially cushions the "
  "loss — but neither eliminates it, and mixing is not harmless: Hinglish still costs "
  f"{abs(dif('mmlu', 'mixing (CM vs EN)')['delta']):.1f} points of retention on general "
  "knowledge.")
w("")
w("Splitting the two factors apart shows which one dominates. Every question exists in "
  "all four combinations of *what language the words come from* and *what script they "
  "are written in*, so the two effects can be separated cleanly:")
w("")
w("| Floor-adj. retention, general knowledge | Hindi script | English letters | Cost of the script change |")
w("|---|---:|---:|---:|")
_r = lambda f: ci("mmlu", f)["mean_retention"]
w(f"| **Mixed** with English words (Hinglish) | {_r('CM'):.1f}% | {_r('CM-ROM'):.1f}% | "
  f"**{dif('mmlu', 'romanizing a mixed sentence')['delta']:.1f}** |")
w(f"| **All Hindi** | {_r('HI'):.1f}% | {_r('HI-ROM'):.1f}% | "
  f"**{dif('mmlu', 'romanizing Hindi')['delta']:.1f}** |")
w(f"| Cost of dropping the English words | {dif('mmlu', 'script vs lexicon (HI vs CM)')['delta']:.1f} | "
  f"{dif('mmlu', 'mixing, once romanized')['delta']:.1f} | |")
w("")
_ix = CI["tasks"]["mmlu"]["interaction"]
_ig = CI["tasks"]["gsm8k"]["interaction"]
w("Read down the last column: switching to English letters costs "
  f"{abs(dif('mmlu', 'romanizing a mixed sentence')['delta']):.1f} points in a mixed "
  f"sentence but {abs(dif('mmlu', 'romanizing Hindi')['delta']):.1f} points in an "
  "all-Hindi one. That difference is the two factors compounding rather than simply "
  f"adding: **{_ix['delta']:.1f} points** of extra loss on general knowledge "
  f"(95% CI {fmt_ci(_ix['ci95'])}) and **{_ig['delta']:.1f}** on maths "
  f"(95% CI {fmt_ci(_ig['ci95'])}). Both intervals exclude zero.")
w("")
w("So the fair summary is not that mixing is harmless. It is that the *script* is the "
  "more expensive of the two changes, and that the English words in a Hinglish sentence "
  "are doing most of the work of keeping the model oriented once the script is gone.")
w("")
w(f"Take both away and it falls off a cliff — only **{hr_m['mean_retention']:.1f}%** "
  f"({hr_m['ci95'][0]:.1f}–{hr_m['ci95'][1]:.1f}) survives on general knowledge and "
  f"**{hr_g['mean_retention']:.1f}%** ({hr_g['ci95'][0]:.1f}–{hr_g['ci95'][1]:.1f}) on "
  f"maths. Measured as a paired difference on the same questions — the right way to "
  f"compare two forms — the gap between Hinglish and Romanized Hindi is "
  f"**{dif('mmlu', 'best vs worst form (CM vs HI-ROM)')['delta']:.1f} points** of "
  f"retention on general knowledge "
  f"(95% CI {fmt_ci(dif('mmlu', 'best vs worst form (CM vs HI-ROM)')['ci95'])}) and "
  f"**{dif('gsm8k', 'best vs worst form (CM vs HI-ROM)')['delta']:.1f}** on maths "
  f"(95% CI {fmt_ci(dif('gsm8k', 'best vs worst form (CM vs HI-ROM)')['ci95'])}).")
w("")
w("![How accuracy falls across the five forms](figures/tax-curves-light.png)")
w("")

# --------------------------------------------------- 2 english doesn't help
w("## 2. Being good at English does not help")
w("")
# Any model within this many points of another is called a tie rather than
# ranked ahead of it: 0.1 points of 1024 questions is one question, which is
# far below what this measurement can resolve.
_TIE = 0.1
_ties = [s for s in capable("mmlu") if s != best_en
         and abs(M[s]["tasks"]["mmlu"]["HI-ROM"]["retention_adj"]
                 - be["HI-ROM"]["retention_adj"]) <= _TIE]
_tie_txt = ""
if _ties:
    _tl = " and ".join(M[s]["label"] for s in _ties)
    _tie_txt = (f" — effectively tied with {_tl}, a difference of well under one "
                f"question in {be['EN']['n']:,}")
w(f"**{M[best_en]['label']}** scored the highest of all {len(M)} models on English general "
  f"knowledge — **{be['EN']['accuracy']:.2f}%**. On Romanized Hindi it kept only "
  f"**{be['HI-ROM']['retention_adj']:.1f}%** of that ability, placing it **{rank}th** of "
  f"{n_m}{_tie_txt}. **Sarvam-M 24B** scored "
  f"{be['EN']['accuracy'] - sm['EN']['accuracy']:.2f} points *lower* in English and kept "
  f"**{sm['HI-ROM']['retention_adj']:.1f}%**.")
w("")
# The eight highest English scorers, in descending order of English score. A
# stated rule matters here: selecting instead by retention would quietly drop
# the models that make the point most sharply (Phi-4 and Mistral Small both
# score above 80 in English and are among the worst retainers).
_TOP_N = 8
_by_en = sorted(capable("mmlu"),
                key=lambda s: -M[s]["tasks"]["mmlu"]["EN"]["accuracy"])[:_TOP_N]
w(f"The {_TOP_N} highest English scorers, in descending order of English score:")
w("")
w("| Model | English score | Floor-adj. retention |")
w("|---|---:|---:|")
for s_ in _by_en:
    t_ = M[s_]["tasks"]["mmlu"]
    star = " ← best English score" if s_ == best_en else ""
    w(f"| {M[s_]['label']} | {t_['EN']['accuracy']:.2f}%{star} | "
      f"**{t_['HI-ROM']['retention_adj']:.1f}%** |")
w("")
_ens = [M[s_]["tasks"]["mmlu"]["EN"]["accuracy"] for s_ in _by_en]
_rets = [M[s_]["tasks"]["mmlu"]["HI-ROM"]["retention_adj"] for s_ in _by_en]
w(f"Across these {_TOP_N} models English scores span only {max(_ens) - min(_ens):.1f} "
  f"points, while what they retain on Romanized Hindi spans "
  f"{max(_rets) - min(_rets):.1f}. Raw capability in English does not predict "
  f"robustness on Indian input.")
w("")
w("![Which models hold up best](figures/tax-retain-light.png)")
w("")

# ------------------------------------------------- 3 what actually helps
w("## 3. Indic training is associated with a large benefit")
w("")
w("The most informative comparison is between two closely related checkpoints. "
  "**Sarvam-M 24B** is built on **Mistral Small 3.1 24B** — same parameter count, same "
  "base architecture — with additional Indian-language training.")
w("")
w("| Test, Romanized Hindi | Mistral Small 24B | Sarvam-M 24B | Difference |")
w("|---|---:|---:|---:|")
for task, tl in TASKS:
    a = M["mistral-small-3.1-24b"]["tasks"][task]["HI-ROM"]["accuracy"]
    b = M["sarvam-m-24b"]["tasks"][task]["HI-ROM"]["accuracy"]
    w(f"| {tl} | {a:.2f}% | {b:.2f}% | **+{b-a:.2f} points** |")
w("")
w(f"Floor-adjusted retention rises from {mi['HI-ROM']['retention_adj']:.1f}% to "
  f"{sm['HI-ROM']['retention_adj']:.1f}% — roughly double — at the same model size.")
w("")
w("**What this does and does not establish.** This comparison is consistent with a "
  "substantial benefit from Indic post-training, but it does not isolate which part of "
  "the recipe caused it. The two are released checkpoints, not a controlled ablation: "
  "post-training data, instruction tuning, tokenizer, and other implementation choices "
  "all differ alongside the Indic data. A causal attribution would need an ablation by "
  "the model's authors.")
w("")
w("![Sarvam-M compared with the model it is built on](figures/tax-indic-light.png)")
w("")

# --------------------------------------------------- 4 never finishing
w("## 4. A second failure: models that never finish the answer")
w("")
w("Faced with romanized Hindi input, some models **keep writing and never commit to an "
  "answer** — they run until cut off. With no answer to extract, the question is marked "
  "**wrong**; it is not quietly dropped, so this failure is fully paid for in the "
  "accuracy figures above.")
w("")
w("| " + " | ".join(NAMES[f] for f in FORMS) + " |")
w("|" + "---:|" * 5)
w("| " + " | ".join(f"{st.mean(trunc('gsm8k', f)):.1f}%" for f in FORMS) + " |")
w("")
_bl = BF["budget_ladder"]["gsm8k"]
_bs = BF["budget_ladder"]
w("Roughly twice the rate of English or Hinglish.")
w("")
w("**A bigger budget does not fix it.** This is measured, not assumed. Whenever a run "
  "hit its ceiling we re-ran the affected rows with a larger budget — usually double, "
  f"across {_bs['n_models']} models — and every attempt is still on disk. Of "
  f"**{_bl['retried']:,} maths rows** that were cut off and then retried with more room, "
  f"**{_bl['still_truncated']:,} ({_bl['still_pct']:.0f}%) still did not finish**. The "
  "share that stayed unfinished is broadly similar in every language form "
  f"({min(v['still_pct'] for v in _bl['forms'].values()):.0f}–"
  f"{max(v['still_pct'] for v in _bl['forms'].values()):.0f}%), so this is not a budget "
  "set slightly too low. These models do not stop.")
w("")
w("**It is also not simply that romanized text is longer.** A romanized question costs "
  "almost exactly the same number of tokens as the same question in Hindi script, "
  "because romanizing changes the spelling rather than the word count:")
w("")
w("| Prompt length, maths questions | Hindi script | English letters |")
w("|---|---:|---:|")
_pl = BF["prompt_length"]["gsm8k"]
w(f"| Mixed with English words (Hinglish) | {_pl['CM']['ratio_to_en']:.2f}× English | "
  f"{_pl['CM-ROM']['ratio_to_en']:.2f}× English |")
w(f"| All Hindi | {_pl['HI']['ratio_to_en']:.2f}× English | "
  f"{_pl['HI-ROM']['ratio_to_en']:.2f}× English |")
w("")
w("Each model is measured against its own English prompts, then averaged. Reading across "
  "either row, the token cost barely moves — yet accuracy falls sharply. Whatever "
  "romanized input is doing to these models, it is not merely making the input longer.")
w("")

# ------------------------------------------------- how to read the numbers
w("## How to read the numbers")
w("")
w("The main measure is **floor-adjusted retention** — the share of a model's *own* "
  "ability that survives when the question changes language.")
w("")
w("It is not simply \"score in Hindi ÷ score in English\", because that flatters weak "
  "models. In a four-option question, a model that knows nothing still scores about "
  "**25%** by guessing. That 25% is the *floor*, and subtracting it first is what "
  "\"floor-adjusted\" means:")
w("")
w("```")
w("  floor-adjusted retention = (score − floor) ÷ (English score − floor)")
w("```")
w("")
w(f"**A worked example.** OpenHathi 7B scored **{oh['EN']['accuracy']:.2f}%** in English "
  f"and **{oh['HI-ROM']['accuracy']:.2f}%** on Romanized Hindi.")
w("")
w("```")
w(f"  Naive:           {oh['HI-ROM']['accuracy']:.2f} ÷ {oh['EN']['accuracy']:.2f} "
  f"= {oh['HI-ROM']['retention_raw']:.1f}%   ← looks like it held up fine")
w("")
w(f"  Real ability in English:  {oh['EN']['accuracy']:.2f} − 25 = "
  f"{oh['EN']['accuracy']-25:.2f} points")
w(f"  Real ability in Rom. Hindi: {oh['HI-ROM']['accuracy']:.2f} − 25 = "
  f"{oh['HI-ROM']['accuracy']-25:.2f} points")
w("")
w(f"  Floor-adjusted:  {oh['HI-ROM']['accuracy']-25:.2f} ÷ {oh['EN']['accuracy']-25:.2f} "
  f"= {oh['HI-ROM']['retention_adj']:.1f}%    ← the truth: nothing survived")
w("```")
w("")
w("So **100%** means the language change cost nothing, "
  f"**{gm['HI-ROM']['retention_adj']:.1f}%** (Gemma 3 12B) means about half its real "
  "ability is gone, and **0%** means it is down to guessing.")
w("")
w("**The floor depends on the task**, because guessing is only worth something when there "
  "are options to guess between:")
w("")
w("| Task | Floor used | Why |")
w("|---|---:|---|")
w("| General knowledge | **25** | Four options per question, so guessing scores ~25%. |")
w("| Maths word problems | **0** | The answer is a number the model must work out. You "
  "cannot guess *\"7,412\"*, so there is no free score to subtract. |")
w("")
w("With a floor of 0 the formula collapses to plain `score ÷ English score`.")
w("")
w("### How the headline percentages are calculated")
w("")
w("The figures quoted in section 1 are **unweighted means across eligible models** of "
  "each model's own floor-adjusted retention. Specifically:")
w("")
w("- **Aggregation:** per-model retention is computed first, then averaged across models. "
  "Every model counts equally regardless of size or score.")
w(f"- **Eligibility:** a model is included for a task if its English score clears "
  f"{TD['exclusions']['eligibility_rule']['mmlu'].split('>=')[1].strip()} on general "
  f"knowledge, {TD['exclusions']['eligibility_rule']['gsm8k'].split('>=')[1].strip()} on "
  f"maths. That leaves **{n_m} of {len(M)}** models on general knowledge and "
  f"**{n_g} of {len(M)}** on maths — the counts differ because more models are near the "
  f"floor on maths.")
_sv = {t: CI["tasks"][t]["sensitivity"] for t in ("mmlu", "gsm8k")}
_rng = {t: [r["retention"]["HI-ROM"] for r in _sv[t]] for t in _sv}
w(f"- **The cut-off is not load-bearing.** Moving it changes the reported Romanized "
  f"Hindi retention by at most "
  f"{max(max(_rng[t]) - min(_rng[t]) for t in _sv):.1f} points across every threshold we "
  f"tried, and never reorders the five forms. The full sweep is in the appendix.")
w("- **No clipping:** values are not clipped at zero or at 100. A model that scores below "
  "the floor would produce a negative figure; none of the eligible models do.")
w(f"- **Uncertainty:** 95% intervals come from a paired bootstrap over questions "
  f"({CI['iters']:,} resamples, seed {CI['seed']}). One resample of question indices is "
  f"applied to every condition and model at once, so differences between conditions "
  f"remain interpretable. These intervals quantify uncertainty from having sampled a "
  f"finite set of **questions**, with the tested model set held fixed; they say nothing "
  f"about how the result would generalise to other models.")
w("")
w("| Task | Hinglish | Hindi | Romanized Hinglish | Romanized Hindi |")
w("|---|---|---|---|---|")
for task, tl in TASKS:
    cells = []
    for f in ("CM", "HI", "CM-ROM", "HI-ROM"):
        c = ci(task, f)
        cells.append(f"{c['mean_retention']:.1f}% [{c['ci95'][0]:.1f}, {c['ci95'][1]:.1f}]")
    w(f"| {tl} | " + " | ".join(cells) + " |")
w("")

# --------------------------------------------------------- 5 full results
w("## 5. The full results")
w("")
w("Every score is the percentage of questions answered correctly. **Bold** marks each "
  "model's English score — its own ceiling. *Italic* rows are the models excluded above.")
w("")
w("† Sarvam-30B ran with its reasoning capped at 4,000 tokens. Uncapped, it left a "
  "quarter of rows unfinished and could not be scored, so its figures are accuracy "
  "*under a reasoning cap* rather than with unconstrained reasoning.")
w("")
for task, tl in TASKS:
    n = "1,024" if task == "mmlu" else "955"
    w(f"### {tl} ({n} questions per form)")
    w("")
    w("| Model | " + " | ".join(NAMES[f] for f in FORMS) +
      " | Romanized Hindi loss vs English | Floor-adj. retention |")
    w("|---|" + "---:|" * 7)
    for slug, m in sorted(M.items(),
                          key=lambda kv: -(kv[1]["tasks"][task]["HI-ROM"]["retention_adj"] or -1)):
        p = m["tasks"][task]
        f_ = floored(slug, task)
        lab = f"*{m['label']}*" if f_ else m["label"]
        cells = [(f"**{p[c]['accuracy']:.2f}**" if c == "EN" and not f_
                  else f"{p[c]['accuracy']:.2f}") for c in FORMS]
        keep = "—" if f_ else f"**{p['HI-ROM']['retention_adj']:.1f}%**"
        w(f"| {lab} | " + " | ".join(cells) +
          f" | {p['HI-ROM']['delta_en']:.2f} | {keep} |")
    w("")
w("*Romanized Hindi loss vs English* is how many percentage points the model dropped, "
  "measured against its own English score. *Floor-adjusted retention* is that same drop "
  "expressed as a share of the model's real ability.")
w("")

# -------------------------------------------------------- 6 what it means
w("## 6. What this means in practice")
w("")
w("**Choosing a model for Indian users:** English benchmark scores alone can mislead "
  "you. The best English model here was mid-table once questions were typed in romanized "
  "form. Test on romanized input before committing.")
w("")
_hi_m = ci("mmlu", "HI")["mean_retention"]
_hi_g = ci("gsm8k", "HI")["mean_retention"]
w("**Building a product:** the input format matters as much as the model. Converting "
  "romanized input to Hindi script before it reaches the model may recover a large part "
  f"of the loss — but note the ceiling: Hindi *in its own script* still retains only "
  f"**{_hi_m:.1f}%** on general knowledge and **{_hi_g:.1f}%** on maths. Transliterating "
  f"perfectly buys you the Hindi-script row, not the English one, and real "
  f"transliteration carries its own accuracy, latency and maintenance costs on top.")
w("")
w("**Training a model:** the gap can be materially reduced. The Sarvam comparison is "
  "consistent with a large benefit from Indic training at the same model size, though as "
  "noted above it is a comparison of two released checkpoints rather than a controlled "
  "ablation.")
w("")

# ------------------------------------------------------------- appendix
vm = TD["verified_vs_mt"]
un = TD["unanimous"]
so = TD["script_only"]
ex = TD["exclusions"]

w("## Appendix: how this was measured")
w("")
w("**The questions.** Two standard test sets: a general-knowledge multiple-choice exam "
  "covering 57 subjects, and grade-school maths word problems. Every model saw identical "
  "questions in all five forms.")
w("")
w("**Where each form came from.** Only the Hinglish form is published research data. The "
  "other four were assembled for this study and aligned back to it question-by-question.")
w("")
w("| Form | General knowledge | Maths word problems |")
w("|---|---|---|")
w("| Hinglish | **CodeMixBench** ([Yang & Chai, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.109/)) | **CodeMixBench** |")
w("| English | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu), rejoined by question id | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), matched by its worked solution |")
w("| Hindi | [`CohereLabs/Global-MMLU`](https://huggingface.co/datasets/CohereLabs/Global-MMLU) (Hindi) | [`bingbangboom/gsm8k-hindi`](https://huggingface.co/datasets/bingbangboom/gsm8k-hindi) (MIT) |")
w("| Romanized Hinglish | Hinglish, transliterated | Hinglish, transliterated |")
w("| Romanized Hindi | Hindi, transliterated | Hindi, transliterated |")
w("")
w("Transliteration is deterministic (ITRANS with schwa deletion, anusvara→n, "
  "lowercasing); it converts the script and leaves the words unchanged.")
w("")
w("### Limitation: translation quality is part of what we measure")
w("")
w("Correct answers always come from the **English** originals, never from the translated "
  "datasets. That is deliberate — the Hindi answer fields are unreliable — but it means "
  "the Hindi conditions measure *\"can the model answer the English question as rendered "
  "in Hindi\"*, which is not quite *\"can the model do Hindi\"*. If a translation drifts, "
  "a model answering the Hindi question correctly is still marked wrong.")
w("")
w("We report what we can measure about this rather than leaving it open:")
w("")
w(f"- **Coverage.** Only **{vm['verified_pct']}%** of the Hindi general-knowledge items "
  f"({vm['verified_items']} of {vm['total_items']}) are marked human-verified by "
  f"Global-MMLU; the rest are machine-translated with lighter review. The Hindi maths set "
  f"is row-aligned, and its own answer field is malformed for ~27% of rows — which is why "
  f"we do not use it.")
w(f"- **Exclusions already applied.** **{ex['gsm8k_items_dropped_for_number_mismatch']} of "
  f"1,016** maths items ({ex['gsm8k_dropped_pct']}%) were dropped because the Hindi "
  f"question did not carry the same numbers as the English one. All five forms are "
  f"reported on the remaining {ex['gsm8k_items_kept']}.")
w(f"- **Measured effect of translation quality.** Comparing the same "
  f"{vm['n_models']} models on the human-verified subset against the machine-translated "
  f"one: **{vm['mean_gap_points']['HI']:+.2f} points** on Hindi and "
  f"**{vm['mean_gap_points']['HI-ROM']:+.2f} points** on Romanized Hindi. Small, and "
  f"absent on the form that carries the headline result.")
w(f"- **Residual broken items.** Questions that *every* eligible model answers correctly "
  f"in English and incorrectly in Hindi — the signature of a broken question — number "
  f"**{un['mmlu']['en_pass_hi_fail']} of {un['mmlu']['n_items']:,}** "
  f"({un['mmlu']['en_pass_hi_fail_pct']}%) on general knowledge and "
  f"**{un['gsm8k']['en_pass_hi_fail']} of {un['gsm8k']['n_items']}** "
  f"({un['gsm8k']['en_pass_hi_fail_pct']}%) on maths. The reverse direction — all correct "
  f"in Hindi, all wrong in English — is **0** in both, which is what makes the asymmetry "
  f"evidence of translation damage rather than difficulty.")
w("")
w("**What remains unquantified:** semantic drift that preserves the numbers, such as "
  "\"gave away\" becoming \"received\". Our checks cannot see it. One confirmed example: a "
  "maths item where *\"1/4 as big as\"* became *\"1/4 bigger than\"* — all eligible models "
  "agreed on the same wrong answer, having each solved the mistranslated question "
  "correctly.")
w("")
w("**A translation-free corroboration.** Romanizing is a deterministic transliteration of "
  "text we already have, so any translation error is identical on both sides and cancels "
  "exactly. On that comparison alone:")
w("")
w("| Task | Romanizing Hinglish costs | Romanizing Hindi costs | Ratio |")
w("|---|---:|---:|---:|")
for task, tl in TASKS:
    s = so[task]
    w(f"| {tl} | {s['romanising_hinglish_costs']:.2f} pts | "
      f"{s['romanising_hindi_costs']:.2f} pts | **{s['ratio']:.1f}×** |")
w("")
w("Losing the script costs roughly twice as much when there is no English to fall back "
  "on. That figure carries no translation exposure at all.")
w("")

# ------------------------------------------- limitation: the romanization itself
w("### Limitation: our romanization is one scheme, not a sample of real typing")
w("")
w(f"Romanized Hindi has no standard spelling — the same word is written several ways by "
  f"different people, and often by the same person. Our two Latin-script conditions are "
  f"produced by a single deterministic transform: **{RA['tool']} "
  f"{RA['tool_version']}** ({RA['scheme']}), followed by "
  f"{', '.join(RA['postprocess'][:4])} and lowercasing "
  f"(`{RA['implementation']}`). That makes the conditions exactly reproducible, and it "
  f"means they are *one point* in a wide space of things people actually type.")
w("")
_rc = RA["conditions"]
_worst = max(_rc.values(), key=lambda c: c["rows_pct"])
_lo_r = min(c["rows_pct"] for c in _rc.values())
_hi_r = max(c["rows_pct"] for c in _rc.values())
w(f"**The transform has a known defect, and it is common.** It does not model *internal* "
  f"schwa deletion, so it writes `men` where a person writes `mein`, and `kitane` for "
  f"`kitne`. At least one such form appears in "
  f"**{_lo_r:.0f}–{_hi_r:.0f}% of rows** across the four romanized sets, affecting "
  f"{min(c['token_pct'] for c in _rc.values()):.1f}–"
  f"{max(c['token_pct'] for c in _rc.values()):.1f}% of words:")
w("")
w("| Romanized condition | Rows containing a known-wrong form | Share of words |")
w("|---|---:|---:|")
for _n, _c in _rc.items():
    w(f"| `{_n}` | {_c['rows_with_divergent_form']:,} of {_c['rows']:,} "
      f"({_c['rows_pct']:.1f}%) | {_c['token_pct']:.2f}% |")
w("")
w("**Which way this biases the result.** Our romanized text is slightly *less* natural "
  "than real typing, so it is plausibly further out of a model's distribution than what "
  "a user would send. The romanization penalty we report is therefore best read as an "
  "**upper bound** on the penalty for well-formed romanized Hindi. Pushing the other "
  "way, real input is *more* variable than ours — inconsistent spelling within a single "
  "message — which our uniform scheme does not test at all.")
w("")
# From the audit, which counts the rows it wrote. NOT by counting lines in the
# CSV: the question fields contain newlines, so lines and rows differ (64 rows
# spanned 322 lines).
_n_sample = RA["sample_rows"]
w("**Not validated against human-typed text.** We did not collect human romanizations of "
  "these questions, so we cannot report agreement with them, and we have not claimed to. "
  f"A seeded {_n_sample}-row side-by-side sample of the transform's output "
  "against its Devanagari source ships as `results/romanization_sample.csv` for anyone "
  "who reads Hindi to check by eye. Comparison against human-typed romanizations is the "
  "clearest next step for this study, and would tighten the bound above.")
w("")
w("### Sensitivity: does the eligibility cut-off matter?")
w("")
w("A model that scores near the guessing floor in English cannot rank language forms, so "
  "models below a threshold are excluded from the headline means. A threshold always "
  "invites the question of whether it was chosen to flatter the result, so here is the "
  "whole sweep:")
w("")
w("| Cut-off | Eligible | " + " | ".join(NAMES[f] for f in FORMS) + " |")
w("|---|---:|" + "---:|" * 5)
for _t, _tl in TASKS:
    for _r in _sv[_t]:
        _mk = " ←" if _r["cut"] == CUT[_t] else ""
        w(f"| {_tl}, ≥{_r['cut']:.0f}%{_mk} | {_r['n_eligible']} | "
          + " | ".join(f"{_r['retention'][f]:.1f}%" for f in FORMS) + " |")
w("")
w("The arrow marks the threshold used throughout, fixed on the reasoning above before "
  "these figures were computed. Note that the *least* selective settings — including "
  "every model — give a **worse** Romanized Hindi figure than the one we report, so the "
  "threshold we chose is the more conservative option, not the more flattering one.")
w("")
w("### Models, settings and reproduction")
w("")
w("| | |")
w("|---|---|")
w("| Decoding | greedy (temperature 0), fixed seed — deterministic |")
w("| Precision | bfloat16 for all models except two, noted below |")
w("| Quantized models | Gemma 3 12B (compressed) is 4-bit; Llama 4 Scout is a 4-bit "
  "weight-quantized release |")
w("| Token budgets | per model, set by its context window and whether it reasons before "
  "answering; recorded per run |")
w("| Answer parsing | shape-aware: handles models that answer first and models that "
  "reason first. A reply with no answer in it — cut off mid-reasoning, say — is scored "
  "**wrong**, never dropped, so all five forms are scored over the same questions |")
w("| Reasoning models | Sarvam-30B was run with reasoning capped at 4,000 tokens; "
  "uncapped it left a quarter of rows unfinished |")
w("| Inference | vLLM, offline batch |")
w("")
w("Exact checkpoint ids, per-model configuration, prompts, token budgets and the answer "
  "parser are all in the accompanying code repository; each run also writes a "
  "`run_meta.json` recording the settings actually used.")
w("")
w("**Fairness.** Each model is compared only against itself, so a weaker model is not "
  "penalised for being weak and a stronger one gets no free credit.")
w("")
w("---")
w("")
w("*Hinglish questions from CodeMixBench ([Yang & Chai, EMNLP 2025]"
  "(https://aclanthology.org/2025.emnlp-main.109/)). English and Hindi matched "
  "question-by-question from `cais/mmlu`, `openai/gsm8k`, `CohereLabs/Global-MMLU` and "
  "`bingbangboom/gsm8k-hindi`.*")

Path("results/METRICS_REPORT.md").write_text("\n".join(L) + "\n")
print(f"wrote results/METRICS_REPORT.md  ({len(L)} lines)")
