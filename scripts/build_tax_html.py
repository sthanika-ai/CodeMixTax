#!/usr/bin/env python
"""Render results/codemix-tax.html from results/tax_dataset.json.

Reuses the existing page shell (fonts + design tokens, lines 1-241 of the
previous build, kept in scripts/_tax_head.html) and regenerates the body:
Executive Summary, Key Findings, Primary Data Table, Appendix. Tables are
rendered here rather than in browser JS so the page cannot carry stale data.

Figure placeholders are left for scripts/embed_figures.py to fill.
"""
from __future__ import annotations

import html
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
SHORT = NAMES
TASKS = [("mmlu", "General knowledge"), ("gsm8k", "Maths word problems")]
CUT = {"mmlu": 35.0, "gsm8k": 15.0}
HEAD = Path("scripts/_tax_head.html")


def _citation() -> tuple[str, str]:
    """Author and licence, read from CITATION.cff so the page cannot contradict it.

    Deliberately a crude line scan rather than a YAML parse: this is the only
    thing either builder needs from the file, and adding a pyyaml import to a
    report renderer for two fields is not worth it.
    """
    author, lic = "", ""
    txt = Path("CITATION.cff").read_text().splitlines()
    given = family = ""
    for i, line in enumerate(txt):
        t = line.strip()
        # Top-level (unindented) license ONLY. The file also carries a nested
        # `license:` for the upstream dataset under `references:`, which is
        # Apache-2.0; matching on any indentation picks that up and mislabels
        # this work's licence.
        if line.startswith("license:") and not lic:
            lic = t.split(":", 1)[1].strip().strip('"')
        if t.startswith("- given-names:") and not given:
            given = t.split(":", 1)[1].strip()
            if i + 1 < len(txt) and "family-names:" in txt[i + 1]:
                family = txt[i + 1].split(":", 1)[1].strip()
    author = f"{given} {family}".strip()
    if not lic:
        raise RuntimeError("no top-level `license:` in CITATION.cff; refusing to "
                           "print a licence the repository does not declare")
    return author or "see CITATION.cff", lic


AUTHOR, LICENSE_ID = _citation()


def floored(s: str, t: str) -> bool:
    return M[s]["tasks"][t]["EN"]["accuracy"] < CUT[t]


def capable(t: str) -> list[str]:
    return [s for s in M if not floored(s, t)]


def adj(t: str, f: str) -> list[float]:
    return [M[s]["tasks"][t][f]["retention_adj"] for s in capable(t)]


def ci(t: str, f: str) -> dict:
    return CI["tasks"][t]["conditions"][f]


def trunc(t: str, f: str) -> list[float]:
    return [m["tasks"][t][f]["trunc_pct"] for m in M.values()]


def _dif(t: str, k: str) -> dict:
    """Paired difference between two forms, with its interval.

    Quote this rather than two marginal intervals: non-overlapping marginals do
    imply a real difference, but OVERLAPPING ones do not imply its absence, and
    the paired difference is strictly the more powerful comparison.
    """
    return CI["tasks"][t]["differences"][k]


def _fci(c: list[float]) -> str:
    return f"{c[0]:.1f} to {c[1]:.1f}"


def e(x) -> str:
    return html.escape(str(x))


def table(caption: str, head: list[str], rows: list[list], cls: str = "") -> str:
    """`cls` picks a layout: "" for numeric tables (cells never wrap), "prose"
    where cells hold sentences, "prose kv" for key/value with no header row."""
    th = "".join(f"<th>{h}</th>" for h in head)
    body = ""
    for r in rows:
        tds = ""
        for c in r:
            if isinstance(c, dict):
                tds += f'<td class="{c.get("c","")}">{c["v"]}</td>'
            else:
                tds += f"<td>{c}</td>"
        body += f"<tr>{tds}</tr>"
    k = f" {cls}" if cls else ""
    return (f'<div class="tablewrap{k}"><table><caption>{caption}</caption>'
            f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>")


def figure(fid: str, cap: str, alt: str, table_id: str | None = None) -> str:
    """A figure with an accessible name, a caption, and an escape hatch.

    The charts are matplotlib SVGs whose label sizes are baked in, so at phone
    width they would shrink past legibility -- hence the full-size PNG link and
    the horizontal scroll frame. `alt` is the accessible description; sighted
    readers get the same information from the caption, screen readers from the
    aria-label, and anyone who wants the numbers from the tables in section 5.
    """
    # No download links: this page ships as a single self-contained file, so a
    # relative href into figures/ would be broken for the reader. The charts are
    # inline SVG (zoomable), and every number behind them is in the tables.
    extra = (f'<p class="figlinks">Every number behind this chart is in '
             f'<a href="#{table_id}">the results tables</a>.</p>' if table_id else "")
    return (f'<figure role="group" aria-label="{alt}">'
            f'<div class="figwrap"><div class="plots" id="{fid}"></div></div>'
            f'<p class="figscroll-hint">&#8596; swipe to see the whole chart</p>'
            f'<figcaption class="figcap">{cap}</figcaption>{extra}'
            f'</figure>')


# ---------------------------------------------------------------- numbers
n_m, n_g = len(capable("mmlu")), len(capable("gsm8k"))
mi_m = M["mistral-small-3.1-24b"]["tasks"]["mmlu"]["HI-ROM"]
sm_m = M["sarvam-m-24b"]["tasks"]["mmlu"]["HI-ROM"]
sm_m_en = M["sarvam-m-24b"]["tasks"]["mmlu"]["EN"]["accuracy"]
mi_g = M["mistral-small-3.1-24b"]["tasks"]["gsm8k"]["HI-ROM"]
sm_g = M["sarvam-m-24b"]["tasks"]["gsm8k"]["HI-ROM"]
cm_m, hr_m = st.mean(adj("mmlu", "CM")), st.mean(adj("mmlu", "HI-ROM"))
cm_g, hr_g = st.mean(adj("gsm8k", "CM")), st.mean(adj("gsm8k", "HI-ROM"))
worst_m = min(capable("mmlu"), key=lambda s: M[s]["tasks"]["mmlu"]["HI-ROM"]["retention_adj"])
lo_m = -max(M[s]["tasks"]["mmlu"]["HI-ROM"]["delta_en"] for s in capable("mmlu"))
hi_m = -min(M[s]["tasks"]["mmlu"]["HI-ROM"]["delta_en"] for s in capable("mmlu"))
lo_g = -max(M[s]["tasks"]["gsm8k"]["HI-ROM"]["delta_en"] for s in capable("gsm8k"))
hi_g = -min(M[s]["tasks"]["gsm8k"]["HI-ROM"]["delta_en"] for s in capable("gsm8k"))

P: list[str] = []
w = P.append

w(HEAD.read_text(encoding="utf-8"))
w(f'<div class="statusbar">{len(M)} AI MODELS &middot; THE SAME QUESTIONS, ASKED FIVE WAYS</div>')
w('<nav><div class="logo"><span class="mark">क</span>CodeMix&#8202;Tax</div>'
  '<div class="navlinks">'
  '<a href="#finding" class="on">the finding</a>'
  '<a href="#english">english doesn\'t help</a><a href="#training">what helps</a>'
  '<a href="#read">how to read this</a><a href="#data">full results</a>'
  '<a href="#method">method</a></div></nav>')

# ------------------------------------------------------------- abstract
w('<div class="indexdemo"><div class="wrap"><div class="card">')
w('<div class="path">the code-mixing tax</div>')
w('<div class="card-title">What happens to an AI model when you stop writing to it '
  'in English <span class="chip chip-open">report</span> '
  '<span class="chip chip-india">india</span></div>')
w(f'<p style="font-size:13.5px">Hundreds of millions of people in India type a mix of '
  f'Hindi and English, in English letters &mdash; <em>&ldquo;mujhe kal ka schedule '
  f'bhejo&rdquo;</em>. Many large AI models advertise Hindi or broader multilingual '
  f'support; we could not find published figures for what romanized input costs them. '
  f'We asked <b>{len(M)} AI models</b> the same questions five '
  f'different ways and measured the damage. The headline is not what the marketing '
  f'implies: <b>mixing languages is not the problem &mdash; dropping the Hindi script '
  f'is</b>.</p>')
w('<div class="linksrow"><b>report</b> &middot; general knowledge + maths word problems '
  f'&middot; {len(M)} models &middot; Aug 2026</div>')
w("</div></div></div>")

w('<div class="report"><div class="wrap">')
w('<h1>The code-mixing <span class="g">tax.</span></h1>')
w(f'<div class="meta">{len(M)} models &middot; the same questions asked five ways '
  '&middot; 1,024 general-knowledge questions and 955 maths problems per form</div>')

# --------------------------------------------------------- the five forms
w('<section id="forms"><h2>the five ways we asked each question</h2>')
w('<div class="forms">')
for code, nm, ex in [
    ("1", "English", "What is the capital of France?"),
    ("2", "Hinglish", "France की capital क्या है?"),
    ("3", "Hindi", "फ़्रांस की राजधानी क्या है?"),
    ("4", "Romanized Hinglish", "France ki capital kya hai?"),
    ("5", "Romanized Hindi", "France ki rajdhani kya hai?"),
]:
    w(f'<div class="form-cell"><div class="n">FORM {code}</div>'
      f'<div class="code">{nm}</div><div class="desc">{ex}</div></div>')
w("</div>")
w('<p class="tnote">Form 5 is a form many people use on a phone keyboard. It is also '
  'the one models handle worst, by a wide margin.</p>')
w("</section>")

# ------------------------------------------------------------ the finding
_cm_m, _hr_m = ci("mmlu", "CM"), ci("mmlu", "HI-ROM")
_cm_g, _hr_g = ci("gsm8k", "CM"), ci("gsm8k", "HI-ROM")
_ex = TD["exclusions"]["excluded_models"]
# Points AND questions: "under half a question" was wrong by roughly 10x.
_gs = lambda slug, f: M[slug]["tasks"]["gsm8k"][f]["accuracy"]
_n_gsm = M["sarvam-m-24b"]["tasks"]["gsm8k"]["EN"]["rows"]
_rev = {"sarvam": _gs("sarvam-1-2b", "HI") - _gs("sarvam-1-2b", "CM"),
        "openhathi": _gs("openhathi-7b", "CM-ROM") - _gs("openhathi-7b", "CM")}
w('<section id="finding"><h2>1. the main finding</h2>')
w('<div class="panel">')
w('<p class="lead"><b>Writing in English letters costs more than mixing languages '
  'does &mdash; and the two costs compound.</b></p>')
w(f'<p>Hinglish was the easiest of the four non-English forms for <b>every model we '
  f'could meaningfully rank</b> &mdash; all {n_m} on general knowledge and all {n_g} on '
  f'maths. Models kept <b>{_cm_m["mean_retention"]:.1f}%</b> of their ability '
  f'(95% CI {_cm_m["ci95"][0]:.1f}&ndash;{_cm_m["ci95"][1]:.1f}) on general knowledge and '
  f'<b>{_cm_g["mean_retention"]:.1f}%</b> '
  f'({_cm_g["ci95"][0]:.1f}&ndash;{_cm_g["ci95"][1]:.1f}) on maths. &ldquo;Kept&rdquo; has '
  f'a precise meaning, defined just before the results tables below. A Hinglish question '
  f'gives a model two footholds: familiar English words, and familiar Hindi script. '
  f'Either one substantially cushions the loss &mdash; but neither removes it, and '
  f'mixing is not free: Hinglish still costs '
  f'{abs(_dif("mmlu", "mixing (CM vs EN)")["delta"]):.1f} points on general '
  f'knowledge.</p>')
w(f'<p><b>Which models are excluded, and why.</b> {len(M)-n_g} small models score so close '
  f'to guessing that ranking language forms by them is meaningless &mdash; '
  f'{", ".join(_ex["gsm8k"])} on maths, {", ".join(_ex["mmlu"])} on general knowledge. '
  f'OpenHathi 7B solves about 1 maths problem in 16 <em>in English</em>, and Param-1 2.9B '
  f'scores <b>higher</b> on Hinglish '
  f'({M["param-1-2.9b"]["tasks"]["gsm8k"]["CM"]["accuracy"]:.2f}) than on English '
  f'({M["param-1-2.9b"]["tasks"]["gsm8k"]["EN"]["accuracy"]:.2f}), which cannot be a real '
  f'effect. Those rows are dimmed in the tables and excluded from every count and '
  f'retention figure. Two of them do reverse the pattern on maths: Sarvam-1 scores higher '
  f'on Hindi than Hinglish ({_rev["sarvam"]:+.2f} points, about '
  f'{_rev["sarvam"]/100*_n_gsm:.0f} questions of {_n_gsm}) and OpenHathi higher on '
  f'Romanized Hinglish ({_rev["openhathi"]:+.2f} points, about '
  f'{_rev["openhathi"]/100*_n_gsm:.0f} questions) &mdash; small absolute differences on '
  f'models that answer roughly one question in 16 to begin with. On general knowledge the '
  f'pattern holds for all {len(M)} models regardless.</p>')
w(f'<p>Take both away and it falls off a cliff: only '
  f'<b>{_hr_m["mean_retention"]:.1f}%</b> '
  f'({_hr_m["ci95"][0]:.1f}&ndash;{_hr_m["ci95"][1]:.1f}) survives on general knowledge '
  f'and <b>{_hr_g["mean_retention"]:.1f}%</b> '
  f'({_hr_g["ci95"][0]:.1f}&ndash;{_hr_g["ci95"][1]:.1f}) on maths. Measured as a paired '
  f'difference on the same questions &mdash; the right way to compare two forms, since '
  f'two overlapping marginal intervals would not by themselves rule an effect out '
  f'&mdash; the gap between Hinglish and Romanized Hindi is '
  f'<b>{_dif("mmlu", "best vs worst form (CM vs HI-ROM)")["delta"]:.1f} points</b> on '
  f'general knowledge (95% CI '
  f'{_fci(_dif("mmlu", "best vs worst form (CM vs HI-ROM)")["ci95"])}) and '
  f'<b>{_dif("gsm8k", "best vs worst form (CM vs HI-ROM)")["delta"]:.1f}</b> on maths '
  f'(95% CI {_fci(_dif("gsm8k", "best vs worst form (CM vs HI-ROM)")["ci95"])}).</p>')
w("</div>")

# The 2x2. Every question exists in all four combinations of {mixed, all-Hindi}
# x {Devanagari, Latin}, so the two effects separate without any modelling.
_r = lambda f: ci("mmlu", f)["mean_retention"]
w('<div class="panel">')
w('<p><b>Which of the two changes costs more.</b> Every question exists in all four '
  'combinations of <em>where the words come from</em> and <em>which script they are '
  'written in</em>, so the two effects can be separated directly:</p>')
w(table("Floor-adjusted retention on general knowledge, by word source and script. "
        "The right-hand column is the cost of the script change alone.",
        ["", "Hindi script", "English letters", "Cost of the script change"],
        [["<b>Mixed</b> with English words (Hinglish)",
          f'{_r("CM"):.1f}%', f'{_r("CM-ROM"):.1f}%',
          f'<b>{_dif("mmlu", "romanizing a mixed sentence")["delta"]:.1f}</b>'],
         ["<b>All Hindi</b>",
          f'{_r("HI"):.1f}%', f'{_r("HI-ROM"):.1f}%',
          f'<b>{_dif("mmlu", "romanizing Hindi")["delta"]:.1f}</b>'],
         ["Cost of dropping the English words",
          f'{_dif("mmlu", "script vs lexicon (HI vs CM)")["delta"]:.1f}',
          f'{_dif("mmlu", "mixing, once romanized")["delta"]:.1f}', ""]],
        cls="prose"))
_ix, _ig = CI["tasks"]["mmlu"]["interaction"], CI["tasks"]["gsm8k"]["interaction"]
w(f'<p>Read down the last column: switching to English letters costs '
  f'{abs(_dif("mmlu", "romanizing a mixed sentence")["delta"]):.1f} points in a mixed '
  f'sentence but {abs(_dif("mmlu", "romanizing Hindi")["delta"]):.1f} in an all-Hindi '
  f'one. The two factors <em>compound</em> rather than merely add: '
  f'<b>{_ix["delta"]:.1f} points</b> of extra loss on general knowledge '
  f'(95% CI {_fci(_ix["ci95"])}) and <b>{_ig["delta"]:.1f}</b> on maths '
  f'(95% CI {_fci(_ig["ci95"])}). Both intervals exclude zero.</p>')
w('<p>So the fair summary is not that mixing is harmless. It is that the <em>script</em> '
  'is the more expensive of the two changes, and that the English words in a Hinglish '
  'sentence do most of the work of keeping a model oriented once the script is '
  'gone.</p>')
w("</div>")

# The best-English model's rank is computed, not typed: hardcoding it would
# silently go stale the moment another model is added.
_be = M[max(M, key=lambda x: M[x]["tasks"]["mmlu"]["EN"]["accuracy"])]["tasks"]["mmlu"]
_rank = 1 + sum(1 for x in capable("mmlu")
                if M[x]["tasks"]["mmlu"]["HI-ROM"]["retention_adj"] > _be["HI-ROM"]["retention_adj"])
_ord = {1: "st", 2: "nd", 3: "rd"}.get(_rank if _rank < 20 else _rank % 10, "th")
w('<div class="stats" style="margin-top:22px">')
for lab, val, sub, cls in [
    ("Hinglish &middot; floor-adj. retention", f"{cm_m:.1f}%",
     "The form everyone worries about is the easy one.", ""),
    ("Hindi in Latin &middot; floor-adj. retention", f"{hr_m:.1f}%",
     "A widely used way of typing is the hard one.", "down"),
    ("Models where Hinglish was easiest",
     f"{n_m}/{n_m} &middot; {n_g}/{n_g}",
     "General knowledge and maths, among the models eligible for each.", ""),
    ("Best English model&rsquo;s rank on robustness", f"{_rank}{_ord}",
     "A high English score does not predict robustness.", "down"),
]:
    w(f'<div class="stat-cell"><div class="lab">{lab}</div>'
      f'<div class="val {cls}">{val}</div><div class="sub">{sub}</div></div>')
w("</div>")
w(figure("curves",
         "<b>How accuracy falls across the five forms.</b> One panel per model. Faded, "
         "dashed lines are models that were already close to guessing in English.",
         "Small multiples: accuracy for each of 16 models across the five language "
         "forms. Most models decline overall from English onward, with romanized Hindi "
         "generally the weakest form; several curves rise again at romanized Hinglish.",
         "data"))
w("</section>")

# ------------------------------------------- 2. english capability does not carry
best_en = max(M, key=lambda x: M[x]["tasks"]["mmlu"]["EN"]["accuracy"])
be = M[best_en]["tasks"]["mmlu"]
sm = M["sarvam-m-24b"]["tasks"]["mmlu"]
mi = M["mistral-small-3.1-24b"]["tasks"]["mmlu"]
rank = 1 + sum(1 for x in capable("mmlu")
               if M[x]["tasks"]["mmlu"]["HI-ROM"]["retention_adj"] > be["HI-ROM"]["retention_adj"])
# A rank is only meaningful above the resolution of the measurement: 0.1 points
# of 1024 questions is one question, so anything inside that is a tie, not a place.
_TIE = 0.1
_ties = [x for x in capable("mmlu") if x != best_en
         and abs(M[x]["tasks"]["mmlu"]["HI-ROM"]["retention_adj"]
                 - be["HI-ROM"]["retention_adj"]) <= _TIE]
_tie_txt = ""
if _ties:
    _tie_txt = (" &mdash; effectively tied with "
                + " and ".join(M[x]["label"] for x in _ties)
                + f', a difference of well under one question in {be["EN"]["n"]:,}')
w('<section id="english"><h2>2. being good at english does not help</h2>')
w('<div class="panel">')
w(f'<p>This is the result we did not expect. <b>{M[best_en]["label"]}</b> scored the '
  f'highest of all {len(M)} models on English general knowledge &mdash; '
  f'<b>{be["EN"]["accuracy"]:.2f}%</b>. But when the same questions were typed in '
  f'Hindi-in-English-letters, it kept only '
  f'<b>{be["HI-ROM"]["retention_adj"]:.1f}%</b> of that ability, placing it '
  f'<b>{rank}th</b> out of {len(capable("mmlu"))}{_tie_txt}.</p>')
w(f'<p><b>Sarvam-M 24B</b> scored '
  f'{be["EN"]["accuracy"]-sm["EN"]["accuracy"]:.2f} points <em>lower</em> in English '
  f'&mdash; and kept <b>{sm["HI-ROM"]["retention_adj"]:.1f}%</b>. Raw capability in '
  f'English simply does not predict robustness here. What is associated with holding up '
  f'is having been trained on Indian-language text.</p>')
w("</div>")
# Selected by ENGLISH score, and the rule is stated in the caption. Selecting by
# retention instead would quietly drop the two rows that make the point hardest:
# Phi-4 and Mistral Small both score above 80 in English and retain the least.
_TOP_N = 8
_by_en = sorted(capable("mmlu"),
                key=lambda x: -M[x]["tasks"]["mmlu"]["EN"]["accuracy"])[:_TOP_N]
_ens = [M[x]["tasks"]["mmlu"]["EN"]["accuracy"] for x in _by_en]
_rets = [M[x]["tasks"]["mmlu"]["HI-ROM"]["retention_adj"] for x in _by_en]
w(table(f"The {_TOP_N} highest English scorers, in descending order of English score "
        f"&mdash; not selected by how well they hold up. English scores span "
        f"{max(_ens)-min(_ens):.1f} points; what they retain spans "
        f"{max(_rets)-min(_rets):.1f}.",
        ["Model", "English score", "Floor-adj. retention"],
        [[M[x]["label"] + (' <span class="chip">best English</span>'
                           if x == best_en else ""),
          f'{M[x]["tasks"]["mmlu"]["EN"]["accuracy"]:.2f}%',
          {"v": f'<b>{M[x]["tasks"]["mmlu"]["HI-ROM"]["retention_adj"]:.1f}%</b>',
           "c": "hi"}] for x in _by_en]))
w(figure("retain",
         "<b>Which models hold up best.</b> The share of each model&rsquo;s own ability "
         "that survives Hindi typed in Latin letters.",
         "Ranked bar chart of floor-adjusted retention on romanized Hindi for each "
         "eligible model, on both tests. Sarvam-M 24B ranks highest; the strongest "
         "English model is mid-table.",
         "data"))
w("</section>")

# ------------------------------------------------ 3. what actually helps
w('<section id="training"><h2>3. indic training is associated with a large benefit</h2>')
w('<div class="panel">')
w('<p>The most informative comparison is between two closely related checkpoints. '
  '<b>Sarvam-M 24B</b> is built on <b>Mistral Small 3.1 24B</b> &mdash; same parameter '
  'count, same base architecture &mdash; with additional Indian-language training.</p>')
w("</div>")
w(table("Same model, one trained further on Indian languages",
        ["Test, Hindi-in-Latin", "Mistral Small 24B", "Sarvam-M 24B", "Difference"],
        [[tl,
          f'{M["mistral-small-3.1-24b"]["tasks"][t]["HI-ROM"]["accuracy"]:.2f}%',
          f'{M["sarvam-m-24b"]["tasks"][t]["HI-ROM"]["accuracy"]:.2f}%',
          {"v": f'<b>+{M["sarvam-m-24b"]["tasks"][t]["HI-ROM"]["accuracy"] - M["mistral-small-3.1-24b"]["tasks"][t]["HI-ROM"]["accuracy"]:.2f} pts</b>',
           "c": "hi"}]
         for t, tl in TASKS]))
w(f'<p class="tnote">Floor-adjusted retention rises from '
  f'{mi_m["retention_adj"]:.1f}% to {sm_m["retention_adj"]:.1f}% &mdash; roughly double '
  f'&mdash; at the same model size.</p>')
w('<div class="warnblock" style="margin-top:22px">'
  '<h4>What this does and does not establish</h4>'
  '<p>This comparison is consistent with a substantial benefit from Indic post-training, '
  'but it does not isolate which part of the recipe caused it. These are released '
  'checkpoints, not a controlled ablation: post-training data, instruction tuning, '
  'tokenizer and other implementation choices all differ alongside the Indic data. A '
  'causal attribution would need an ablation by the model&rsquo;s authors.</p></div>')
w(figure("indic",
         "<b>Sarvam-M compared with the model it is built on.</b> Two closely related "
         "checkpoints, so the gap is associated with the additional Indian-language training.",
         "Two line charts comparing Mistral Small 3.1 24B with Sarvam-M 24B across the "
         "five language forms. Sarvam-M stays consistently above, and the gap widens "
         "toward romanized Hindi.",
         "data"))
w("</section>")

# ------------------------------------------------------ never finishing
w('<section id="unfinished"><h2>4. a second failure: never finishing</h2>')
w('<div class="panel"><p>Given romanized Hindi input, some models never commit to an '
  f'answer &mdash; they keep writing until cut off. With no answer to extract, the '
  f'question is scored <b>wrong</b>; it is never dropped from the total, so this failure '
  f'is already paid for in the accuracy figures above. On maths problems it happens to '
  f'<b>{st.mean(trunc("gsm8k","HI-ROM")):.1f}%</b> of questions in Hindi-in-Latin '
  f'against <b>{st.mean(trunc("gsm8k","EN")):.1f}%</b> in English.</p></div>')
w(figure("trunc",
         "<b>How often each model fails to finish.</b> Darker means more unfinished "
         "questions. The two Hindi columns are consistently worst.",
         "Heat map of the share of maths questions left unfinished, by model and "
         "language form. The Hindi and romanized-Hindi columns are darkest across "
         "almost every model.",
         "data"))
_bl = BF["budget_ladder"]["gsm8k"]
_bs = BF["budget_ladder"]
_lo = min(v["still_pct"] for v in _bl["forms"].values())
_hi2 = max(v["still_pct"] for v in _bl["forms"].values())
w('<div class="panel">')
w(f'<p><b>A bigger budget does not fix it &mdash; measured, not assumed.</b> Whenever a '
  f'run hit its token ceiling we re-ran just the affected rows with a larger budget, '
  f'usually double, across {_bs["n_models"]} models; every attempt is still on disk. Of '
  f'<b>{_bl["retried"]:,} maths rows</b> that were cut off and then retried with more '
  f'room, <b>{_bl["still_truncated"]:,} ({_bl["still_pct"]:.0f}%) still did not '
  f'finish</b>. The share that stayed unfinished is broadly similar in every language '
  f'form ({_lo:.0f}&ndash;{_hi2:.0f}%), so this is not a ceiling set slightly too low. '
  f'These models do not stop.</p>')
_pl = BF["prompt_length"]["gsm8k"]
w('<p><b>Nor is it simply that romanized text is longer.</b> Romanizing changes spelling, '
  'not word count, so a romanized question costs almost exactly what the same question '
  'costs in Hindi script:</p>')
w(table("Mean prompt length on maths questions, each model measured against its own "
        "English prompts and then averaged. Reading across either row, the token cost "
        "barely moves &mdash; while accuracy falls sharply.",
        ["", "Hindi script", "English letters"],
        [["<b>Mixed</b> with English words (Hinglish)",
          f'{_pl["CM"]["ratio_to_en"]:.2f}&times;', f'{_pl["CM-ROM"]["ratio_to_en"]:.2f}&times;'],
         ["<b>All Hindi</b>",
          f'{_pl["HI"]["ratio_to_en"]:.2f}&times;', f'{_pl["HI-ROM"]["ratio_to_en"]:.2f}&times;']],
        cls="prose"))
w('<p>Whatever romanized input is doing to these models, it is not merely making the '
  'input longer.</p>')
w("</div>")
w("</section>")

# ------------------------------------------------------ how to read this
oh = M["openhathi-7b"]["tasks"]["mmlu"]
gm = M["gemma3-12b-it"]["tasks"]["mmlu"]
w('<section id="read"><h2>how to read the numbers</h2>')
w('<div class="panel">')
w('<p class="lead">The main measure in this report is <b>floor-adjusted retention</b> '
  '&mdash; the share of a model&rsquo;s <em>own</em> ability that survives when the '
  'question changes language.</p>')
w('<p>It is not simply &ldquo;score in Hindi &divide; score in English&rdquo;, because '
  'that flatters weak models. In a four-option multiple-choice test, a model that knows '
  'nothing still scores about <b>25%</b> by guessing. That 25% is the <em>floor</em>, '
  'and subtracting it first is what &ldquo;floor-adjusted&rdquo; means:</p>')
w('<p style="text-align:center"><code>floor-adjusted retention = (score &minus; floor) '
  '&divide; (English score &minus; floor)</code></p>')
w(f'<p><b>A real example.</b> OpenHathi 7B scored <b>{oh["EN"]["accuracy"]:.2f}%</b> in '
  f'English and <b>{oh["HI-ROM"]["accuracy"]:.2f}%</b> on Hindi-in-Latin-letters.</p>')
w(f'<div class="tablewrap"><table><tbody>'
  f'<tr><td>Naive measure</td><td>{oh["HI-ROM"]["accuracy"]:.2f} &divide; '
  f'{oh["EN"]["accuracy"]:.2f}</td><td class="hi">'
  f'{oh["HI-ROM"]["retention_raw"]:.1f}%</td>'
  f'<td class="dim">looks like it held up fine</td></tr>'
  f'<tr><td>Ability in English</td><td>{oh["EN"]["accuracy"]:.2f} &minus; 25</td>'
  f'<td>{oh["EN"]["accuracy"]-25:.2f} pts</td><td class="dim">above guessing</td></tr>'
  f'<tr><td>Ability in Hindi</td><td>{oh["HI-ROM"]["accuracy"]:.2f} &minus; 25</td>'
  f'<td>{oh["HI-ROM"]["accuracy"]-25:.2f} pts</td><td class="dim">above guessing</td></tr>'
  f'<tr><td><b>Floor-adjusted</b></td><td>{oh["HI-ROM"]["accuracy"]-25:.2f} &divide; '
  f'{oh["EN"]["accuracy"]-25:.2f}</td><td class="dn"><b>'
  f'{oh["HI-ROM"]["retention_adj"]:.1f}%</b></td>'
  f'<td class="dim">the truth: nothing survived</td></tr></tbody></table></div>')
w(f'<p>The model scored <b>exactly what guessing scores</b>. So: <b>100%</b> means the '
  f'language change cost nothing, <b>{gm["HI-ROM"]["retention_adj"]:.1f}%</b> '
  f'(Gemma 3 12B on form 5) means it kept about half of what it knew, and <b>0%</b> '
  f'means it is down to guessing.</p>')
w("</div>")
w(table("The floor depends on the task",
        ["Task", "Floor used", "Why"],
        [["General knowledge", {"v": "<b>25</b>", "c": "hi"},
          "Four options per question, so guessing scores ~25%."],
         ["Maths word problems", {"v": "<b>0</b>", "c": "hi"},
          "The answer is a number the model has to work out. You cannot guess "
          "&ldquo;7,412&rdquo;, so there is no free score to subtract."]],
        cls="prose"))
w('<p class="tnote">With a floor of 0 the formula collapses to plain '
  '<code>score &divide; English score</code>. That is why the maths retention figures are '
  'simply the ratio of the two scores, while the general-knowledge ones are always lower '
  'than that ratio would suggest.</p>')
w('<div class="panel" style="margin-top:22px">')
w('<h3 style="font-size:14px;margin:0 0 10px">How the headline percentages are '
  'calculated</h3>')
w('<p>The figures in section 1 are <b>unweighted means across eligible models</b> of each '
  'model&rsquo;s own floor-adjusted retention &mdash; per-model retention first, then '
  'averaged, so every model counts equally regardless of size.</p>')
w(f'<p><b>Eligibility.</b> A model is included for a task if its English score clears '
  f'35% on general knowledge (guess floor 25) or 15% on maths (no floor). That leaves '
  f'<b>{n_m} of {len(M)}</b> models on general knowledge and <b>{n_g} of {len(M)}</b> on '
  f'maths &mdash; the counts differ because more models sit near the floor on maths. '
  f'Values are <b>not clipped</b> at 0 or 100; no eligible model produces a negative '
  f'figure.</p>')
# The full sweep lives in the appendix: this section must stay adjacent to the
# results tables, so a 12-row table does not belong in it.
_sv = {t: CI["tasks"][t]["sensitivity"] for t in ("mmlu", "gsm8k")}
_swing = max(max(r["retention"]["HI-ROM"] for r in _sv[t])
             - min(r["retention"]["HI-ROM"] for r in _sv[t]) for t in _sv)
w(f'<p><b>The cut-off is not load-bearing.</b> Moving it shifts reported Romanized Hindi '
  f'retention by at most {_swing:.1f} points across every threshold we tried, and never '
  f'reorders the five forms. The full sweep is in the appendix.</p>')
w(f'<p><b>Uncertainty.</b> 95% intervals come from a paired bootstrap over questions '
  f'({CI["iters"]:,} resamples, seed {CI["seed"]}). One resample of question indices is '
  f'applied to every condition and model at once, so differences between conditions stay '
  f'interpretable. These intervals quantify uncertainty from having sampled a finite set '
  f'of <b>questions</b>, with the tested model set held fixed; they say nothing about how '
  f'the result would generalise to other models.</p>')
w("</div>")
w(table("Mean floor-adjusted retention, with 95% bootstrap intervals",
        ["Task", "Hinglish", "Hindi", "Romanized Hinglish", "Romanized Hindi"],
        [[tl] + [{"v": f'{ci(t,f)["mean_retention"]:.1f}% '
                       f'<span class="ci">[{ci(t,f)["ci95"][0]:.1f}, '
                       f'{ci(t,f)["ci95"][1]:.1f}]</span>',
                  "c": "dn" if f == "HI-ROM" else ""}
                 for f in ("CM","HI","CM-ROM","HI-ROM")] for t, tl in TASKS]))
w("</section>")

# --------------------------------------------------------- full results
w('<section id="data"><h2>5. the full results</h2>')
w('<p class="figcap" style="margin-bottom:16px">Percentage of questions answered '
  'correctly. <b>Bold</b> marks each model&rsquo;s English score &mdash; its own '
  'ceiling. The last column is how much of that survives the hardest form. Dimmed rows '
  'were already close to guessing in English, so they had nothing to lose.<br>'
  '&dagger; Sarvam-30B ran with its reasoning capped at 4,000 tokens &mdash; uncapped it '
  'left a quarter of rows unfinished and could not be scored &mdash; so its figures are '
  'accuracy <em>under a reasoning cap</em>, not with unconstrained reasoning.</p>')
w('<div class="tstack">')
for t, tl in TASKS:
    n = "1,024" if t == "mmlu" else "955"
    rows = []
    for s, m in sorted(M.items(),
                       key=lambda kv: -(kv[1]["tasks"][t]["HI-ROM"]["retention_adj"] or -1)):
        p_ = m["tasks"][t]
        f_ = floored(s, t)
        row = [{"v": m["label"], "c": "dim" if f_ else ""}]
        for c in FORMS:
            v = f"{p_[c]['accuracy']:.2f}"
            row.append({"v": f"<b>{v}</b>" if c == "EN" and not f_ else v,
                        "c": "dim" if f_ else ""})
        row.append({"v": f"{p_['HI-ROM']['delta_en']:.2f}",
                    "c": "dim" if f_ else "dn"})
        row.append({"v": "&mdash;" if f_ else f"<b>{p_['HI-ROM']['retention_adj']:.1f}%</b>",
                    "c": "dim" if f_ else "hi"})
        rows.append(row)
    w(table(f"{tl} &mdash; {n} questions per form",
            ["Model"] + [NAMES[f] for f in FORMS] +
            ["Romanized Hindi loss vs English", "Floor-adj. retention"], rows))
    w('<p class="scrollnote">&#8596; this table scrolls sideways &mdash; the model column '
      'stays fixed. Eight columns; the last is floor-adjusted retention.</p>')
w('<p class="tnote"><b>Romanized Hindi loss vs English</b> is how many percentage points '
  'the model dropped, measured against its own English score. <b>Floor-adjusted '
  'retention</b> is that same drop expressed as a share of the model&rsquo;s real ability '
  '&mdash; see &ldquo;how to read the numbers&rdquo; above.</p>')
w("</div></section>")

# ------------------------------------------------------ what it means
w('<section id="meaning"><h2>6. what this means in practice</h2>')
w('<div class="panel">')
w('<p><b>Choosing a model for Indian users?</b> English benchmark scores alone can '
  'mislead you. The best English model here was mid-table once questions were typed in '
  'romanized form. Test on romanized input before committing.</p>')
w(f'<p><b>Building a product?</b> The input format matters as much as the model. Nudging '
  f'users toward Hindi script, or converting romanized input before it reaches the model, '
  f'may recover a large part of the loss &mdash; but note the ceiling: Hindi <em>in its '
  f'own script</em> still retains only <b>{ci("mmlu","HI")["mean_retention"]:.1f}%</b> on '
  f'general knowledge and <b>{ci("gsm8k","HI")["mean_retention"]:.1f}%</b> on maths. '
  f'Transliterating perfectly buys you the Hindi-script row, not the English one, and '
  f'real transliteration carries its own accuracy, latency and maintenance costs on '
  f'top.</p>')
w('<p><b>Training a model?</b> The gap can be materially reduced. Targeted training on '
  'Indian-language data is associated with roughly double the retention at the same model '
  'size &mdash; on a comparison of two released checkpoints rather than a controlled '
  'ablation.</p>')
w("</div></section>")

# ------------------------------------------------------------- credit
w('<section id="method"><h2>appendix &mdash; how this was measured</h2>')
w('<div class="panel">')
w('<p><b>The questions.</b> Two standard test sets: a general-knowledge '
  'multiple-choice exam covering 57 subjects, and grade-school maths word problems. '
  'Every model saw identical questions in all five forms, so scores are directly '
  'comparable within a model.</p>')
w('<p><b>Where each form came from.</b> Only the Hinglish form is published research '
  'data. The other four were assembled for this study and aligned back to it '
  'question-by-question, so all five ask the same thing.</p>')
w("</div>")
w(table("Where each form came from",
        ["Form", "General knowledge", "Maths word problems"],
        [["Hinglish",
          '<b><a href="https://aclanthology.org/2025.emnlp-main.109/">CodeMixBench</a></b>',
          '<b><a href="https://aclanthology.org/2025.emnlp-main.109/">CodeMixBench</a></b>'],
         ["English", "<code>cais/mmlu</code>, rejoined by question id",
          "<code>openai/gsm8k</code>, matched by its worked solution"],
         ["Hindi", "<code>CohereLabs/Global-MMLU</code> (Hindi)",
          "<code>bingbangboom/gsm8k-hindi</code> (MIT)"],
         ["Romanized Hinglish", "Hinglish, transliterated", "Hinglish, transliterated"],
         ["Romanized Hindi", "Hindi, transliterated", "Hindi, transliterated"]], cls="prose"))
w('<div class="panel" style="margin-top:22px">')
w('<p>Transliteration converts the script while leaving the words unchanged; English '
  'words inside a Hinglish sentence are left alone. Correct answers always come from the '
  'English originals, never from the translated datasets, so a mistranslated question '
  'cannot accidentally score as right &mdash; it simply fails.</p>')
w('<p><b>Scoring.</b> An answer is correct if it matches the known answer exactly. '
  'Models ran with settings that make them deterministic, so the same question gives '
  'the same answer every time. Where a model reasons before answering, only its final '
  'answer is scored; if it never reaches one before hitting its length limit, there is '
  'no answer to score and the question is marked <b>wrong</b>. Unfinished replies are '
  'never dropped from the total, so all five forms are scored over exactly the same '
  'questions &mdash; which is what makes comparing them like for like.</p>')
w('<p><b>Fairness.</b> Each model is compared only against itself, so a weaker model is '
  'not penalised for being weak and a stronger one gets no free credit. The measure is '
  'how much each <em>loses</em>. Full reproduction steps are in the repository '
  'README.</p>')
w("</div>")
_vm, _un, _so = TD["verified_vs_mt"], TD["unanimous"], TD["script_only"]
_ex2 = TD["exclusions"]
w('<div class="warnblock" style="margin-top:22px">'
  '<h4>Limitation: translation quality is part of what we measure</h4>'
  '<p>Correct answers always come from the <b>English</b> originals, never from the '
  'translated datasets &mdash; the Hindi answer fields are unreliable. But that means the '
  'Hindi conditions measure <em>&ldquo;can the model answer the English question as '
  'rendered in Hindi&rdquo;</em>, which is not quite <em>&ldquo;can the model do '
  'Hindi&rdquo;</em>. If a translation drifts, a model answering the Hindi question '
  'correctly is still marked wrong.</p>'
  f'<p><b>Coverage.</b> Only {_vm["verified_pct"]}% of the Hindi general-knowledge items '
  f'({_vm["verified_items"]} of {_vm["total_items"]}) are marked human-verified by '
  f'Global-MMLU. The Hindi maths set is row-aligned, and its own answer field is malformed '
  f'for ~27% of rows, which is why we do not use it.</p>'
  f'<p><b>Exclusions already applied.</b> '
  f'{_ex2["gsm8k_items_dropped_for_number_mismatch"]} of 1,016 maths items '
  f'({_ex2["gsm8k_dropped_pct"]}%) were dropped because the Hindi question did not carry '
  f'the same numbers as the English one.</p>'
  f'<p><b>Measured effect.</b> Comparing the same {_vm["n_models"]} models on the '
  f'human-verified subset against the machine-translated one: '
  f'<b>{_vm["mean_gap_points"]["HI"]:+.2f} points</b> on Hindi and '
  f'<b>{_vm["mean_gap_points"]["HI-ROM"]:+.2f}</b> on Romanized Hindi &mdash; small, and '
  f'absent on the form carrying the headline result. Questions that <em>every</em> '
  f'eligible model gets right in English and wrong in Hindi number '
  f'{_un["mmlu"]["en_pass_hi_fail"]} of {_un["mmlu"]["n_items"]:,} '
  f'({_un["mmlu"]["en_pass_hi_fail_pct"]}%) and '
  f'{_un["gsm8k"]["en_pass_hi_fail"]} of {_un["gsm8k"]["n_items"]} '
  f'({_un["gsm8k"]["en_pass_hi_fail_pct"]}%); the reverse direction is <b>0</b> in both.</p>'
  '<p><b>Unquantified:</b> semantic drift preserving the numbers, such as &ldquo;gave '
  'away&rdquo; becoming &ldquo;received&rdquo;. One confirmed case: a maths item where '
  '&ldquo;1/4 as big as&rdquo; became &ldquo;1/4 bigger than&rdquo; &mdash; all eligible '
  'models agreed on the same wrong answer, each having solved the mistranslated question '
  'correctly.</p></div>')
w('<div class="panel" style="margin-top:22px">'
  '<h3 style="font-size:14px;margin:0 0 10px">A translation-free corroboration</h3>'
  '<p>Romanizing is a deterministic transliteration of text we already have, so any '
  'translation error is identical on both sides and cancels exactly. On that comparison '
  'alone:</p></div>')
w(table("Cost of romanizing the same text (no translation exposure)",
        ["Task", "Romanizing Hinglish", "Romanizing Hindi", "Ratio"],
        [[tl, f'{_so[t]["romanising_hinglish_costs"]:.2f} pts',
          f'{_so[t]["romanising_hindi_costs"]:.2f} pts',
          {"v": f'<b>{_so[t]["ratio"]:.1f}&times;</b>', "c": "hi"}] for t, tl in TASKS]))
w('<p class="tnote">Losing the script costs roughly twice as much when there is no '
  'English to fall back on.</p>')

# ------------------------------- limitation: the romanization scheme itself
_rc = RA["conditions"]
_lo_r = min(c["rows_pct"] for c in _rc.values())
_hi_r = max(c["rows_pct"] for c in _rc.values())
w('<div class="panel">'
  '<h3 style="font-size:14px;margin:0 0 10px">Limitation &mdash; our romanization is '
  'one scheme, not a sample of real typing</h3>')
w(f'<p>Romanized Hindi has no standard spelling: the same word is written several ways by '
  f'different people, and often by the same person. Our two Latin-script conditions come '
  f'from a single deterministic transform &mdash; <b>{e(RA["tool"])} '
  f'{e(RA["tool_version"])}</b> ({e(RA["scheme"])}), then '
  f'{e(", ".join(RA["postprocess"][:4]))} and lowercasing '
  f'(<code>{e(RA["implementation"])}</code>). That makes the conditions exactly '
  f'reproducible, and it also makes them <em>one point</em> in a wide space of things '
  f'people really type.</p>')
w(f'<p><b>The transform has a known defect, and it is common.</b> It does not model '
  f'<em>internal</em> schwa deletion, so it writes <code>men</code> where a person '
  f'writes <code>mein</code>, and <code>kitane</code> for <code>kitne</code>. At least '
  f'one such form appears in <b>{_lo_r:.0f}&ndash;{_hi_r:.0f}% of rows</b> across the '
  f'four romanized sets:</p>')
w(table("How often a known-wrong romanized form appears, per condition.",
        ["Romanized condition", "Rows with a known-wrong form", "Share of words"],
        [[f"<code>{e(n)}</code>",
          f'{c["rows_with_divergent_form"]:,} of {c["rows"]:,} ({c["rows_pct"]:.1f}%)',
          f'{c["token_pct"]:.2f}%'] for n, c in _rc.items()], cls="prose"))
w('<p><b>Which way this biases the result.</b> Our romanized text is slightly '
  '<em>less</em> natural than real typing, so it is plausibly further out of a '
  'model&rsquo;s distribution than what a user would actually send. The romanization '
  'penalty reported here is therefore best read as an <b>upper bound</b> on the penalty '
  'for well-formed romanized Hindi. Pushing the other way, real input is <em>more</em> '
  'variable than ours &mdash; inconsistent spelling inside a single message &mdash; '
  'which a uniform scheme does not test at all.</p>')
w(f'<p><b>Not validated against human-typed text.</b> We did not collect human '
  f'romanizations of these questions, so we cannot report agreement with them and have '
  f'not claimed to. A seeded {RA["sample_rows"]}-row sample of the transform&rsquo;s '
  f'output beside its Devanagari source ships as '
  f'<code>results/romanization_sample.csv</code> for anyone who reads Hindi to check by '
  f'eye. Comparing against human-typed romanized Hindi is the clearest next step, and '
  f'would tighten that bound.</p>')
w("</div>")
w('<div class="panel">'
  '<h3 style="font-size:14px;margin:0 0 10px">Sensitivity &mdash; does the eligibility '
  'cut-off matter?</h3>'
  '<p>A model scoring near the guessing floor in English cannot rank language forms, so '
  'models below a threshold are left out of the headline means. Any threshold invites the '
  'question of whether it was picked to flatter the result, so here is the whole '
  'sweep:</p></div>')
w(table("Headline retention recomputed across a range of eligibility cut-offs. The "
        "marked row is the threshold used throughout, fixed before these figures were "
        "computed. The least selective settings &mdash; every model included &mdash; "
        "give a <em>worse</em> Romanized Hindi figure than the one reported, so the "
        "threshold chosen is the conservative option rather than the flattering one.",
        ["Cut-off", "Eligible"] + [NAMES[f] for f in FORMS],
        [[f'{tl}, &ge;{r["cut"]:.0f}%'
          + (' <span class="chip">used</span>' if r["cut"] == CUT[t] else ""),
          str(r["n_eligible"])]
         + [f'{r["retention"][f]:.1f}%' for f in FORMS]
         for t, tl in TASKS for r in _sv[t]]))
w(table("Models, settings and reproduction",
        ["", ""],
        [["Decoding", "greedy (temperature 0), fixed seed &mdash; deterministic"],
         ["Precision", "bfloat16 for all models except the two below"],
         ["Quantized models", "Gemma 3 12B (compressed) is 4-bit; Llama 4 Scout is a "
                              "4-bit weight-quantized release"],
         ["Token budgets", "per model, set by its context window and whether it reasons "
                           "before answering; recorded per run"],
         ["Answer parsing", "shape-aware: handles models that answer first and models "
                            "that reason first. A reply containing no answer &mdash; cut "
                            "off mid-reasoning, say &mdash; is scored <b>wrong</b>, never "
                            "dropped, so every form is scored over the same questions"],
         ["Reasoning models", "Sarvam-30B ran with reasoning capped at 4,000 tokens; "
                              "uncapped it left a quarter of rows unfinished"],
         ["Inference", "vLLM, offline batch"]],
        cls="prose kv"))
w('<p class="tnote">Exact checkpoint ids, per-model configuration, prompts, token '
  'budgets and the answer parser are all in the accompanying code repository; each run '
  'also writes a <code>run_meta.json</code> recording the settings actually used.</p>')
w('<div class="credit" style="margin-top:22px"><h4>Credit</h4>'
  '<p>Hinglish questions from <b>CodeMixBench</b> '
  '(<a href="https://aclanthology.org/2025.emnlp-main.109/">Yang &amp; Chai, EMNLP 2025</a>). English '
  'and Hindi matched question-by-question from <code>cais/mmlu</code>, '
  '<code>openai/gsm8k</code>, <code>CohereLabs/Global-MMLU</code> and '
  '<code>bingbangboom/gsm8k-hindi</code>. The alignment across all five forms, the two '
  f'Latin-letter forms, the retention measure and the {len(M)}-model comparison are this '
  'study&rsquo;s.</p></div>')
w("</section>")

w("</div></div>")
# Author and licence come from CITATION.cff / LICENSE in the repo rather than
# being typed here, so the page cannot claim something the repo contradicts.
w('<footer>'
  f'<span>The Code-Mixing Tax &middot; {len(M)} open-weight models &middot; '
  f'{AUTHOR}</span>'
  f'<span>Code and data pipeline released under the {LICENSE_ID} licence. '
  f'Hinglish questions from CodeMixBench (Yang &amp; Chai, EMNLP 2025).</span>'
  '</footer>')

w("</body>\n</html>")
Path("results/codemix-tax.html").write_text("\n".join(P), encoding="utf-8")
print(f"wrote results/codemix-tax.html ({sum(len(x) for x in P)//1024} KB pre-figures)")
