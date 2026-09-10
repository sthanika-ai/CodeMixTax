#!/usr/bin/env python
"""Figures for the condensed Code-Mixing Tax report (MMLU + GSM8K only).

Four figures, each rendered light and dark (SVG for the page, PNG for slides):

  curves  small multiples, one panel per model: accuracy across the five forms
  retain  floor-adjusted retention at Romanized Hindi, ranked
  trunc   GSM8K truncation heat map, model x language form
  indic   the controlled pair - Mistral Small 24B vs its Indic fine-tune Sarvam-M

Palette is the validated default: categorical slots 1-2 (blue/orange) and the
blue sequential ramp. Every panel carries at most two series, so the all-pairs
CVD gate holds. Run `node dataviz/scripts/validate_palette.js` to re-check.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

DATA = Path("results/tax_dataset.json")
OUT = Path("results/figures")
FORMS = ["EN", "CM", "HI", "CM-ROM", "HI-ROM"]
FORM_LABEL = ["English", "Hinglish", "Native\nHindi", "Rom.\nHinglish", "Rom.\nHindi"]
TASKS = [("mmlu", "MMLU"), ("gsm8k", "GSM8K")]
# A curve is "floored" when its own English score leaves no headroom above the
# task's guess floor -- the drop it would show is noise, not degradation. This is
# per (model, task): Sarvam-1 and Param-1 are floored on GSM8K but not on MMLU.
FLOOR_CUT = {"mmlu": 35.0, "gsm8k": 15.0}


def floored(task: str, en: float) -> bool:
    return en < FLOOR_CUT[task]


#: models floored on at least one task -- excluded from the retention ranking
FLOORED = {"openhathi-7b", "param-1-2.9b", "sarvam-1-2b"}

THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8880",
                  grid="#e6e5e1", s1="#2a78d6", s2="#eb6834",
                  ramp=["#fcfcfb", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#8a8880",
                  grid="#333330", s1="#3987e5", s2="#d95926",
                  ramp=["#1a1a19", "#104281", "#184f95", "#256abf", "#3987e5", "#5598e7", "#86b6ef", "#cde2fb"]),
}


def style(t: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "axes.edgecolor": t["grid"], "grid.color": t["grid"],
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "medium",
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "path",
    })


def recede(ax, t: dict, *, left=True) -> None:
    ax.grid(axis="y", lw=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    if not left:
        ax.spines["left"].set_visible(False)
    ax.tick_params(length=0, pad=4)


def rbarh(ax, y: float, w: float, h: float, color: str, surface: str) -> None:
    """Horizontal bar with a rounded data-end and a square baseline end.

    matplotlib has no rounded bar, and FancyBboxPatch rounds all four corners,
    so the baseline corners are squared back off with a thin overlay.
    """
    r = min(h * 0.28, max(w, 1e-6) * 0.5)
    ax.add_patch(FancyBboxPatch(
        (0, y - h / 2), max(w - r, 1e-6), h, linewidth=0, facecolor=color, zorder=3,
        boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=1))
    ax.add_patch(Rectangle((0, y - h / 2), min(r, w), h, linewidth=0,
                           facecolor=color, zorder=3))


def save(fig, name: str, mode: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}-{mode}.{ext}", format=ext,
                    bbox_inches="tight", pad_inches=0.06,
                    dpi=200 if ext == "png" else None)
    plt.close(fig)


# ---------------------------------------------------------------- figure 1
def fig_curves(D: dict, mode: str) -> None:
    t = THEME[mode]; style(t)
    order = sorted(D["models"].items(),
                   key=lambda kv: -(kv[1]["tasks"]["mmlu"]["HI-ROM"]["retention_adj"] or -1))
    # 3 columns, not 4. The SVG scales to the page column, so panel size on
    # screen is set by the fraction of figure width each panel gets -- widening
    # the figure alone just scales everything back down. Three columns make each
    # panel a third of the width instead of a quarter, and the fonts below are
    # sized against the narrower grid.
    ncol = 3
    nrow = -(-len(order) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.6, 2.55 * nrow),
                             sharex=True, sharey=True)
    flat = axes.ravel()
    for ax, (slug, m) in zip(flat, order):
        recede(ax, t)
        for (task, tlabel), col in zip(TASKS, (t["s1"], t["s2"])):
            y = [m["tasks"][task][f]["accuracy"] for f in FORMS]
            flat_ = floored(task, y[0])
            ax.plot(range(5), y, lw=2.0 if not flat_ else 1.6, color=col,
                    marker="o", ms=4.5, mec=t["surface"], mew=1.2,
                    ls="-" if not flat_ else (0, (3, 2)),
                    alpha=1.0 if not flat_ else 0.55,
                    zorder=3, clip_on=False)
        ax.set_title(f"{m['label']}{'  •' if m['indic'] else ''}",
                     color=t["ink"], pad=7, loc="left", fontsize=12)
        ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=10)
        ax.axhline(25, color=t["ink3"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    for ax in flat[len(order):]:
        ax.set_visible(False)      # the legend host is re-enabled below
    # Any panel with no panel beneath it needs its own x labels.
    for i, ax in enumerate(flat[:len(order)]):
        if i + ncol >= len(order):
            ax.set_xticks(range(5))
            ax.set_xticklabels(FORM_LABEL, fontsize=9.5)
            ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy %", color=t["ink2"], fontsize=10.5)
    handles = [Line2D([], [], color=c, lw=2.4, marker="o", ms=5, label=l)
               for (_, l), c in zip(TASKS, (t["s1"], t["s2"]))]
    handles += [
        Line2D([], [], color=t["ink3"], lw=0.8, ls=(0, (4, 3)), label="MMLU guess floor (25%)"),
        Line2D([], [], color=t["ink3"], lw=1.6, ls=(0, (3, 2)), alpha=0.7,
               marker="o", ms=4.5, label="floored: no headroom, curve is noise"),
    ]
    # The grid leaves (nrow*ncol - len(order)) cells empty on the last row; the
    # legend goes in the first of them, so it never collides with the title.
    spare = nrow * ncol - len(order)
    if spare:
        host = flat[len(order)]
        host.set_visible(True)
        host.set_frame_on(False)
        # NOT set_xticks([]) -- these axes share their locators with every other
        # panel, so clearing ticks here would clear them across the whole grid.
        # Hide this axes' own decorations only.
        host.tick_params(left=False, bottom=False,
                         labelleft=False, labelbottom=False)
        host.grid(False)
        host.legend(handles=handles, loc="upper left", frameon=False,
                    fontsize=11, labelcolor=t["ink2"], handlelength=2.6,
                    labelspacing=0.9, borderpad=0.6)
    else:
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 1.004),
                   frameon=False, ncol=3, fontsize=10.5, labelcolor=t["ink2"])
    fig.text(0.02, 0.997, "Accuracy across the five language forms",
             fontsize=15, color=t["ink"], weight="medium", va="top")
    fig.text(0.02, 0.968,
             "One panel per model, shared axes.  • = Indic post-training",
             fontsize=10.5, color=t["ink2"], va="top")
    # Reserve the header in POINTS, not as a fraction: the fraction has to shrink
    # as the figure grows taller, and the panel titles (12pt + 7pt pad) reach back
    # up into whatever gap is left. 92pt covers title + subtitle + clear space.
    fig_h_pt = 2.55 * nrow * 72
    fig.subplots_adjust(top=1 - 92 / fig_h_pt, hspace=0.34, wspace=0.11)
    save(fig, "tax-curves", mode)


# ---------------------------------------------------------------- figure 2
def fig_retain(D: dict, mode: str) -> None:
    t = THEME[mode]; style(t)
    keep = [(s, m) for s, m in D["models"].items() if s not in FLOORED]
    keep.sort(key=lambda kv: kv[1]["tasks"]["mmlu"]["HI-ROM"]["retention_adj"])
    labels = [m["label"] + ("  •" if m["indic"] else "") for _, m in keep]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    recede(ax, t)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", lw=0.6, alpha=0.9)
    h = 0.34                      # leaves a visible surface gap between the pair
    for i, ((slug, m), lab) in enumerate(zip(keep, labels)):
        for k, ((task, tlabel), col) in enumerate(zip(TASKS, (t["s1"], t["s2"]))):
            v = m["tasks"][task]["HI-ROM"]["retention_adj"]
            y = i + (0.5 - k) * (h + 0.07)
            rbarh(ax, y, v, h, col, t["surface"])
            ax.text(v + 1.6, y, f"{v:.0f}", va="center",
                    ha="left", fontsize=7.8, color=t["ink2"], zorder=4)
    ax.set_yticks(range(len(keep))); ax.set_yticklabels(labels, fontsize=9, color=t["ink"])
    # add_patch does not drive autoscale, so both limits are set explicitly
    ax.set_ylim(-0.62, len(keep) - 0.38)
    ax.set_xlim(0, 100); ax.set_xlabel("floor-adjusted retention at Romanized Hindi (%)", color=t["ink2"])
    ax.spines["left"].set_visible(False)
    handles = [Patch(facecolor=c, label=l) for (_, l), c in zip(TASKS, (t["s1"], t["s2"]))]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, labelcolor=t["ink2"])
    ax.set_title("How much of each model's English headroom survives Romanized Hindi\n"
                 "• = Indic post-training.  Floored base models excluded.",
                 loc="left", color=t["ink"], pad=10, fontsize=11)
    save(fig, "tax-retain", mode)


# ---------------------------------------------------------------- figure 3
def fig_trunc(D: dict, mode: str) -> None:
    t = THEME[mode]; style(t)
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    order = sorted(D["models"].items(),
                   key=lambda kv: -max(kv[1]["tasks"]["gsm8k"][f]["trunc_pct"] for f in FORMS))
    M = [[m["tasks"]["gsm8k"][f]["trunc_pct"] for f in FORMS] for _, m in order]
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=42, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(FORM_LABEL, fontsize=8.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([m["label"] + ("  •" if m["indic"] else "") for _, m in order],
                       fontsize=9, color=t["ink"])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i, row in enumerate(M):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.1f}" if v >= 0.05 else "0",
                    ha="center", va="center", fontsize=8,
                    color=t["surface"] if v > 20 else t["ink2"], zorder=3)
    # 2px surface gap between cells
    ax.set_xticks([x - 0.5 for x in range(1, 5)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(order))], minor=True)
    ax.grid(which="minor", color=t["surface"], lw=2)
    ax.grid(which="major", visible=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("% of rows truncated", color=t["ink2"], fontsize=8.5)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, colors=t["ink2"], labelsize=8)
    ax.set_title("GSM8K truncation by language form\n"
                 "Native Hindi and Romanized Hindi fail to converge; code-mixing does not.",
                 loc="left", color=t["ink"], pad=10, fontsize=11)
    save(fig, "tax-trunc", mode)


# ---------------------------------------------------------------- figure 4
def fig_indic(D: dict, mode: str) -> None:
    t = THEME[mode]; style(t)
    pair = [("mistral-small-3.1-24b", t["ink3"], "Mistral Small 3.1 24B  (base)"),
            ("sarvam-m-24b", t["s1"], "Sarvam-M 24B  (Indic post-trained)")]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    for ax, (task, tlabel) in zip(axes, TASKS):
        recede(ax, t)
        for slug, col, lab in pair:
            y = [D["models"][slug]["tasks"][task][f]["accuracy"] for f in FORMS]
            ax.plot(range(5), y, lw=2.2, color=col, marker="o", ms=6.5,
                    mec=t["surface"], mew=1.4, zorder=3, clip_on=False)
            ax.annotate(f"{y[-1]:.1f}", (4, y[-1]), textcoords="offset points",
                        xytext=(7, 0), fontsize=8.5, color=t["ink2"], va="center")
        gap = (D["models"]["sarvam-m-24b"]["tasks"][task]["HI-ROM"]["accuracy"]
               - D["models"]["mistral-small-3.1-24b"]["tasks"][task]["HI-ROM"]["accuracy"])
        ax.set_title(f"{tlabel}   → +{gap:.1f} pts at Romanized Hindi",
                     loc="left", color=t["ink"], pad=8, fontsize=10)
        ax.set_xticks(range(5)); ax.set_xticklabels(FORM_LABEL, fontsize=8)
        ax.set_xlim(-0.25, 4.55)
        # One shared y-range across both panels: different ranges per panel would
        # make a 14-point gap look like a 22-point one.
        ax.set_ylim(38, 100)
    axes[0].set_ylabel("accuracy %", color=t["ink2"])
    axes[1].tick_params(labelleft=False)
    handles = [Line2D([], [], color=c, lw=2.4, marker="o", ms=6, label=l)
               for _, c, l in pair]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               frameon=False, ncol=2, fontsize=9.5, labelcolor=t["ink2"])
    fig.text(0.005, 1.03, "Indic post-training, measured on one architecture",
             fontsize=12.5, color=t["ink"], weight="medium", va="top")
    fig.text(0.005, 0.985,
             "Sarvam-M is a Mistral Small 3.1 24B fine-tune, so the gap isolates post-training from scale.",
             fontsize=9, color=t["ink2"], va="top")
    fig.subplots_adjust(top=0.80, wspace=0.10)
    save(fig, "tax-indic", mode)


def main() -> None:
    D = json.loads(DATA.read_text())
    for mode in ("light", "dark"):
        fig_curves(D, mode); fig_retain(D, mode)
        fig_trunc(D, mode);  fig_indic(D, mode)
    print("wrote:")
    for p in sorted(OUT.glob("tax-*")):
        print(f"  {p}  {p.stat().st_size//1024} KB")


if __name__ == "__main__":
    main()
