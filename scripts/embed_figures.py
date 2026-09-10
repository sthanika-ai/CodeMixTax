#!/usr/bin/env python
"""Inline the matplotlib SVGs into the HTML report, one light and one dark per figure.

Two things make this more than a paste:

* **id namespacing.** matplotlib emits ``<defs>`` entries (glyph outlines, marker
  shapes) and refers to them with ``<use xlink:href="#id">``. Ids are only unique
  within one file, so dropping fourteen figures into one document makes the first
  definition win and figures render each other's markers. Every id and every
  reference is rewritten with a per-figure prefix.
* **theme pairing.** A vector figure with baked-in colours cannot adapt to the
  viewer's theme, so both variants are embedded and CSS shows one. That has to
  cover all three viewer states: explicit dark, explicit light, and the unstamped
  default where only ``prefers-color-scheme`` is known.

    python scripts/embed_figures.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

#: figure id in the HTML -> figure basename in results/figures
FIGURES = {
    "indic":  "tax-indic",
    "retain": "tax-retain",
    "trunc":  "tax-trunc",
    "curves": "tax-curves",
}

_STRIP = (
    re.compile(r"<\?xml[^>]*\?>\s*"),
    re.compile(r"<!DOCTYPE[^>]*>\s*", re.I),
    re.compile(r"<metadata>.*?</metadata>\s*", re.S),
    re.compile(r"<!--.*?-->\s*", re.S),
)


def prepare(svg_text: str, prefix: str) -> str:
    """Strip the standalone-document wrapper and namespace every internal id."""
    for pat in _STRIP:
        svg_text = pat.sub("", svg_text)

    ids = set(re.findall(r'id="([^"]+)"', svg_text))
    # Longest first, so `m1a` is not rewritten inside `m1ab`.
    for i in sorted(ids, key=len, reverse=True):
        new = f"{prefix}-{i}"
        svg_text = svg_text.replace(f'id="{i}"', f'id="{new}"')
        svg_text = svg_text.replace(f'href="#{i}"', f'href="#{new}"')
        svg_text = svg_text.replace(f"url(#{i})", f"url(#{new})")

    # Let the figure scale to its column instead of asserting a pt size.
    svg_text = re.sub(r'(<svg[^>]*?)\swidth="[^"]*"\sheight="[^"]*"', r"\1", svg_text, count=1)
    svg_text = svg_text.replace("<svg ", '<svg preserveAspectRatio="xMidYMid meet" ', 1)
    return svg_text.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default="results/codemix-tax.html")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--out", default="results/codemix-tax.html")
    args = ap.parse_args()

    src = Path(args.html).read_text(encoding="utf-8")
    figdir = Path(args.figures)
    n = 0

    for fid, base in FIGURES.items():
        blocks = []
        for mode in ("light", "dark"):
            p = figdir / f"{base}-{mode}.svg"
            if not p.exists():
                raise FileNotFoundError(f"missing {p} -- run scripts/plot_degradation.py first")
            blocks.append(f'<div class="fig-{mode}">'
                          + prepare(p.read_text(encoding="utf-8"), f"{fid}{mode[0]}")
                          + "</div>")
        # A plain container, not a styled one: the page decides the figure frame
        # (border, padding, spacing) so the embedder stays layout-agnostic.
        replacement = '<div class="figpair">' + "".join(blocks) + "</div>"

        pat = re.compile(r'<div class="plots[^"]*" id="' + re.escape(fid) + r'"></div>')
        if not pat.search(src):
            raise SystemExit(f"no placeholder for figure {fid!r} in {args.html}")
        src = pat.sub(lambda _m: replacement, src, count=1)
        src = re.sub(r'\s*<div class="legend" id="lg-' + re.escape(fid) + r'"></div>', "", src)
        n += 1

    Path(args.out).write_text(src, encoding="utf-8")
    print(f"embedded {n} figures ({2*n} SVGs) into {args.out}")
    print(f"size: {len(src)/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
