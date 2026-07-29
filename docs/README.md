# TrafficRuleBench — rebuttal supplementary page

Static GitHub Pages site (no build step): `index.html` + `static/`.

Page structure: compact header (paper title, brief description, resource links)
with **three reviewer buttons** that jump to per-reviewer chapters:

| Anchor | Reviewer | Content |
|---|---|---|
| `#reviewer-k1vn` | K1Vn (Minor Weakness #3) | Qualitative demos: featured GIF, violation-vs-compliant pairs, 18-scenario gallery |
| `#reviewer-ybmx` | YBmX (Weakness #2) | Enlarged zoomable Fig. 1 & Fig. 2 with before/after teaser comparison |
| `#reviewer-pqwl` | pQWL (Question #5) | Twin-gap chart (RC vs RCR), correlation heatmap, bootstrap-CI forest plot |

General material (pipeline, benchmark comparison, citation) follows the chapters.

## Enable GitHub Pages

Repo **Settings → Pages → Build and deployment**:
- Source: *Deploy from a branch*
- Branch: `main`, folder: `/docs`

The site will be served at `https://emb-ai.github.io/traffic-rule-bench/`.

Local preview:

```bash
cd docs && python -m http.server 8000
# open http://localhost:8000
```

## Adding demo GIFs (Reviewer K1Vn chapter)

The page works without any GIFs: every media slot shows the rule schematic with a
"Demo GIF soon" badge. As soon as a GIF file appears at the expected path, the
slot automatically switches to the animation (no HTML edits needed).

| Slot | Path |
|---|---|
| Featured demo | `static/gifs/hero.gif` |
| Scenario cards | `static/gifs/<sign code>.gif`, e.g. `static/gifs/3.24.gif` |
| Comparison pairs — violation | `static/gifs/pairs/<code>_violation.gif` |
| Comparison pairs — compliant | `static/gifs/pairs/<code>_compliant.gif` |

Scenario card codes (see `SCENARIOS` in `static/js/main.js`):
`2.4`, `2.5`, `3.1`, `3.2`, `3.20`, `3.24`, `3.27`, `4.2.1`, `4.2.2`, `4.2.3`,
`4.6`, `5.11.1`, `5.11.2`, `5.14.1`, `5.14.2`, `5.15.2`, `5.19`, `5.31`.

Comparison pair codes (see `PAIRS` in `static/js/main.js`): `2.4`, `3.24`, `5.19`.

Tips:
- Keep GIFs under ~5 MB each; 480–640 px wide top-down renders at 10–15 fps look great.
- MP4/WebM are not probed by the current logic — convert to GIF, or ask to extend
  `initMediaSlot` in `static/js/main.js` with `<video>` support.

## Other fillable slots

- **Reviewer pQWL — twin-gap chart, correlation heatmap & bootstrap-CI forest**:
  regenerate with `python docs/_build_figures.py` (reads CSVs from
  `pdd-bench/.../reviewer_evidence/tables/`, writes dark-styled PNGs).
- **Reviewer YBmX — Fig. 2**: slider compares
  `fig2_nuplan_stats_orig.png` vs `fig2_nuplan_stats_new.png`
  (new PNG rendered from `nuplan_stats_new.pdf`). Regenerate with:
  `python -c "import fitz; ..."` or ask to re-run the conversion.

## Things to update before/after de-anonymization

- Author list: `index.html`, `hero-authors` paragraph.
- arXiv link: two `https://arxiv.org/abs/TODO` occurrences in `index.html`.
- BibTeX block in `index.html` (`#bibtex-code`).
