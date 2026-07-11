# SPEAR-Net manuscript

Full-length manuscript targeting **Earth Science Informatics** (Springer) — a free
(subscription-route, no APC), reasonably fast SCI journal. Written to Q1 standards and
hedged strictly to the measured single-run results (see `../docs/MANUSCRIPT_CLAIMS.md`).

## Files
- `spearnet.tex` — the manuscript (compiles standalone with `article`).
- `references.bib` — bibliography. Entries preceded by `% NOTE: VERIFY ...` need their
  year/venue/pages/DOI confirmed against the publisher record before submission.
- `figures/` — drop the four figure PDFs here (see placeholders in the `.tex`).

## Build
```bash
pdflatex spearnet
bibtex   spearnet
pdflatex spearnet
pdflatex spearnet
```
Produces `spearnet.pdf` (~11 pages). Requires `lmodern`, `natbib`, `booktabs`, `siunitx`,
`orcidlink` (all standard; present on Overleaf).

## Submitting to Earth Science Informatics
1. Open the **Springer Nature LaTeX template** (`sn-jnl.cls`, default/`pdflatex` option) on
   Overleaf ("Springer Nature Journal" template).
2. Paste the body (Abstract → Declarations) into the template; map `\title`/`\author` to the
   template's `\title{}`, `\author[...]{}`, `\affil[...]{}` macros.
3. Switch the bibliography style to Springer's: `\bibliographystyle{spbasic}` (it ships with
   the template; `plainnat` is used here only so this repo compiles anywhere).
4. Fill the **Declarations** (funding, competing interests, author contributions).

## Figures to generate (placeholders are in the `.tex`)
1. `figures/architecture.pdf` — SPEAR-Net block diagram (Fig. 1).
2. `figures/priors.pdf` — S2 RGB + mask + the four PISP index maps (Fig. 5) — the notebook's
   PISP-visualization cell produces this panel.
3. `figures/qualitative.pdf` — RGB | GT | prediction | attention overlay (Fig. 4) — the
   notebook's explainability cell produces these panels.
4. (optional) a global map of the dataset's site distribution.

## Numbers of record
All tables use the full-budget (40-epoch, full-data) single-run results. If you later run
the 3-seed cell, replace the point estimates in Tables 2–3 with mean ± std and delete the
"single run / treat as comparable" hedges in Sections 4.2 and 5.

## Before you submit — checklist
- [ ] Verify the `% NOTE: VERIFY` bib entries (dataset, Tang 2023, Gallwey 2020, etc.).
- [ ] Insert the four figures.
- [ ] Fill author names, affiliations, ORCID, corresponding email, Declarations.
- [ ] Confirm the dataset tile/split counts in Table 1 against the annotation file you used.
- [ ] Keep the hedged language ("comparable / at a fraction of the cost"); do **not** add
      "beats / outperforms" or claim the gate improves accuracy (see `MANUSCRIPT_CLAIMS.md`).
