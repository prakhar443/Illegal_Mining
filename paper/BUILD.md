# Building the manuscript (RSASE / Elsevier)

The manuscript is **`spearnet.tex`**, written for Elsevier's **`elsarticle`** class
(RSASE = *Remote Sensing Applications: Society and Environment*).

> ⚠️ Do **not** compile it inside a Springer Nature project (`sn-article.tex` /
> `sn-jnl.cls`). Those use different author macros (`\cnotenum`, `\@corref`) and will throw
> "Option clash", "Token not allowed", and author-macro errors. RSASE is Elsevier — use
> `elsarticle`.

## Overleaf (recommended)

1. **New Project → Blank Project.** Delete the default `main.tex`.
2. Upload from this `paper/` folder:
   - `spearnet.tex`
   - `references.bib`
   - `highlights.tex` (separate Elsevier Highlights file)
   - the **`paper_figures/`** folder containing the 7 figure PDFs (see below)
3. **Menu → Settings → Main document → `spearnet.tex`.**
4. **Recompile.** `elsarticle.cls` and `elsarticle-num.bst` ship with Overleaf.

If you already have a Springer project open: **delete `sn-article.tex`** (and `sn-jnl.cls`),
or set the Main document to `spearnet.tex`. Every error that names `sn-article.tex` is coming
from that file, not from the manuscript.

## Local

```bash
pdflatex spearnet
bibtex   spearnet
pdflatex spearnet
pdflatex spearnet      # two final passes resolve all \cite references
```

The **"Author undefined for citation …"** warnings are normal on the first pass — they clear
after `bibtex` + the two final `pdflatex` runs (Overleaf does this automatically; just
Recompile once more if any linger).

The three **"empty pages in …"** BibTeX messages (`oktay2018attentionunet`,
`loshchilov2019adamw`, `loshchilov2017sgdr`) are harmless: those are ICLR/MIDL papers with no
page numbers. They do not affect the PDF.

## Figures required (in `paper_figures/`)

Produced by the Colab notebook — cells **17** and **17b**:

| File | Source |
|------|--------|
| `architecture.pdf` | compile `figures/architecture.tex` standalone |
| `priors.pdf` | notebook cell 9 |
| `qualitative.pdf` | notebook cell 13 |
| `fig_dataset_map.pdf`, `fig_efficiency.pdf`, `fig_ablation.pdf`, `fig_area_recall.pdf` | notebook cell 17 (`make_figures.py`) |

`\graphicspath{{paper_figures/}{figures/}}` in the preamble finds them in either folder.

## Submission package (Editorial Manager upload order)

1. Cover letter — `cover_letter.md`
2. Highlights — `highlights.tex` (3–5 bullets, ≤85 chars each)
3. Manuscript — `spearnet.tex` (+ `references.bib`, figures)
4. Suggested reviewers — `suggested_reviewers.md`
