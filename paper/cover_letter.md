# Cover letter — *Remote Sensing Applications: Society and Environment*

*(Replace the bracketed placeholders, paste into the submission system or an institutional
letterhead, and remove this line.)*

---

23 July 2026

To the Editor-in-Chief
*Remote Sensing Applications: Society and Environment*

Dear Editor,

We submit our manuscript, **"Distinguishing Artisanal from Industrial Mining in Sentinel-2
Imagery: A Spectral-Prior, Recall-Oriented Segmentation Approach and a Diagnosis of Where
Small-Site Detection Fails,"** for consideration as a research article.

Artisanal and small-scale mining (ASM) is one of the most consequential yet poorly mapped
land uses in the Global South: it drives mercury contamination, deforestation and river
siltation, and it sits at the centre of a governance gap because regulators must know not
just *where* mining occurs but *at what scale* — artisanal (often informal or illegal) versus
industrial (regulated). This is squarely a *society-and-environment* problem, and it is the
problem our paper addresses. We believe it fits the journal's scope better than a
methods-only venue because our contribution is framed around the monitoring need, the
policy-relevant scale distinction, and honest reporting of what such a tool can and cannot
support operationally.

The manuscript makes four contributions of interest to the journal's readership:

1. **A scale-aware mapping task.** We train and evaluate on a globally distributed,
   expert-verified dataset with an explicit artisanal-versus-industrial label, producing an
   output of direct regulatory relevance rather than a generic "mining/no-mining" mask.

2. **A compact, deployable model.** SPEAR-Net (1.06 M parameters, 0.60 GFLOPs) uses a
   six-band spectral-index prior (NDVI, MNDWI, NDTI, BSI) and a recall-weighted objective,
   and runs within free-tier and CPU-only budgets — the hardware realistically available to
   many environmental agencies and academic groups.

3. **A diagnosis of where detection fails.** Using area-stratified recall evaluated across
   our model *and* three much larger baselines, we show that the small-site deficit is not
   closed by 21–25× more parameters — it is a *structural* limitation — while the six-band
   spectral prior measurably improves recovery of the hardest small sites over both the
   baselines and an RGB-only variant.

4. **A reproducible, free-tier configuration.** The full pipeline reproduces from a 35.6 MB
   annotation set and on-demand Sentinel-2, with no credentials and no bulk download, on a
   free cloud GPU.

We report single-run results and frame our claims conservatively as *parity plus a specific
small-site advantage*, and we include an explicit responsible-use/dual-use discussion given
that ASM is a livelihood for many communities. We believe this candour is appropriate for a
society-and-environment audience.

The work is original, is not under consideration elsewhere, and all authors have approved the
submission. We declare no competing interests. The code, configurations and data-preparation
pipeline are openly released.

Thank you for your consideration.

Sincerely,

Prakhar Mishra, on behalf of all authors
(Bhavya Yajush Awasthi, Prakhar Mishra, Tanuja, Akshita Mishra, Pranav Mishra)
Department of Information Technology, Delhi Technological University, New Delhi, India
443prakharmishra@gmail.com
