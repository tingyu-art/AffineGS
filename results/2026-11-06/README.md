# Selected results — 2026-11-06

This directory is a curated saved-result snapshot for the five main experiment notebooks. It contains figures, CSV/JSON metrics, synthetic vorticity arrays, and run provenance. It does **not** claim that every result was rerun with the final source version.

| Paper item | Files | Provenance / status |
| --- | --- | --- |
| Figure 5 | `figures/Fig_05*.pdf`, `data/Fig05/` | Complete saved 60-point Jacobian ablation. Original records contain trimmed means, not individual repeats or full source fingerprints. DNS at N=450/500 favors ConstGS. |
| Figure 6 | `figures/Fig_06*full_range.pdf`, `data/Fig06/` | Saved regularization sweeps; full-range exports reveal previously clipped no-regularization points. Notebook code retains its original axis limits. |
| Figure 7 | `figures/Fig_07*.pdf`, `data/Fig07/` | Complete saved density/noise aggregates; original data do not contain full per-repeat histories. |
| Figure 8–9; Tables 1–2 | `figures/*SAVED_RUN*`, `data/EXP2/`, `tables/Table_01*`, `tables/Table_02*` | Saved run `2026-11-06T01:47:25+08:00`; all 31 registered artifacts match its manifest. Some current computation sources differ. Notebook previews are explicitly restored from this run, not presented as a new training execution. |
| Figure 10 | `figures/Fig_10*`, `provenance/Fig10_*` | Both recordings and all six methods; source fingerprints matched the checked computation code. Main figures and Table C1 use identical saved velocity fields. Full real velocity arrays and event windows remain local. |
| Table 3 / additional experiments | `tables/Table_03*`, `tables/Gaussian_Count*`, `provenance/EXP_OTHER_run_metadata.json` | Fresh complete EXP_OTHER execution: 8 efficiency cases, 15 synthetic samples, 10 real windows. Other CUDA processes were active; times are shared-device single-pass observations, **not isolated performance benchmark values**. |
| Figure C1 | `figures/Fig_C1*`, `data/FigC1/` | Three datasets × 100 distinct samples × adaptive/fixed = 600 training records. Fixed count matches the adaptive final count. Current settings recorded; original record has no complete computation-source fingerprint. |
| Table C1 | `tables/Table_C1_RealData.*`, `data/TableC1/` | All 24 metrics evaluated from the exact Figure 10 fields, with flow/event/figure hashes recorded. |
| Frozen geometry and other vorticity plots | `supplementary_figures/`, `data/Supp01/` | Supporting results; frozen geometry also changes initialization and training schedule, so it does not isolate one causal factor. |

## Viewing notebooks

Open [EXP1](../../experiments/EXP1.ipynb), [EXP2](../../experiments/EXP2.ipynb), [EXP3](../../experiments/EXP3.ipynb), [EXP_OTHER](../../experiments/EXP_OTHER.ipynb), or [EXP_APPENDIX](../../experiments/EXP_APPENDIX.ipynb) directly on GitHub. Figures and HTML tables are saved inline. Missing-state progress-widget references are removed. Raster previews are sized for notebook browsing; original PDF/SVG files remain in this snapshot.

EXP1 plotting outputs and the appendix density-control plots were refreshed from their saved JSON files without retraining. The original training cells are unchanged. EXP2 restoration is labelled with its original saved run. EXP3 embeds all twelve real-data panels in its final gallery. EXP_OTHER contains actual complete execution outputs. The original local raw recordings and external PIV datasets are required to rerun data-dependent sections.

## Units and conventions

- Velocity error: px/ms; divergence/vorticity error: ms^-1; FWL/RSAT: dimensionless.
- EXP_OTHER memory is peak CUDA **allocated** memory from upload through optimization; final rendering is excluded from that reported scalar. Decimal MB = 1,000,000 bytes; MiB is also retained in the detailed table. Time includes final method outputs.
- EXP1 Jacobian/density/noise records use their existing trimmed mean of ten repetitions (drop one minimum and one maximum). Other aggregates retain their original arithmetic means and population standard deviations (ddof=0). Missing individual repeats are not reconstructed.
- CSV/JSON retain the precision present in the source records. Original files, recorded configurations and fingerprints are retained under `data/`, `config/` and `provenance/`.
- Some provenance records retain original local file paths and SHA256 fingerprints. They document source identity and are not instructions to access those original paths on another machine.

The manuscript PDF, local audit/diagnostic reports, historical result copies, original H5 recordings, checkpoints, CUDA binaries and the 109 MB download ZIP are excluded from this Git commit. `MANIFEST.json` lists every file in this curated directory with SHA256 and size.
