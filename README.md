# RestoreNet

A unified pipeline for deblurring, colorization and vintage-photo restoration using classical filtering and deep priors (undergraduate Computer Vision project).

**Research question:** does classical restoration *before* deep colorization improve the restored result?

```
degraded gray ──► [1 classical: median → Wiener] ──► [2 pretrained colorizer] ──► [3 scratch/dust removal] ──► output
A baseline:  2            B hybrid:  1 → 2            C full RestoreNet:  1 → 2 → 3
```

## Honest status (read this first)

| Item | Status |
|---|---|
| Stages 1 and 3, evaluation, figures, CSVs, ablations, LaTeX report | Implemented **and executed** end-to-end (Python 3.12, CPU, 1 core). Results are bit-for-bit reproducible except timings. |
| Stage 2 deep colorizer (Zhang et al. 2016, Caffe via OpenCV DNN) | **Implemented but NOT run against real weights.** The development sandbox had no internet and its OpenCV 5.0 build has no Caffe importer. Only the surrounding pre/post-processing was tested (stub network, `python main.py selftest`). Download URLs were written from memory and are untested. **Expect to debug this once on a networked machine.** |
| Results currently in `outputs/` and the report | Produced with the **handcrafted fallback colorizer (fixed Lab tint, NOT deep learning)** on 4 synthetic images. They describe the restoration/artifact stages only. The research question is **not yet answered**. |
| LPIPS | Optional code path; not executed (torch/lpips not installed). |
| Streamlit UI | Not implemented (lowest priority; not enough time). |
| Fresh-environment `pip install` | Not testable offline; `requirements.txt` ranges are the intended ones. Developed on Python 3.12; code uses `from __future__ import annotations`, so 3.9+ should work (untested). |

## Setup from a fresh environment

```bash
git clone <your-repo> RestoreNet && cd RestoreNet
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # keeps opencv-python < 5 (needed for the Caffe importer)
python main.py selftest              # fast checks, no weights needed
```

## Commands

```bash
python main.py download-weights      # fetch + validate pretrained colorizer into models/ (needs internet)
python -m src.colorization check     # validate weights incl. a real forward pass  <-- run this first on the deep path!
python main.py all --colorizer deep  # samples -> prepare -> run (A/B/C) -> ablation -> report tables
# or step by step:
python main.py samples               # 4 bundled scikit-image images -> data/input/
python main.py prepare               # data/input -> data/ground_truth + data/degraded (+ data/degraded/scratch_masks)
python main.py run                   # -> outputs/experiment_results.csv, figures/, classical/, colorized/, masks/, final/
python main.py ablation              # -> outputs/ablation_classical.csv, ablation_artifact.csv
python main.py tables                # -> report/tables_generated.tex, report/results_meta.tex
cd report && latexmk -pdf main.tex   # build the report (pdflatex + bibtex)
```

`--colorizer auto` (default) uses the deep model if the weights validate and otherwise falls back **with a loud warning**; the colorizer name and a `colorizer_is_deep` flag are written into every CSV row and figure title. Use `--colorizer deep` to fail instead of falling back.

Each module also runs alone: `python -m src.classical_restoration --selftest`, `python -m src.artifact_removal --input x.png`, `python -m src.colorization colorize --input g.png --output c.png`, `python -m src.data_utils --help`.

### Manual weight download (if `download-weights` fails)
Put these three files in `models/`: `colorization_deploy_v2.prototxt`, `pts_in_hull.npy`, `colorization_release_v2.caffemodel` (~120+ MB), from the official release of Zhang et al., *Colorful Image Colorization* (github.com/richzhang/colorization, `caffe` branch, and the demo-v2 model link in that repo's README). Then `python -m src.colorization check`. If you find the right mirror, update `SOURCES` in `src/colorization.py`.

## Your own data
* **Clean colour images → synthetic benchmark:** drop JPG/JPEG/PNG into `data/input/`, then `python main.py prepare` (resizes to `--max-side`, creates degraded gray images, the scratch masks and keeps the clean originals in `data/ground_truth/`).
* **Real vintage photos (no ground truth):** `python main.py run --degraded-dir path/to/photos`. Outputs are saved, metrics are skipped (never compared against unrelated images). Colour inputs are converted to gray. Set `--blur-sigma` to your best guess of the blur (it is the assumed Wiener PSF) or the Wiener step may hurt.
* **Manual masks (recommended for real photos):** `--manual-mask-dir masks/` with `<image-stem>.png`, white = damaged.
* **Larger/real datasets (optional, not used here):** clean images from DIV2K (https://data.vision.ee.ethz.ch/cvl/DIV2K/, take a few validation images only); historical photos from Flickr Commons (https://www.flickr.com/commons — check each image's rights statement). Do not download the full datasets.
* Sample images come from scikit-image's bundled data; check its data licences before redistributing.

## What was measured (fallback colorizer, 4 images, seed 42; see `outputs/*.csv`)
These numbers are reproduced from the CSVs, not hand-written; re-run to refresh them.

| Config | gray PSNR (dB) | gray SSIM | RGB PSNR (dB) | RGB SSIM |
|---|---|---|---|---|
| degraded input | 19.03 | 0.463 | – | – |
| classical only | 19.24 | 0.585 | – | – |
| A baseline | 19.14 | 0.463 | 16.54 | 0.424 |
| B hybrid | 19.42 | 0.585 | 16.56 | 0.525 |
| C full | 22.93 | 0.623 | 17.90 | 0.559 |

Things we found (worth discussing honestly in the report): Wiener deconvolution alone *lowered* PSNR at noise σ=10 (its implementation is verified on a blur-only image); median/Gaussian denoising helped; scratch/dust removal gave the largest gain; running artifact removal *before* the classical stage scored higher than the specified order; the automatic scratch detector reaches only ≈0.6–0.7 precision/recall. A first detector version used a 3×3 median pre-filter that deleted 1-px scratches — caught by `python -m src.artifact_removal --selftest` and replaced. Parameter variants were evaluated on the same 4 images, so they are sensitivity analyses, not tuned results.

## Team workflow
| Member | Owns | Interface |
|---|---|---|
| 1 | `src/classical_restoration.py` | `restore_classical(gray, ClassicalConfig) -> ClassicalResult` |
| 2 | `src/colorization.py` | any object with `.name`, `.is_deep`, `.colorize(img) -> BGR uint8` |
| 3 | `src/artifact_removal.py`, `src/evaluation.py` | `remove_artifacts(img, mask, ArtifactConfig)`, `gray_metrics`, `color_metrics` |
| all | `src/pipeline.py`, `main.py` (integration) | `run_configs`, `process_dataset`, `run_ablation` |

Suggested remaining work, in order: (1) Member 2: get real weights working, run `python main.py all --colorizer deep`; (2) everyone: re-read `report/main.tex` Sections 5, 8–11 (marked TODO) against the new numbers; (3) add more images/degradation levels; (4) fill in author names; (5) optional LPIPS.

## Layout
```
data/{input,ground_truth,degraded}   models/   outputs/{classical,colorized,masks,final,figures}
src/{classical_restoration,colorization,artifact_removal,pipeline,evaluation,data_utils,visualization}.py
main.py  requirements.txt  report/{main.tex,references.bib}
```
