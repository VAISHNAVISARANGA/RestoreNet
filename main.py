"""RestoreNet command-line interface.

    python main.py samples            # export 4 bundled sample images into data/input/
    python main.py prepare            # data/input -> data/ground_truth + data/degraded (+ synthetic masks)
    python main.py download-weights   # fetch + validate the pretrained colorization model (needs internet)
    python main.py run                # configurations A/B/C -> outputs/ + experiment_results.csv
    python main.py ablation           # ablation CSVs
    python main.py tables             # LaTeX tables for the report (generated from the CSVs)
    python main.py all                # samples + prepare + run + ablation + tables
    python main.py selftest           # fast checks that need no weights

Colorizer choice (--colorizer): auto (deep if weights valid, otherwise labeled fallback), deep, fallback.
Bhai yehnkyu nhi ho raha changeeeeeee
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src import colorization, data_utils, evaluation, pipeline

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("restorenet")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # all synthetic degradation uses numpy Generators seeded per image


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--colorizer", choices=["auto", "deep", "fallback"], default="auto")
    ap.add_argument("--max-side", type=int, default=256, help="resize clean images so the longer side <= this")
    ap.add_argument("--blur-sigma", type=float, default=1.5, help="synthetic blur sigma (also the assumed Wiener PSF sigma)")
    ap.add_argument("--noise-sigma", type=float, default=10.0)
    ap.add_argument("--degraded-dir", type=Path, default=None, help="custom folder of degraded/vintage images for `run`")
    ap.add_argument("--manual-mask-dir", type=Path, default=None, help="folder of <stem>.png masks (white=damage)")
    ap.add_argument("--lpips", action="store_true", help="also compute LPIPS (needs torch+lpips installed)")
    ap.add_argument("cmd", choices=["samples", "prepare", "download-weights", "run", "ablation", "tables", "all", "selftest"])
    return ap


def cmd_samples(a) -> None:
    data_utils.export_sample_images(a.data_dir / "input")


def cmd_prepare(a) -> None:
    data_utils.make_synthetic_dataset(a.data_dir / "input", a.data_dir / "ground_truth", a.data_dir / "degraded",
                                      a.max_side, a.blur_sigma, a.noise_sigma, a.seed)


def _config(a) -> pipeline.PipelineConfig:
    cfg = pipeline.PipelineConfig()
    cfg.classical.wiener_blur_sigma = a.blur_sigma  # assumed PSF = the (known) synthetic blur; set manually for real photos
    return cfg


def cmd_run(a) -> pd.DataFrame:
    a.out_dir.mkdir(parents=True, exist_ok=True)
    col = colorization.load_colorizer(a.colorizer, a.models_dir)
    LOG.info("Colorizer: %s", col.name)
    return pipeline.process_dataset(a.degraded_dir or a.data_dir / "degraded", a.data_dir / "ground_truth", a.out_dir,
                                    col, _config(a), a.manual_mask_dir, a.lpips)


def cmd_ablation(a) -> None:
    col = colorization.load_colorizer(a.colorizer, a.models_dir)
    d1, d2 = pipeline.run_ablation(a.data_dir / "degraded", a.data_dir / "ground_truth", a.out_dir, col, _config(a))
    pd.options.display.width = 200
    print(evaluation.summarize(d1, "variant", ["classical_psnr_gray", "classical_ssim_gray", "colorized_psnr_rgb", "colorized_ssim_rgb", "colorized_ab_rmse"]).round(3).to_string())
    print(evaluation.summarize(d2, "variant", ["psnr_gray", "ssim_gray", "psnr_rgb", "ssim_rgb", "mask_precision", "mask_recall", "mask_f1"]).round(3).to_string())


def cmd_tables(a) -> None:
    """Generate report/tables_generated.tex and report/results_meta.tex from the CSV files (no hand-typed numbers)."""
    rep = ROOT / "report"
    main_df = pd.read_csv(a.out_dir / "experiment_results.csv")
    parts = [
        evaluation.to_latex_table(main_df, ["psnr_gray", "ssim_gray"], "Grayscale restoration metrics vs.\\ clean ground truth (mean $\\pm$ std over images).", "tab:gray"),
        evaluation.to_latex_table(main_df[main_df.config.isin(["A_baseline", "B_hybrid", "C_full"])],
                                  ["psnr_rgb", "ssim_rgb", "ab_rmse", "time_s"], "Colour metrics vs.\\ the original colour image and processing time (mean $\\pm$ std).", "tab:color"),
    ]
    for csv, group, cols, cap, lab in [
        ("ablation_classical.csv", "variant", ["classical_psnr_gray", "classical_ssim_gray", "colorized_psnr_rgb", "colorized_ab_rmse"], "Ablation of the classical stage (no artifact removal).", "tab:abl1"),
        ("ablation_artifact.csv", "variant", ["psnr_rgb", "ssim_rgb", "mask_precision", "mask_recall", "mask_f1"], "Ablation of artifact removal on top of configuration B (oracle = synthetic ground-truth mask).", "tab:abl2"),
    ]:
        p = a.out_dir / csv
        if p.is_file():
            parts.append(evaluation.to_latex_table(pd.read_csv(p), cols, cap, lab, group))
    (rep / "tables_generated.tex").write_text("\n".join(parts))
    name = str(main_df["colorizer"].iloc[0]).replace("_", "\\_")
    deep = "deep" if bool(main_df["colorizer_is_deep"].iloc[0]) else "NOT deep"
    n = main_df["image"].nunique()
    macros = [f"\\newcommand{{\\ColorizerName}}{{{name}}}", f"\\newcommand{{\\ColorizerKind}}{{{deep}}}", f"\\newcommand{{\\NImages}}{{{n}}}"]
    cfg_tag = {"degraded_input": "Deg", "classical_only": "Cls", "A_baseline": "A", "B_hybrid": "B", "C_full": "C"}
    met_tag = {"psnr_gray": "PsnrGray", "ssim_gray": "SsimGray", "psnr_rgb": "PsnrRgb", "ssim_rgb": "SsimRgb", "ab_rmse": "AbRmse"}
    means = main_df.groupby("config").mean(numeric_only=True)
    for cfg_name, ct in cfg_tag.items():
        for met, mt in met_tag.items():
            if cfg_name in means.index and met in means.columns and pd.notna(means.loc[cfg_name, met]):
                macros.append(f"\\newcommand{{\\m{ct}{mt}}}{{{means.loc[cfg_name, met]:.2f}}}")
    (rep / "results_meta.tex").write_text("\n".join(macros) + "\n")
    LOG.info("Wrote report/tables_generated.tex and report/results_meta.tex")


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(a.seed)
    if a.cmd == "samples":
        cmd_samples(a)
    elif a.cmd == "prepare":
        cmd_prepare(a)
    elif a.cmd == "download-weights":
        raise SystemExit(0 if colorization.download_weights(a.models_dir) else 1)
    elif a.cmd == "run":
        print(evaluation.summarize(cmd_run(a)).round(3).to_string())
    elif a.cmd == "ablation":
        cmd_ablation(a)
    elif a.cmd == "tables":
        cmd_tables(a)
    elif a.cmd == "selftest":
        from src import artifact_removal, classical_restoration

        classical_restoration.selftest()
        artifact_removal.selftest()
        colorization.selftest()
    elif a.cmd == "all":
        cmd_samples(a)
        cmd_prepare(a)
        print(evaluation.summarize(cmd_run(a)).round(3).to_string())
        cmd_ablation(a)
        cmd_tables(a)


if __name__ == "__main__":
    main()
