"""Integration of the three stages and the experimental configurations.

    A_baseline : degraded -> colorize
    B_hybrid   : degraded -> classical restoration -> colorize
    C_full     : degraded -> classical restoration -> colorize -> artifact removal

The SAME colorizer object (same weights, same settings, deterministic) is applied to the raw and to the
classically restored input. Stage outputs are computed once and re-used (C re-uses B's colorized image), and
reported times are the sum of the stages a configuration actually needs.

Metrics are only computed when a ground-truth image with the same file stem exists in ``gt_dir``.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy

try:
    from src import evaluation as ev
    from src import visualization as viz
    from src.artifact_removal import ArtifactConfig, detect_artifacts, load_manual_mask, remove_artifacts
    from src.classical_restoration import ClassicalConfig, restore_classical
    from src.colorization import Colorizer
    from src.data_utils import MASK_SUBDIR, list_images, read_image, to_gray, write_image
except ImportError:  # pragma: no cover  (python src/pipeline.py)
    import evaluation as ev
    import visualization as viz
    from artifact_removal import ArtifactConfig, detect_artifacts, load_manual_mask, remove_artifacts
    from classical_restoration import ClassicalConfig, restore_classical
    from colorization import Colorizer
    from data_utils import MASK_SUBDIR, list_images, read_image, to_gray, write_image

LOG = logging.getLogger("restorenet.pipeline")
NAN = float("nan")


@dataclass
class PipelineConfig:
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)


@dataclass
class RunResult:
    images: dict[str, np.ndarray]   # degraded, classical, A, B, C
    mask: np.ndarray                # mask used for C
    times: dict[str, float]         # per configuration, seconds
    info: dict[str, float]


def run_configs(degraded: np.ndarray, colorizer: Colorizer, cfg: PipelineConfig,
                manual_mask: np.ndarray | None = None) -> RunResult:
    """Run configurations A/B/C plus the artifact-first comparison D."""
    gray = to_gray(degraded)
    # Detection is deliberately done before the classical stage: denoising and
    # deblurring can spread or weaken a scratch.  A supplied manual mask wins.
    mask_source = manual_mask if manual_mask is not None else detect_artifacts(gray, cfg.artifact)
    t0 = time.perf_counter()
    cls = restore_classical(gray, cfg.classical)
    t_cls = time.perf_counter() - t0
    t0 = time.perf_counter()
    A = colorizer.colorize(gray)
    t_A = time.perf_counter() - t0
    t0 = time.perf_counter()
    B = colorizer.colorize(cls.final)
    t_B = time.perf_counter() - t0
    t0 = time.perf_counter()
    C, mask = remove_artifacts(B, mask_source, cfg.artifact)
    t_art = time.perf_counter() - t0
    t0 = time.perf_counter()
    pre, _ = remove_artifacts(gray, None, cfg.artifact)
    D = colorizer.colorize(restore_classical(pre, cfg.classical).final)
    t_D = time.perf_counter() - t0
    return RunResult(
        images={"degraded": gray, "classical": cls.final, "A_baseline": A, "B_hybrid": B, "C_full": C,
            "D_artifact_first": D},
        mask=mask,
        times={"classical_only": t_cls, "A_baseline": t_A, "B_hybrid": t_cls + t_B,
               "C_full": t_cls + t_B + t_art, "D_artifact_first": t_D},
        info=cls.info,
    )


def _warmup(colorizer: Colorizer) -> None:
    """One throw-away call so first-call overhead does not distort the timing of config A."""
    colorizer.colorize(np.full((64, 64), 128, np.uint8))


def _row(image: str, config: str, colorizer: Colorizer, metrics: dict[str, float], t: float = NAN) -> dict:
    return {"image": image, "config": config, "colorizer": colorizer.name, "colorizer_is_deep": colorizer.is_deep,
            **metrics, "time_s": t}


def process_dataset(
    degraded_dir: Path, gt_dir: Path, out_dir: Path, colorizer: Colorizer, cfg: PipelineConfig,
    manual_mask_dir: Path | None = None, use_lpips: bool = False,
) -> pd.DataFrame:
    """Run A/B/C/D, save outputs, and export an oracle-selected best image when GT exists.

    ``*_best_by_rgb_psnr.png`` is for synthetic evaluation only: its selection
    uses the clean reference and therefore cannot be used for real photographs.
    """
    images = list_images(degraded_dir)
    if not images:
        raise FileNotFoundError(f"No degraded images in {degraded_dir}. Run `python main.py prepare` first.")
    lp = ev.LPIPSMetric() if use_lpips else None
    _warmup(colorizer)
    rows: list[dict] = []
    best_rows: list[dict[str, object]] = []
    for path in images:
        stem = path.stem
        degraded = read_image(path, grayscale=True)
        gt_path = next((p for p in (Path(gt_dir) / f"{stem}{e}" for e in (".png", ".jpg", ".jpeg")) if p.is_file()), None)
        gt = read_image(gt_path) if gt_path else None
        if gt is not None and gt.shape[:2] != degraded.shape:
            raise ValueError(f"{stem}: ground truth {gt.shape[:2]} and degraded {degraded.shape} differ in size")
        manual = None
        if manual_mask_dir is not None:
            mp = Path(manual_mask_dir) / f"{stem}.png"
            if mp.is_file():
                manual = load_manual_mask(mp, degraded.shape)
            else:
                LOG.warning("%s: no manual mask at %s, using automatic detection", stem, mp)
        res = run_configs(degraded, colorizer, cfg, manual)
        im = res.images
        # ---- save outputs
        write_image(out_dir / "classical" / f"{stem}.png", im["classical"])
        write_image(out_dir / "colorized" / f"{stem}_A.png", im["A_baseline"])
        write_image(out_dir / "colorized" / f"{stem}_B.png", im["B_hybrid"])
        write_image(out_dir / "masks" / f"{stem}_mask.png", res.mask)
        write_image(out_dir / "final" / f"{stem}_C.png", im["C_full"])
        write_image(out_dir / "final" / f"{stem}_D.png", im["D_artifact_first"])
        # ---- metrics + figure (need ground truth)
        oracle = None
        op = Path(degraded_dir) / MASK_SUBDIR / f"{stem}.png"
        if op.is_file():
            oracle = read_image(op, grayscale=True)
        if gt is None:
            LOG.warning("%s: no ground truth found; saved outputs but skipped metrics", stem)
            for k in ("classical_only", "A_baseline", "B_hybrid", "C_full", "D_artifact_first"):
                rows.append(_row(stem, k, colorizer, {}, res.times[k]))
            continue
        gt_gray = to_gray(gt)
        rows.append(_row(stem, "degraded_input", colorizer, ev.gray_metrics(im["degraded"], gt_gray)))
        rows.append(_row(stem, "classical_only", colorizer, ev.gray_metrics(im["classical"], gt_gray), res.times["classical_only"]))
        titles = {}
        color_metrics: dict[str, dict[str, float]] = {}
        for k in ("A_baseline", "B_hybrid", "C_full", "D_artifact_first"):
            m = ev.color_metrics(im[k], gt, lp)
            if k == "C_full" and oracle is not None:
                m["mask_precision"], m["mask_recall"], m["mask_f1"] = ev.mask_prf(res.mask, oracle)
            rows.append(_row(stem, k, colorizer, m, res.times[k]))
            color_metrics[k] = m
            titles[k] = f"{k.replace('_', ' ')}\nRGB PSNR {m['psnr_rgb']:.2f} dB | SSIM {m['ssim_rgb']:.3f}"
        # This is deliberately an oracle selector: only synthetic/evaluation
        # images have the clean reference needed to calculate RGB PSNR.
        best_config = max(
            color_metrics,
            key=lambda k: (color_metrics[k]["psnr_rgb"], color_metrics[k]["ssim_rgb"]),
        )
        write_image(out_dir / "final" / f"{stem}_best_by_rgb_psnr.png", im[best_config])
        write_image(out_dir / "final" / f"{stem}_final.png", im[best_config])
        best_rows.append({
            "image": stem,
            "selected_config": best_config,
            "selection_metric": "RGB PSNR (SSIM tie-break)",
            "psnr_rgb": color_metrics[best_config]["psnr_rgb"],
            "ssim_rgb": color_metrics[best_config]["ssim_rgb"],
        })
        gd, gc = ev.gray_metrics(im["degraded"], gt_gray), ev.gray_metrics(im["classical"], gt_gray)
        viz.comparison_figure(
            [("Original clean (ground truth)", gt),
             (f"Degraded input\ngray PSNR {gd['psnr_gray']:.2f} dB | SSIM {gd['ssim_gray']:.3f}", im["degraded"]),
             (f"Classical restoration\ngray PSNR {gc['psnr_gray']:.2f} dB | SSIM {gc['ssim_gray']:.3f}", im["classical"]),
             (titles["A_baseline"], im["A_baseline"]), (titles["B_hybrid"], im["B_hybrid"]),
            (titles["C_full"] + "\n(final RestoreNet)", im["C_full"]),
             (titles["D_artifact_first"] + "\n(extra comparison)", im["D_artifact_first"])],
            out_dir / "figures" / f"{stem}_comparison.png",
            f"{stem}  |  colorizer: {colorizer.name}",
        )
        viz.mask_figure(im["degraded"], res.mask, oracle, out_dir / "figures" / f"{stem}_masks.png", f"{stem}: scratch/dust masks")
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out_dir / "experiment_results.csv", index=False)
    if best_rows:
        pd.DataFrame(best_rows).round(4).to_csv(out_dir / "best_by_rgb_psnr.csv", index=False)
    viz.summary_barplot(out_dir / "experiment_results.csv", out_dir / "figures" / "summary.png")
    write_metadata(out_dir, colorizer, cfg, [p.stem for p in images], use_lpips and bool(lp and lp.available))
    LOG.info("Wrote %s", out_dir / "experiment_results.csv")
    return df


# ------------------------------------------------------------------------ ablation
CLASSICAL_VARIANTS: dict[str, ClassicalConfig] = {
    "none": ClassicalConfig(0, 0.0, 0.0),
    "median3": ClassicalConfig(3, 0.0, 0.0),
    "gaussian1.0": ClassicalConfig(0, 1.0, 0.0),
    "wiener_autoK": ClassicalConfig(0, 0.0, 1.5),
    "median3+wiener_autoK": ClassicalConfig(3, 0.0, 1.5),
    "wiener_autoK+median3": ClassicalConfig(3, 0.0, 1.5, order="deblur_first"),
    "median3+wiener_K0.01": ClassicalConfig(3, 0.0, 1.5, 0.01),
    "median3+wiener_K0.05": ClassicalConfig(3, 0.0, 1.5, 0.05),
    "median3+wiener_wrongPSF2.5": ClassicalConfig(3, 0.0, 2.5),
    "nlm10": ClassicalConfig(0, 0.0, 0.0, nlm_h=10),
    "nlm10+wiener_K0.05": ClassicalConfig(0, 0.0, 1.5, 0.05, nlm_h=10),
}


def run_ablation(degraded_dir: Path, gt_dir: Path, out_dir: Path, colorizer: Colorizer, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(1) classical-stage variants -> colorize (no artifact removal); (2) artifact-removal variants on config B."""
    _warmup(colorizer)
    rows1, rows2 = [], []
    for path in list_images(degraded_dir):
        stem = path.stem
        gt_path = Path(gt_dir) / f"{stem}.png"
        if not gt_path.is_file():
            LOG.warning("ablation skipped for %s (no ground truth)", stem)
            continue
        gt, deg = read_image(gt_path), read_image(path, grayscale=True)
        gt_gray = to_gray(gt)
        for name, ccfg in CLASSICAL_VARIANTS.items():
            cls = restore_classical(deg, ccfg).final
            g = ev.gray_metrics(cls, gt_gray)
            col = ev.color_metrics(colorizer.colorize(cls), gt)
            rows1.append({"image": stem, "variant": name, "classical_psnr_gray": g["psnr_gray"],
                          "classical_ssim_gray": g["ssim_gray"], **{f"colorized_{k}": v for k, v in col.items() if k != "lpips"}})
        # artifact-removal variants applied to the default hybrid (B) colorized image
        B = colorizer.colorize(restore_classical(deg, cfg.classical).final)
        op = Path(degraded_dir) / MASK_SUBDIR / f"{stem}.png"
        oracle = read_image(op, grayscale=True) if op.is_file() else None
        variants: dict[str, tuple[np.ndarray | None, str]] = {"none (=B)": (None, "")}
        for meth in ("telea", "ns"):
            variants[f"auto_{meth}"] = (None, meth)
            if oracle is not None:
                variants[f"oracle_{meth}"] = (oracle, meth)
        for name, (mask, meth) in variants.items():
            if name.startswith("none"):
                out, used = B, np.zeros(B.shape[:2], np.uint8)
            else:
                out, used = remove_artifacts(B, mask, dataclasses.replace(cfg.artifact, method=meth))
            m = ev.color_metrics(out, gt)
            if oracle is not None:
                m["mask_precision"], m["mask_recall"], m["mask_f1"] = ev.mask_prf(used, oracle)
            rows2.append({"image": stem, "variant": name, **{k: v for k, v in m.items() if k != "lpips"}})
        # alternative stage ORDER: artifact removal first (on the gray degraded image), then classical + colorize
        for src_name, mask in (("auto", None), ("oracle", oracle)):
            if src_name == "oracle" and oracle is None:
                continue
            pre, used = remove_artifacts(deg, mask, cfg.artifact)
            m = ev.color_metrics(colorizer.colorize(restore_classical(pre, cfg.classical).final), gt)
            if oracle is not None:
                m["mask_precision"], m["mask_recall"], m["mask_f1"] = ev.mask_prf(used, oracle)
            rows2.append({"image": stem, "variant": f"{src_name}_telea_BEFORE_classical", **{k: v for k, v in m.items() if k != "lpips"}})
    d1, d2 = pd.DataFrame(rows1), pd.DataFrame(rows2)
    d1.round(4).to_csv(out_dir / "ablation_classical.csv", index=False)
    d2.round(4).to_csv(out_dir / "ablation_artifact.csv", index=False)
    return d1, d2


def write_metadata(out_dir: Path, colorizer: Colorizer, cfg: PipelineConfig, images: list[str], lpips_used: bool) -> None:
    meta = {
        "colorizer": colorizer.name, "colorizer_is_deep": colorizer.is_deep, "lpips_computed": lpips_used,
        "images": images, "config": dataclasses.asdict(cfg),
        "versions": {"python": sys.version.split()[0], "platform": platform.platform(), "opencv": cv2.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__, "scikit-image": version("scikit-image"),
                     "pandas": pd.__version__},
    }
    (Path(out_dir) / "run_metadata.json").write_text(json.dumps(meta, indent=2))
