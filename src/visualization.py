"""Figures: 2x3 comparison grids, mask overlays and aggregate bar charts (matplotlib, Agg backend).

Run standalone to re-plot from a results CSV:
    python -m src.visualization --csv outputs/experiment_results.csv --out outputs/figures/summary.png
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless / cross-platform
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LOG = logging.getLogger("restorenet.viz")


def _panel(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to a common (w, h) and convert to RGB so all panels share one dimension."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if img.shape[1] > size[0] else cv2.INTER_NEAREST)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def comparison_figure(panels: list[tuple[str, np.ndarray]], out_path: Path, suptitle: str, ncols: int = 3) -> None:
    """Grid of (title, image) panels, all resized to the first panel's size."""
    h, w = panels[0][1].shape[:2]
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.3 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (title, img) in zip(np.atleast_1d(axes).ravel(), panels):
        ax.imshow(_panel(img, (w, h)))
        ax.set_title(title, fontsize=10)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def mask_figure(degraded: np.ndarray, detected: np.ndarray, oracle: np.ndarray | None, out_path: Path, title: str) -> None:
    """Degraded image next to the detected mask (and the synthetic ground-truth mask if known)."""
    items = [("Degraded input", degraded), ("Detected mask", detected)]
    if oracle is not None:
        items.append(("Synthetic GT mask", oracle))
    comparison_figure(items, out_path, title, ncols=len(items))


def summary_barplot(csv_path: Path, out_path: Path) -> None:
    """Mean PSNR/SSIM per configuration; error bars = std over images (only if >1 image)."""
    df = pd.read_csv(csv_path)
    order = [c for c in ["degraded_input", "classical_only", "A_baseline", "B_hybrid", "C_full"] if c in set(df["config"])]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    specs = [("psnr_gray", "PSNR gray (dB)"), ("ssim_gray", "SSIM gray"), ("psnr_rgb", "PSNR RGB (dB)"), ("ab_rmse", "ab-RMSE (lower=better)")]
    for ax, (col, lab) in zip(axes, specs):
        g = df[df["config"].isin(order)].groupby("config")[col]
        m, s = g.mean().reindex(order), g.std().reindex(order).fillna(0)
        ax.bar(range(len(order)), m.values, yerr=s.values, capsize=3, color="#4c78a8")
        ax.set_xticks(range(len(order)), [o.replace("_", "\n") for o in order], fontsize=8)
        ax.set_title(lab, fontsize=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Plot summary bars from experiment_results.csv")
    ap.add_argument("--csv", type=Path, default=Path("outputs") / "experiment_results.csv")
    ap.add_argument("--out", type=Path, default=Path("outputs") / "figures" / "summary.png")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary_barplot(a.csv, a.out)
    LOG.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
