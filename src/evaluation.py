"""Evaluation metrics (always computed against the corresponding clean ground truth).

PSNR  = 10 log10( 255^2 / MSE )
SSIM  = skimage.metrics.structural_similarity (Wang et al. 2004, 7x7 uniform window defaults),
        channel_axis used for colour images.
ab-RMSE = sqrt(mean(da^2 + db^2)) between Lab chroma of output and ground truth (colour fidelity only).
LPIPS = optional (needs `pip install torch lpips` and pretrained AlexNet weights, i.e. internet on first use).

Interpretation caveat: colorization is ill-posed - many plausible colourings exist - so PSNR/SSIM/ab-RMSE
versus ONE original colour image penalise plausible-but-different colours. Gray-level metrics
(``*_gray``) isolate restoration quality (blur/noise/scratches); colour metrics (``*_rgb``, ``ab_rmse``)
mix restoration and colorization. Keep the two groups separate when reporting.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LOG = logging.getLogger("restorenet.eval")


def _check(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    _check(a, b)
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10.0 * math.log10((255.0 ** 2) / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    _check(a, b)
    if a.ndim not in (2, 3):
        raise ValueError(f"expected a grayscale or colour image, got {a.ndim} dimensions")
    if min(a.shape[:2]) < 7:
        raise ValueError("images must be at least 7x7 for SSIM")

    # Match skimage's default 7x7 uniform-window SSIM without requiring skimage.
    channels_a = np.moveaxis(a, 2, 0) if a.ndim == 3 else a[None, ...]
    channels_b = np.moveaxis(b, 2, 0) if b.ndim == 3 else b[None, ...]
    scores = []
    for channel_a, channel_b in zip(channels_a, channels_b):
        x, y = channel_a.astype(np.float64), channel_b.astype(np.float64)
        mean_x = cv2.boxFilter(x, -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT)
        mean_y = cv2.boxFilter(y, -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT)
        covariance_scale = 49.0 / 48.0
        var_x = (cv2.boxFilter(x * x, -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT) - mean_x**2) * covariance_scale
        var_y = (cv2.boxFilter(y * y, -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT) - mean_y**2) * covariance_scale
        cov_xy = (cv2.boxFilter(x * y, -1, (7, 7), normalize=True, borderType=cv2.BORDER_REFLECT) - mean_x * mean_y) * covariance_scale
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        score = ((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)) / ((mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2))
        scores.append(float(score[3:-3, 3:-3].mean()))
    return float(np.mean(scores))


def ab_rmse(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    _check(a_bgr, b_bgr)
    la = cv2.cvtColor(a_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)[:, :, 1:]
    lb = cv2.cvtColor(b_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)[:, :, 1:]
    return float(math.sqrt(np.mean(np.sum((la - lb) ** 2, axis=2))))


def mask_prf(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    """Pixel-wise precision, recall, F1 of a detected mask against a ground-truth mask."""
    p, g = pred > 0, gt > 0
    tp = float(np.logical_and(p, g).sum())
    prec = tp / p.sum() if p.any() else float("nan")
    rec = tp / g.sum() if g.any() else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) > 0 else float("nan")
    return prec, rec, f1


class LPIPSMetric:
    """Lazy optional LPIPS (AlexNet). ``available`` is False if torch/lpips/weights are missing."""

    def __init__(self) -> None:
        self.available = False
        try:
            import lpips  # type: ignore
            import torch  # type: ignore

            self._torch, self._model = torch, lpips.LPIPS(net="alex", verbose=False).eval()
            self.available = True
        except Exception as exc:  # noqa: BLE001
            LOG.info("LPIPS disabled (%s)", exc)

    def __call__(self, a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
        if not self.available:
            return float("nan")
        t = lambda x: self._torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 127.5 - 1
        with self._torch.no_grad():
            return float(self._model(t(a_bgr), t(b_bgr)).item())


def gray_metrics(out_gray: np.ndarray, gt_gray: np.ndarray) -> dict[str, float]:
    return {"psnr_gray": psnr(out_gray, gt_gray), "ssim_gray": ssim(out_gray, gt_gray)}


def color_metrics(out_bgr: np.ndarray, gt_bgr: np.ndarray, lp: LPIPSMetric | None = None) -> dict[str, float]:
    gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    return {
        **gray_metrics(gray(out_bgr), gray(gt_bgr)),  # luminance of the coloured output vs luminance of GT
        "psnr_rgb": psnr(out_bgr, gt_bgr),
        "ssim_rgb": ssim(out_bgr, gt_bgr),
        "ab_rmse": ab_rmse(out_bgr, gt_bgr),
        "lpips": lp(out_bgr, gt_bgr) if lp else float("nan"),
    }


METRIC_COLS = ["psnr_gray", "ssim_gray", "psnr_rgb", "ssim_rgb", "ab_rmse", "lpips", "time_s"]


def summarize(df: pd.DataFrame, group: str = "config", cols: list[str] | None = None) -> pd.DataFrame:
    """Mean and std per group over images (std is NaN for a single image; not a significance test)."""
    cols = [c for c in (cols or METRIC_COLS) if c in df.columns and df[c].notna().any()]
    g = df.groupby(group, sort=False)[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    out.insert(0, "n_images", df.groupby(group, sort=False).size())
    return out.reset_index()


def to_latex_table(df: pd.DataFrame, cols: list[str], caption: str, label: str, group: str = "config") -> str:
    """Hand-rolled booktabs table of mean +- std (avoids pandas' jinja2 dependency)."""
    s = summarize(df, group, cols)
    head = " & ".join([group.capitalize()] + [c.replace("_", r"\_") for c in cols]) + r" \\"
    rows = []
    for _, r in s.iterrows():
        cells = [str(r[group]).replace("_", r"\_")]
        for c in cols:
            m, sd = r.get(f"{c}_mean", float("nan")), r.get(f"{c}_std", float("nan"))
            cells.append("--" if m != m else (f"{m:.3f}" if sd != sd else f"{m:.3f} $\\pm$ {sd:.3f}"))
        rows.append(" & ".join(cells) + r" \\")
    return ("\\begin{table}[t]\n\\centering\\small\n\\caption{" + caption + "}\\label{" + label + "}\n"
            "\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n" + head + "\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


if __name__ == "__main__":  # tiny self-check
    x = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert math.isinf(psnr(x, x)) and abs(ssim(x, x) - 1) < 1e-9 and ab_rmse(x, x) == 0
    print("evaluation self-check OK")
