"""Stage 3 - vintage artifact (scratch / dust) removal with OpenCV.

Candidate detection (heuristic, NOT a perfect detector):
  1. L = luminance, lightly Gaussian-smoothed (sigma 0.8) to suppress noise. (A median filter was tried first but
     it deletes 1-px-wide scratches before detection; found by ``--selftest``.)
  2. Residual r = max( white-top-hat(L), black-hat(L) ) with a small elliptical structuring element.
     Top-hat = L - opening(L) keeps bright structures thinner than the element (bright scratches/dust);
     black-hat = closing(L) - L keeps thin dark ones.
  3. Threshold r > t, where t = max(abs_thresh, mad_k * 1.4826 * MAD(r)) (adaptive to texture/noise level).
  4. Connected components are kept if elongated and long enough (scratch) or small and strong (dust).
  5. Mask is dilated slightly so inpainting covers anti-aliased borders.
Known failure modes: thin real structures (hair, wires, text strokes, fine edges) are false positives; faint or
very wide scratches are missed. Manually drawn masks (``load_manual_mask``) are more reliable.

Inpainting: cv2.inpaint with Telea (fast marching, Telea 2004) or Navier-Stokes (Bertalmio et al. 2001).

Run standalone:
    python -m src.artifact_removal --input img.png --output-dir outputs/final [--mask my_mask.png] [--method telea|ns]
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from src.data_utils import read_image, write_image
except ImportError:  # pragma: no cover
    from data_utils import read_image, write_image

LOG = logging.getLogger("restorenet.artifacts")
INPAINT_FLAGS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}


@dataclass
class ArtifactConfig:
    prefilter_sigma: float = 0.8  # Gaussian pre-smoothing; 0 disables
    kernel_size: int = 5          # structuring element diameter; must exceed scratch width
    abs_thresh: float = 25.0      # minimum residual (grey levels) to count as an artifact
    mad_k: float = 6.0            # adaptive threshold multiplier on residual MAD
    min_length: int = 15          # min long side (px) of an elongated component (scratch)
    min_elongation: float = 3.0   # long/short side ratio of the component's min-area rectangle
    dust_max_area: int = 30       # max area (px) of a component treated as a dust speck
    dust_strength: float = 60.0   # dust must exceed this residual to be kept (limits noise false positives)
    dilate: int = 1               # iterations of 3x3 dilation of the final mask
    inpaint_radius: int = 3
    method: str = "telea"


def detect_artifacts(img: np.ndarray, cfg: ArtifactConfig | None = None) -> np.ndarray:
    """Return a uint8 mask (255 = suspected scratch/dust) for a gray or BGR image."""
    cfg = cfg or ArtifactConfig()
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), cfg.prefilter_sigma) if cfg.prefilter_sigma > 0 else gray
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.kernel_size, cfg.kernel_size))
    resid = np.maximum(cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, se),
                       cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, se)).astype(np.float32)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    thresh = max(cfg.abs_thresh, cfg.mad_k * 1.4826 * mad)
    # Hysteresis: retain faint pixels connected to a confident detection, but
    # reject weak isolated noise.  Components are defined on the low threshold.
    low = (resid > 0.6 * thresh).astype(np.uint8)
    high = resid > thresh
    n, labels, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    mask = np.zeros_like(gray)
    for i in range(1, n):
        component = labels == i
        if not np.any(high & component):
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        ys, xs = np.nonzero(component)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        long_s, short_s = max(rw, rh), max(min(rw, rh), 1.0)
        is_scratch = long_s >= cfg.min_length and long_s / short_s >= cfg.min_elongation
        is_dust = area <= cfg.dust_max_area and float(resid[ys, xs].max()) >= cfg.dust_strength
        if is_scratch or is_dust:
            mask[component] = 255
    if cfg.dilate > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=cfg.dilate)
    return mask


def load_manual_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a user-drawn mask (any non-black pixel / >127 = damaged), resized to ``shape`` if needed."""
    m = read_image(Path(path), grayscale=True)
    if m.shape != tuple(shape):
        LOG.warning("Mask %s has shape %s, resizing to %s (nearest)", path, m.shape, shape)
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(m > 127, 255, 0).astype(np.uint8)


def inpaint(img: np.ndarray, mask: np.ndarray, method: str = "telea", radius: int = 3) -> np.ndarray:
    """Fill masked pixels with OpenCV inpainting (works for gray or BGR uint8)."""
    if method not in INPAINT_FLAGS:
        raise ValueError(f"method must be one of {sorted(INPAINT_FLAGS)}")
    if mask.shape != img.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != image shape {img.shape[:2]}")
    if not mask.any():
        return img.copy()
    return cv2.inpaint(img, mask, radius, INPAINT_FLAGS[method])


def remove_artifacts(
    img: np.ndarray, mask: np.ndarray | None = None, cfg: ArtifactConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Detect (unless a manual ``mask`` is given) and inpaint. Returns (result, mask used)."""
    cfg = cfg or ArtifactConfig()
    used = mask if mask is not None else detect_artifacts(img, cfg)
    return inpaint(img, used, cfg.method, cfg.inpaint_radius), used


def selftest() -> None:
    """A clean synthetic scratch on a smooth gradient must be detected and largely inpainted away."""
    base = np.tile(np.linspace(60, 190, 128, dtype=np.float32), (128, 1)).astype(np.uint8)
    img = base.copy()
    cv2.line(img, (40, 5), (45, 120), 255, 1)
    truth = np.zeros_like(base)
    cv2.line(truth, (40, 5), (45, 120), 255, 1)
    mask = detect_artifacts(img)
    recall = float(((mask > 0) & (truth > 0)).sum() / (truth > 0).sum())
    out = inpaint(img, mask)
    err_before = float(np.abs(img.astype(int) - base)[truth > 0].mean())
    err_after = float(np.abs(out.astype(int) - base)[truth > 0].mean())
    assert recall > 0.9 and err_after < err_before / 3, (recall, err_before, err_after)
    print(f"artifact selftest OK (recall {recall:.2f}, scratch-pixel error {err_before:.1f} -> {err_after:.1f})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Scratch/dust removal.")
    ap.add_argument("--selftest", action="store_true", help="run built-in sanity check and exit")
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs") / "final")
    ap.add_argument("--mask", type=Path, default=None, help="optional manual mask (white = damaged)")
    ap.add_argument("--method", choices=sorted(INPAINT_FLAGS), default="telea")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.selftest:
        return selftest()
    if a.input is None:
        ap.error("--input is required unless --selftest is given")
    img = read_image(a.input)
    manual = load_manual_mask(a.mask, img.shape[:2]) if a.mask else None
    out, used = remove_artifacts(img, manual, ArtifactConfig(method=a.method))
    write_image(a.output_dir / f"{a.input.stem}_restored.png", out)
    write_image(a.output_dir / f"{a.input.stem}_mask.png", used)
    LOG.info("Masked %.2f%% of pixels (%s mask)", 100 * (used > 0).mean(), "manual" if a.mask else "auto")


if __name__ == "__main__":
    main()
