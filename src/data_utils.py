"""Data utilities: image I/O, synthetic degradation and sample-image export.

Degradation model used to create "vintage-like" inputs from a clean image x:

    y = S( clip( (x_gray * k_sigma_b) + n ) )

where x_gray is the grayscale clean image, k_sigma_b a Gaussian blur kernel,
n ~ N(0, sigma_n^2) additive Gaussian noise and S(.) overlays synthetic
scratches and dust. The scratch/dust mask is stored so detection quality can be
measured (and so it can serve as a stand-in for a manually drawn mask).

Run standalone:  python -m src.data_utils --help
"""
from __future__ import annotations

import argparse
import json
import logging
import zlib
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger("restorenet.data")

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MASK_SUBDIR = "scratch_masks"  # inside data/degraded/ (synthetic ground-truth masks)


# --------------------------------------------------------------------------- I/O
def list_images(folder: Path) -> list[Path]:
    """Return sorted JPG/JPEG/PNG files directly inside ``folder``."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image as uint8 (BGR or gray). Uses imdecode so non-ASCII paths work on Windows."""
    path = Path(path)
    if path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"Unsupported image type '{path.suffix}' (use jpg/jpeg/png): {path}")
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read {path}: {exc}") from exc
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(buf, flag)
    if img is None:
        raise ValueError(f"File is not a valid image: {path}")
    return img


def write_image(path: Path, img: np.ndarray) -> None:
    """Write an image (format from extension), creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix.lower() or ".png", img)
    if not ok:
        raise IOError(f"Failed to encode image for {path}")
    buf.tofile(str(path))


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longer side is at most ``max_side`` (never upscales)."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def to_gray(img: np.ndarray) -> np.ndarray:
    """BGR -> gray (uint8); gray inputs are returned unchanged."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def stable_seed(base_seed: int, name: str) -> int:
    """Platform-independent per-image seed (python's hash() is randomized per process)."""
    return (base_seed + zlib.crc32(name.encode("utf-8"))) % (2**32)


# ------------------------------------------------------------------ degradations
def gaussian_blur(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with kernel truncated at 3 sigma."""
    if sigma <= 0:
        return gray.copy()
    k = 2 * int(np.ceil(3 * sigma)) + 1
    return cv2.GaussianBlur(gray, (k, k), sigma, borderType=cv2.BORDER_REFLECT_101)


def add_gaussian_noise(gray: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive zero-mean Gaussian noise (sigma in 0-255 intensity units), clipped to uint8."""
    if sigma <= 0:
        return gray.copy()
    noisy = gray.astype(np.float32) + rng.normal(0.0, sigma, gray.shape).astype(np.float32)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def add_scratches(
    gray: np.ndarray,
    rng: np.random.Generator,
    n_scratches: int = 6,
    n_dust: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay thin, mostly vertical, slightly curved scratches and round dust specks.

    Returns (image, mask) where mask (0/255) marks every pixel that may have been altered.
    """
    h, w = gray.shape
    canvas = gray.copy()
    mask = np.zeros_like(gray)
    for _ in range(n_scratches):
        x0 = int(rng.integers(0, w))
        y0 = int(rng.integers(0, max(1, h // 3)))
        length = int(rng.integers(h // 3, h))
        pts = [(x0, y0)]
        x, y = float(x0), float(y0)
        for _ in range(6):  # polyline with random lateral drift => curved scratch
            y += length / 6
            x += rng.normal(0, 3)
            pts.append((int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        thick = int(rng.integers(1, 3))
        value = int(rng.choice([rng.integers(225, 256), rng.integers(0, 35)]))  # bright or dark
        cv2.polylines(canvas, [poly], False, value, thick, cv2.LINE_AA)
        cv2.polylines(mask, [poly], False, 255, thick, cv2.LINE_8)
    for _ in range(n_dust):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(1, 3))
        value = int(rng.choice([rng.integers(225, 256), rng.integers(0, 35)]))
        cv2.circle(canvas, (cx, cy), r, value, -1, cv2.LINE_AA)
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))  # cover anti-aliased fringe pixels
    return canvas, mask


def degrade_image(
    clean_bgr: np.ndarray,
    seed: int,
    blur_sigma: float = 1.5,
    noise_sigma: float = 10.0,
    n_scratches: int = 6,
    n_dust: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (degraded gray image, scratch mask) from a clean colour image."""
    rng = np.random.default_rng(seed)
    gray = to_gray(clean_bgr)
    out = gaussian_blur(gray, blur_sigma)
    out = add_gaussian_noise(out, noise_sigma, rng)
    return add_scratches(out, rng, n_scratches, n_dust)


# --------------------------------------------------------------- dataset building
def export_sample_images(dest: Path) -> list[Path]:
    """Copy a few images bundled with scikit-image (no download needed) into ``dest``.

    Check scikit-image's data documentation for the licence of each image before
    redistributing them outside your course project.
    """
    from skimage import data  # local import: only needed for this helper

    dest = Path(dest)
    written = []
    for name in ("astronaut", "chelsea", "coffee", "rocket"):
        rgb = getattr(data, name)()
        path = dest / f"{name}.png"
        write_image(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        written.append(path)
    LOG.info("Exported %d sample images to %s", len(written), dest)
    return written


def make_synthetic_dataset(
    input_dir: Path,
    gt_dir: Path,
    degraded_dir: Path,
    max_side: int = 256,
    blur_sigma: float = 1.5,
    noise_sigma: float = 10.0,
    seed: int = 42,
) -> list[str]:
    """input_dir (clean colour images) -> ground_truth/ (resized PNG) + degraded/ (gray PNG) + masks."""
    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No jpg/jpeg/png images found in {input_dir}")
    stems = []
    for path in images:
        clean = resize_max_side(read_image(path), max_side)
        degraded, mask = degrade_image(clean, stable_seed(seed, path.stem), blur_sigma, noise_sigma)
        write_image(Path(gt_dir) / f"{path.stem}.png", clean)
        write_image(Path(degraded_dir) / f"{path.stem}.png", degraded)
        write_image(Path(degraded_dir) / MASK_SUBDIR / f"{path.stem}.png", mask)
        stems.append(path.stem)
    params = dict(max_side=max_side, blur_sigma=blur_sigma, noise_sigma=noise_sigma, seed=seed, images=stems)
    (Path(degraded_dir) / "degradation_params.json").write_text(json.dumps(params, indent=2))
    LOG.info("Created %d degraded images (blur=%.2f, noise=%.1f, seed=%d)", len(stems), blur_sigma, noise_sigma, seed)
    return stems


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Create synthetic degraded images from clean ones.")
    ap.add_argument("--samples", action="store_true", help="export scikit-image sample images to --input-dir first")
    ap.add_argument("--input-dir", type=Path, default=DATA_DIR / "input")
    ap.add_argument("--gt-dir", type=Path, default=DATA_DIR / "ground_truth")
    ap.add_argument("--degraded-dir", type=Path, default=DATA_DIR / "degraded")
    ap.add_argument("--max-side", type=int, default=256)
    ap.add_argument("--blur-sigma", type=float, default=1.5)
    ap.add_argument("--noise-sigma", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.samples:
        export_sample_images(a.input_dir)
    make_synthetic_dataset(a.input_dir, a.gt_dir, a.degraded_dir, a.max_side, a.blur_sigma, a.noise_sigma, a.seed)


if __name__ == "__main__":
    main()
