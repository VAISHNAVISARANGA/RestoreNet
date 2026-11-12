"""Stage 1 - classical grayscale restoration (OpenCV + SciPy).

Observation model (assumed, and exactly true for our synthetic data):

    g = h * f + n            (* = 2-D convolution, h = blur PSF, n = additive noise)

Components
----------
* Median filter  - non-linear rank filter; removes impulsive/thin outliers, preserves edges
                   better than a box filter. Parameter: odd window ``median_ksize``.
* Gaussian filter - linear low-pass; reduces zero-mean Gaussian noise but also blurs edges.
                   Parameter: ``gaussian_sigma``.
* Wiener deconvolution - minimum-MSE linear restoration filter in the Fourier domain:

        F_hat(u,v) = conj(H(u,v)) / ( |H(u,v)|^2 + K ) * G(u,v),   K = S_n / S_f  (noise-to-signal ratio)

  K -> 0 gives the (noise-amplifying) inverse filter; large K gives a smooth, under-deblurred result.
  We approximate S_n/S_f by a *constant* sigma_n^2 / var(signal) (flat spectrum assumption, which is
  crude for natural images) and expose ``wiener_K`` so it can be overridden / swept.
  Wiener deconvolution needs the PSF. We assume a Gaussian PSF with user-supplied ``wiener_blur_sigma``;
  on real photographs this is unknown and a wrong value can make results worse (ringing).

None of these steps is guaranteed to improve every image; the ablation study measures it.

Run standalone:  python -m src.classical_restoration --input some.png --output-dir outputs/classical
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import scipy.fft as sfft

try:  # works both as `python -m src.x` and `python src/x.py`
    from src.data_utils import read_image, to_gray, write_image
except ImportError:  # pragma: no cover
    from data_utils import read_image, to_gray, write_image

LOG = logging.getLogger("restorenet.classical")


@dataclass
class ClassicalConfig:
    """Parameters of the classical stage. A value of 0 disables the corresponding step."""

    median_ksize: int = 3          # odd integer >= 3, or 0 to disable
    gaussian_sigma: float = 0.0    # std-dev (pixels) of Gaussian denoiser, 0 disables
    wiener_blur_sigma: float = 1.5  # assumed Gaussian PSF sigma for deconvolution, 0 disables
    wiener_K: float | None = None   # None => estimate sigma_n^2/var(signal) from the image
    order: str = "denoise_first"  # "denoise_first" or "deblur_first"
    nlm_h: float = 0.0             # non-local-means strength; 0 disables it

    def validate(self) -> None:
        if self.median_ksize not in (0,) and (self.median_ksize < 3 or self.median_ksize % 2 == 0):
            raise ValueError("median_ksize must be 0 or an odd integer >= 3")
        if self.gaussian_sigma < 0 or self.wiener_blur_sigma < 0 or self.nlm_h < 0:
            raise ValueError("denoising strengths and sigmas must be non-negative")
        if self.wiener_K is not None and self.wiener_K <= 0:
            raise ValueError("wiener_K must be positive")
        if self.order not in ("denoise_first", "deblur_first"):
            raise ValueError("order must be 'denoise_first' or 'deblur_first'")


@dataclass
class ClassicalResult:
    final: np.ndarray
    stages: dict[str, np.ndarray] = field(default_factory=dict)  # intermediate images, in order
    info: dict[str, float] = field(default_factory=dict)         # estimated sigma_n, K used, ...


# ------------------------------------------------------------------ building blocks
def denoise_median(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(gray, ksize)


def denoise_gaussian(gray: np.ndarray, sigma: float) -> np.ndarray:
    k = 2 * int(np.ceil(3 * sigma)) + 1
    return cv2.GaussianBlur(gray, (k, k), sigma, borderType=cv2.BORDER_REFLECT_101)

def denoise_nlm(gray: np.ndarray, h: float = 10) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)

def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Noise std-dev estimate using Immerkaer's (1996) Laplacian-difference operator with a
    median (robust) aggregation, so scratches/edges inflate it less than the mean version.

    The mask M = [[1,-2,1],[-2,4,-2],[1,-2,1]] cancels smooth structure; on white noise its response has
    std 6*sigma (sum of squared coefficients = 36). For N(0, s^2), median(|x|) = 0.6745 s.
    """
    m = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    resp = cv2.filter2D(gray.astype(np.float32), -1, m, borderType=cv2.BORDER_REFLECT_101)
    return float(np.median(np.abs(resp)) / (0.6745 * 6.0))


def gaussian_psf(sigma: float) -> np.ndarray:
    """Normalised 2-D Gaussian PSF truncated at 3 sigma."""
    k = 2 * int(np.ceil(3 * sigma)) + 1
    g = cv2.getGaussianKernel(k, sigma)
    return (g @ g.T).astype(np.float64)


def psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Zero-pad the PSF to ``shape`` and circularly shift its centre to (0,0) before the FFT."""
    pad = np.zeros(shape, dtype=np.float64)
    ph, pw = psf.shape
    pad[:ph, :pw] = psf
    pad = np.roll(pad, (-(ph // 2), -(pw // 2)), axis=(0, 1))
    return sfft.fft2(pad)


def wiener_deconvolve(gray: np.ndarray, psf: np.ndarray, K: float) -> np.ndarray:
    """Frequency-domain Wiener deconvolution with constant noise-to-signal ratio K.

    The image is reflect-padded before the FFT to limit wrap-around (circular convolution) artefacts.
    """
    ph, pw = psf.shape
    padded = np.pad(gray.astype(np.float64), ((ph, ph), (pw, pw)), mode="reflect")
    H = psf_to_otf(psf, padded.shape)
    G = sfft.fft2(padded)
    F_hat = np.conj(H) / (np.abs(H) ** 2 + K) * G
    out = np.real(sfft.ifft2(F_hat))[ph:-ph, pw:-pw]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------ pipeline
def restore_classical(gray: np.ndarray, cfg: ClassicalConfig | None = None) -> ClassicalResult:
    """Apply the configured classical steps; every intermediate image is returned in ``stages``."""
    cfg = cfg or ClassicalConfig()
    cfg.validate()
    gray = to_gray(gray)
    if gray.dtype != np.uint8:
        raise ValueError("expected a uint8 image")
    sigma_n = estimate_noise_sigma(gray)
    if cfg.wiener_K is not None:
        K = cfg.wiener_K
    else:  # flat-spectrum NSR approximation, floored to avoid an (unstable) near-inverse filter
        K = max(sigma_n**2 / max(float(gray.var()) - sigma_n**2, 1.0), 1e-3)
    res = ClassicalResult(final=gray, stages={"input": gray}, info={"noise_sigma_est": sigma_n, "wiener_K": K})

    def denoise(x: np.ndarray) -> np.ndarray:
        if cfg.median_ksize:
            x = denoise_median(x, cfg.median_ksize)
            res.stages["median"] = x
        if cfg.gaussian_sigma > 0:
            x = denoise_gaussian(x, cfg.gaussian_sigma)
            res.stages["gaussian"] = x
        if cfg.nlm_h > 0:
            x = denoise_nlm(x, cfg.nlm_h)
            res.stages["nlm"] = x
        return x

    def deblur(x: np.ndarray) -> np.ndarray:
        if cfg.wiener_blur_sigma > 0:
            x = wiener_deconvolve(x, gaussian_psf(cfg.wiener_blur_sigma), K)
            res.stages["wiener"] = x
        return x

    x = deblur(denoise(gray)) if cfg.order == "denoise_first" else denoise(deblur(gray))
    res.final = x
    return res


def selftest() -> None:
    """Wiener must undo a pure Gaussian blur (no noise) and the noise estimator must be roughly right."""
    try:
        from src.evaluation import psnr
    except ImportError:  # pragma: no cover
        from evaluation import psnr
    # Keep the sanity test independent of the optional scikit-image package.
    gt = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(gt, (48, 48), (240, 260), 210, thickness=-1)
    cv2.circle(gt, (350, 160), 90, 90, thickness=-1)
    cv2.line(gt, (40, 440), (470, 300), 255, thickness=9)
    gt[300:460, 280:460] = np.indices((160, 180)).sum(axis=0) % 2 * 180 + 40
    blurred = cv2.GaussianBlur(gt, (11, 11), 1.5)
    restored = wiener_deconvolve(blurred, gaussian_psf(1.5), 1e-3)
    gain = psnr(restored, gt) - psnr(blurred, gt)
    assert gain > 2.0, f"Wiener gain on blur-only image too small: {gain:.2f} dB"
    noisy = np.clip(gt + np.random.default_rng(0).normal(0, 10, gt.shape), 0, 255).astype(np.uint8)
    est = estimate_noise_sigma(noisy)
    assert 8.0 < est < 12.0, f"noise estimate off: {est:.2f}"
    print(f"classical selftest OK (Wiener gain {gain:.2f} dB on blur-only image, noise sigma est {est:.2f} for true 10)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Classical restoration of one grayscale image.")
    ap.add_argument("--selftest", action="store_true", help="run built-in sanity checks and exit")
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs") / "classical")
    ap.add_argument("--median", type=int, default=3)
    ap.add_argument("--gaussian", type=float, default=0.0)
    ap.add_argument("--blur-sigma", type=float, default=1.5)
    ap.add_argument("--K", type=float, default=None)
    ap.add_argument("--order", default="denoise_first", choices=["denoise_first", "deblur_first"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.selftest:
        return selftest()
    if a.input is None:
        ap.error("--input is required unless --selftest is given")
    cfg = ClassicalConfig(a.median, a.gaussian, a.blur_sigma, a.K, a.order)
    res = restore_classical(read_image(a.input, grayscale=True), cfg)
    for name, img in res.stages.items():
        write_image(a.output_dir / f"{a.input.stem}_{name}.png", img)
    LOG.info("Estimated noise sigma=%.2f, Wiener K=%.4f; wrote %d images to %s",
             res.info["noise_sigma_est"], res.info["wiener_K"], len(res.stages), a.output_dir)


if __name__ == "__main__":
    main()
