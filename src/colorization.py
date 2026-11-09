"""Stage 2 - colorization.

Primary model (deep learning, pretrained, CPU-compatible via OpenCV DNN):
    Zhang, Isola, Efros, "Colorful Image Colorization", ECCV 2016 (Caffe release v2).
    The network takes the CIE-Lab L channel (224x224, mean-centred by 50) and predicts a
    distribution over 313 quantised ab bins; the released model uses annealed-mean decoding
    (T=0.38, folded into the 'conv8_313_rh' scale 2.606 as in OpenCV's reference sample).
    The predicted ab (56x56) is up-sampled and combined with the ORIGINAL full-resolution L.

Fallback (NOT deep learning): ``TintColorizer`` applies a fixed luminance-dependent warm tint
in Lab space. It has no semantic knowledge of the image. It exists only so the pipeline can run
when the pretrained weights are unavailable, and every output/CSV row records which colorizer
produced it. Numbers obtained with it say nothing about deep colorization.

NOTE: the Caffe importer exists in OpenCV 4.x (``cv2.dnn.readNetFromCaffe``); it is missing from the
OpenCV 5.0 wheel tested here, so requirements.txt pins opencv-python<5.

Run standalone:
    python -m src.colorization download            # fetch + validate weights into models/
    python -m src.colorization check               # validate existing weights
    python -m src.colorization colorize --input g.png --output c.png [--fallback]
    python -m src.colorization selftest            # plumbing test with a stub network (no weights)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

try:
    from src.data_utils import PROJECT_ROOT, read_image, write_image
except ImportError:  # pragma: no cover
    from data_utils import PROJECT_ROOT, read_image, write_image

LOG = logging.getLogger("restorenet.colorization")
MODELS_DIR = PROJECT_ROOT / "models"

PROTOTXT = "colorization_deploy_v2.prototxt"
CAFFEMODEL = "colorization_release_v2.caffemodel"
POINTS = "pts_in_hull.npy"
CHECKSUM_FILE = "checksums.json"

# Candidate mirrors, tried in order. These were written from memory of the public release and could
# NOT be tested from the offline development sandbox: if one is dead, download manually (README).
SOURCES: dict[str, list[str]] = {
    PROTOTXT: ["https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt"],
    POINTS: ["https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy"],
    CAFFEMODEL: [
        "https://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel",
        "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel",
    ],
}
MIN_CAFFEMODEL_BYTES = 100 * 1024 * 1024  # the real file is >100 MB; guards against HTML error pages


class Colorizer(Protocol):
    name: str
    is_deep: bool

    def colorize(self, img: np.ndarray) -> np.ndarray: ...


def _as_l_channel(img: np.ndarray) -> np.ndarray:
    """uint8 gray/BGR image -> float32 Lab L channel in [0, 100] (colour inputs are converted to gray first)."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)[:, :, 0]


def _lab_to_bgr_u8(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    lab = np.concatenate([L[:, :, None], ab], axis=2).astype(np.float32)
    bgr = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0.0, 1.0)
    return np.rint(bgr * 255.0).astype(np.uint8)


# ----------------------------------------------------------------------- deep model
class ZhangColorizer:
    """Pretrained deep colorizer (Zhang et al. 2016) running on CPU through cv2.dnn."""

    is_deep = True
    name = "Zhang2016-pretrained (deep, Caffe via OpenCV DNN)"

    def __init__(self, models_dir: Path = MODELS_DIR, net: object | None = None) -> None:
        """``net`` is only for tests (a stub exposing setInput/forward); normally weights are loaded from disk."""
        if net is None:
            ok, msg = validate_weights(models_dir)
            if not ok:
                raise RuntimeError(f"Colorization weights invalid/missing: {msg}")
            if not hasattr(cv2.dnn, "readNetFromCaffe"):
                raise RuntimeError(f"cv2 {cv2.__version__} has no Caffe importer; install opencv-python<5")
            net = cv2.dnn.readNetFromCaffe(str(models_dir / PROTOTXT), str(models_dir / CAFFEMODEL))
            pts = np.load(models_dir / POINTS).transpose().reshape(2, 313, 1, 1).astype(np.float32)
            net.getLayer(net.getLayerId("class8_ab")).blobs = [pts]
            net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]
        self.net = net

    def colorize(self, img: np.ndarray) -> np.ndarray:
        """Deterministic: identical input -> identical output (no sampling, no randomness)."""
        L = _as_l_channel(img)
        h, w = L.shape
        l_rs = cv2.resize(L, (224, 224)) - 50.0
        self.net.setInput(cv2.dnn.blobFromImage(l_rs.astype(np.float32)))
        ab = np.asarray(self.net.forward())[0].transpose((1, 2, 0))  # 56x56x2
        if ab.shape[:2] != (56, 56) or ab.shape[2] != 2:
            raise RuntimeError(f"unexpected network output shape {ab.shape}")
        return _lab_to_bgr_u8(L, cv2.resize(ab, (w, h)))


# ------------------------------------------------------------------------- fallback
class TintColorizer:
    """Handcrafted fallback - NOT deep learning. Fixed warm tint as a function of luminance only."""

    is_deep = False
    name = "FALLBACK fixed Lab tint (handcrafted, NOT deep learning)"

    def colorize(self, img: np.ndarray) -> np.ndarray:
        L = _as_l_channel(img)
        t = L / 100.0
        a = 6.0 * (1.0 - t) + 2.0          # slightly reddish in shadows
        b = 10.0 + 18.0 * t                 # yellowish, stronger in highlights
        return _lab_to_bgr_u8(L, np.stack([a, b], axis=2))


# ----------------------------------------------------------------- weight handling
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_weights(models_dir: Path = MODELS_DIR, deep_check: bool = False) -> tuple[bool, str]:
    """Check presence, size and structure of the three required files.

    There is no vendor-published SHA-256 we can verify offline, so integrity is checked structurally
    (size, points array shape, prototxt layer names, optionally a real forward pass with
    ``deep_check=True``) and the hash of the first successful download is stored in models/checksums.json
    so team members can compare their copies.
    """
    models_dir = Path(models_dir)
    for fname in (PROTOTXT, POINTS, CAFFEMODEL):
        if not (models_dir / fname).is_file():
            return False, f"missing {models_dir / fname}"
    if (models_dir / CAFFEMODEL).stat().st_size < MIN_CAFFEMODEL_BYTES:
        return False, f"{CAFFEMODEL} is too small (<100 MB): probably a failed download"
    try:
        pts = np.load(models_dir / POINTS, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - any parse failure means invalid
        return False, f"cannot read {POINTS}: {exc}"
    if pts.shape != (313, 2):
        return False, f"{POINTS} has shape {pts.shape}, expected (313, 2)"
    proto = (models_dir / PROTOTXT).read_text(errors="ignore")
    if "class8_ab" not in proto or "conv8_313_rh" not in proto:
        return False, f"{PROTOTXT} does not look like the colorization deploy file"
    if deep_check:
        try:
            z = ZhangColorizer(models_dir)
            z.colorize(np.full((64, 64), 128, np.uint8))
        except Exception as exc:  # noqa: BLE001
            return False, f"network failed a forward pass: {exc}"
    return True, "ok"


def download_weights(models_dir: Path = MODELS_DIR, timeout: int = 60) -> bool:
    """Download missing files (resumable-by-skip), validate, record SHA-256. Returns True on success."""
    import requests  # imported here so the module imports without network libs

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for fname, urls in SOURCES.items():
        target = models_dir / fname
        if target.is_file() and target.stat().st_size > 0:
            continue
        for url in urls:
            part = target.with_suffix(target.suffix + ".part")
            try:
                LOG.info("Downloading %s from %s", fname, url)
                with requests.get(url, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    with open(part, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                part.replace(target)
                break
            except Exception as exc:  # noqa: BLE001 - try next mirror
                LOG.warning("  failed: %s", exc)
                part.unlink(missing_ok=True)
        else:
            LOG.error("Could not download %s from any source. See README for manual download.", fname)
    ok, msg = validate_weights(models_dir)
    if ok:
        sums = {f: _sha256(models_dir / f) for f in (PROTOTXT, POINTS, CAFFEMODEL)}
        (models_dir / CHECKSUM_FILE).write_text(json.dumps(sums, indent=2))
        LOG.info("Weights valid. SHA-256 recorded in %s", models_dir / CHECKSUM_FILE)
    else:
        LOG.error("Weights not usable: %s", msg)
    return ok


def load_colorizer(mode: str = "auto", models_dir: Path = MODELS_DIR) -> Colorizer:
    """mode: 'deep' (error if unavailable), 'fallback' (force handcrafted), 'auto' (deep if possible, else fallback + warning)."""
    if mode == "fallback":
        return TintColorizer()
    try:
        return ZhangColorizer(models_dir)
    except Exception as exc:  # noqa: BLE001
        if mode == "deep":
            raise
        LOG.warning("Deep colorizer unavailable (%s). USING HANDCRAFTED FALLBACK - results are NOT deep learning.", exc)
        return TintColorizer()


# --------------------------------------------------------------------------- CLI
class _StubNet:
    """Returns a constant (1,2,56,56) tensor. Tests the surrounding code only, NOT the model."""

    def setInput(self, blob: np.ndarray) -> None:  # noqa: N802
        assert blob.shape == (1, 1, 224, 224), blob.shape

    def forward(self) -> np.ndarray:
        out = np.zeros((1, 2, 56, 56), np.float32)
        out[:, 0], out[:, 1] = 10.0, 20.0
        return out


def selftest() -> None:
    gray = (np.linspace(0, 255, 96 * 128).reshape(96, 128)).astype(np.uint8)
    for col in (ZhangColorizer(net=_StubNet()), TintColorizer()):
        out = col.colorize(gray)
        assert out.shape == (96, 128, 3) and out.dtype == np.uint8
        assert np.array_equal(out, col.colorize(gray)), "colorizer must be deterministic"
    print("selftest OK (stub-network plumbing + fallback). This does NOT validate the real pretrained model.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    sub.add_parser("check")
    sub.add_parser("selftest")
    c = sub.add_parser("colorize")
    c.add_argument("--input", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--fallback", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.cmd == "download":
        raise SystemExit(0 if download_weights() else 1)
    if a.cmd == "check":
        ok, msg = validate_weights(deep_check=True)
        print("weights:", msg)
        raise SystemExit(0 if ok else 1)
    if a.cmd == "selftest":
        selftest()
        return
    col = load_colorizer("fallback" if a.fallback else "deep")
    write_image(a.output, col.colorize(read_image(a.input)))
    LOG.info("Colorized with: %s", col.name)


if __name__ == "__main__":
    main()
