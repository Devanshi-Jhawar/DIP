"""Shared helpers for the Digital Image Processing practicals (OpenCV + NumPy + Matplotlib).

Every practical accepts an optional image path on the command line:
    python 01_basic_operations.py my_photo.jpg
If no path is given, a sample image shipped with scikit-image is used.

Figures are saved to practicals/outputs/<practical>/. Set DIP_SHOW=1 to also open
matplotlib windows.
"""

import os
import sys

import cv2
import matplotlib

SHOW = os.environ.get("DIP_SHOW", "0") == "1"
if not SHOW:
    matplotlib.use("Agg")  # save figures only, no GUI windows

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage import data  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(BASE_DIR, "outputs")


def input_path():
    """Image path passed on the command line, or None."""
    return sys.argv[1] if len(sys.argv) > 1 else None


def load_color():
    """Colour image as uint8 BGR (OpenCV convention). Default sample: skimage 'astronaut'."""
    path = input_path()
    if path:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img
    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


def load_gray(default="camera"):
    """Grayscale uint8 image. Default sample: skimage 'camera' (or any name passed in)."""
    path = input_path()
    if path:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return img
    return np.ascontiguousarray(getattr(data, default)())


def load_pair_gray():
    """Two grayscale uint8 images of identical size, for two-image operations."""
    a = load_gray("camera")
    b = cv2.resize(data.coins(), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return a, np.ascontiguousarray(b)


def out_dir(practical):
    path = os.path.join(OUT_ROOT, practical)
    os.makedirs(path, exist_ok=True)
    return path


def save_fig(fig, practical, filename):
    """Save a matplotlib figure into outputs/<practical>/ (and show it if DIP_SHOW=1)."""
    fig.savefig(os.path.join(out_dir(practical), filename), dpi=90, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)
    print(f"  saved outputs/{practical}/{filename}")


def show_grid(images, titles, practical, filename, cols=None):
    """Plot several images in one figure. Accepts BGR colour and grayscale arrays."""
    n = len(images)
    cols = cols or min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, img, title in zip(axes.ravel(), images, titles):
        if img.ndim == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        elif img.dtype == np.uint8:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    save_fig(fig, practical, filename)


def psnr(reference, test):
    """Peak signal-to-noise ratio (dB) between two 8-bit images. Higher is closer."""
    mse = np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def to_uint8(x):
    """Min-max normalise any array to 0..255 uint8 (for display)."""
    return cv2.normalize(np.asarray(x, dtype=np.float64), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)


def clip_u8(x):
    """Clip values to 0..255 and convert to uint8 (no rescaling)."""
    return np.clip(np.real(x), 0, 255).astype(np.uint8)
