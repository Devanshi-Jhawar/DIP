"""Practical 5 - Histogram calculation and equalisation.

5a uses OpenCV built-in functions (cv2.calcHist, cv2.equalizeHist).
5b performs the same steps with NumPy only, following the formulas:

    h[k] = number of pixels with intensity k
    p[k] = h[k] / N                                  (probability of level k)
    c[k] = sum of p[j] for j <= k                    (cumulative distribution)
    s[k] = round((L - 1) * (c[k] - c_min) / (1 - c_min))   (mapping, L = 256)
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import load_gray, save_fig

PRACTICAL = "05_histogram"


def histogram_builtin(gray):
    """5a. Histogram and equalisation using OpenCV built-in functions."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    equalised = cv2.equalizeHist(gray)
    return hist, equalised


def histogram_manual(gray):
    """5b. Histogram and equalisation written from the formulas above (no cv2/skimage)."""
    levels = 256
    hist = np.bincount(gray.ravel(), minlength=levels)
    pdf = hist / gray.size
    cdf = np.cumsum(pdf)
    cdf_min = cdf[hist > 0].min()
    lut = np.round((levels - 1) * (cdf - cdf_min) / (1.0 - cdf_min))
    lut = np.clip(lut, 0, levels - 1).astype(np.uint8)
    equalised = lut[gray]  # apply the look-up table to every pixel
    return hist, lut, equalised


def plot_histograms(before, after, hist_before, hist_after, filename, title):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].imshow(before, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(after, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Equalised")
    axes[1, 0].bar(range(256), hist_before, width=1.0, color="gray")
    axes[1, 0].set_title("Histogram (original)")
    axes[1, 1].bar(range(256), hist_after, width=1.0, color="gray")
    axes[1, 1].set_title("Histogram (equalised)")
    for ax in axes[1]:
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Pixel count")
        ax.set_xlim(0, 255)
    for ax in axes[0]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    save_fig(fig, PRACTICAL, filename)


def main():
    gray = load_gray()

    # 5a. Built-in functions ----------------------------------------------
    hist_a, eq_a = histogram_builtin(gray)
    print(f"5a. Histogram total = {int(hist_a.sum())} (image has {gray.size} px)")
    plot_histograms(gray, eq_a, hist_a, np.bincount(eq_a.ravel(), minlength=256),
                    "5a_builtin.png", "5a. Histogram equalisation (OpenCV built-in)")

    # 5b. Without built-in functions ----------------------------------------
    hist_b, lut, eq_b = histogram_manual(gray)
    diff = np.abs(eq_a.astype(int) - eq_b.astype(int))
    print(f"5b. Manual histogram identical to built-in: {np.array_equal(hist_a, hist_b)}")
    print(f"5b. Manual equalisation vs cv2.equalizeHist: max difference {diff.max()} "
          f"grey level(s), mean difference {diff.mean():.3f}")
    print(f"5b. Mapping function: input 0 -> {lut[0]}, input 128 -> {lut[128]}, "
          f"input 255 -> {lut[255]}")
    plot_histograms(gray, eq_b, hist_b, np.bincount(eq_b.ravel(), minlength=256),
                    "5b_manual.png", "5b. Histogram equalisation (NumPy only)")


if __name__ == "__main__":
    main()
