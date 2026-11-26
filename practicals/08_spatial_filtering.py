"""Practical 8 - Spatial filtering: 1-D and 2-D convolution, 3x3 low-pass and high-pass masks.

Convolution: y[n] = sum_k x[k] h[n - k]  (the kernel is flipped before sliding).
For the symmetric masks used here, convolution and correlation (cv2.filter2D) give
the same result.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

from utils import load_gray, save_fig, show_grid, to_uint8, clip_u8

PRACTICAL = "08_spatial_filtering"

LOW_PASS_3X3 = np.ones((3, 3)) / 9.0                       # averaging (smoothing)
HIGH_PASS_3X3 = np.array([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]], dtype=np.float64)  # sums to 0: keeps only detail


def conv1d_manual(x, h):
    """a. 1-D full convolution written with explicit loops."""
    y = np.zeros(len(x) + len(h) - 1)
    for n in range(len(y)):
        for k in range(len(x)):
            if 0 <= n - k < len(h):
                y[n] += x[k] * h[n - k]
    return y


def conv2d_manual(img, kernel):
    """a. 2-D 'same' convolution with zero padding, written with explicit loops."""
    kh, kw = kernel.shape
    flipped = kernel[::-1, ::-1]
    padded = np.pad(img.astype(np.float64), ((kh // 2, kh // 2), (kw // 2, kw // 2)))
    out = np.zeros(img.shape, dtype=np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * flipped)
    return out


def plot_1d(x, h, y, filename):
    fig, axes = plt.subplots(3, 1, figsize=(8, 7))
    axes[0].stem(x, basefmt=" ")
    axes[0].set_title("Input signal x[n] (step + noise)")
    axes[1].stem(h, basefmt=" ")
    axes[1].set_title("Kernel h[n] (5-point average)")
    axes[2].stem(y, basefmt=" ")
    axes[2].set_title("Output y[n] = x * h (full convolution)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, PRACTICAL, filename)


def main():
    # a. 1-D convolution ----------------------------------------------------
    rng = np.random.default_rng(0)
    x = np.r_[np.zeros(20), np.ones(20)] + rng.normal(0, 0.2, 40)
    h = np.ones(5) / 5.0
    y_manual = conv1d_manual(x, h)
    print(f"8a. 1-D convolution: manual == np.convolve: {np.allclose(y_manual, np.convolve(x, h))}; "
          f"output length {len(y_manual)} = {len(x)} + {len(h)} - 1")
    plot_1d(x, h, y_manual, "8a_1d_convolution.png")

    # a. 2-D convolution (explicit loops on a small patch, checked against SciPy) ---
    gray = load_gray()
    patch = gray[100:140, 100:140].astype(np.float64)
    manual2d = conv2d_manual(patch, LOW_PASS_3X3)
    reference2d = convolve2d(patch, LOW_PASS_3X3, mode="same", boundary="fill")
    print(f"8a. 2-D convolution: manual == scipy.signal.convolve2d: "
          f"{np.allclose(manual2d, reference2d)}")

    # b. 3x3 low-pass and high-pass masks ------------------------------------
    low = clip_u8(cv2.filter2D(gray.astype(np.float64), -1, LOW_PASS_3X3))
    high_raw = cv2.filter2D(gray.astype(np.float64), -1, HIGH_PASS_3X3)  # can be negative
    high_view = to_uint8(np.abs(high_raw))
    sharpened = clip_u8(gray + high_raw)
    print(f"8b. Low-pass (mean of 3x3): std {gray.std():.1f} -> {low.std():.1f} (smoother)")
    print(f"8b. High-pass (3x3 Laplacian-style): mean response {high_raw.mean():.2f}, "
          f"max |response| {np.abs(high_raw).max():.0f}")
    show_grid([gray, low, high_view, sharpened],
              ["Original", "8b. Low-pass 3x3 (smoothing)", "8b. High-pass 3x3 (|response|)",
               "8b. Sharpened (original + high-pass)"],
              PRACTICAL, "8b_lowpass_highpass.png")

    # Noise removal with the low-pass mask ----------------------------------
    noisy = clip_u8(gray + rng.normal(0, 15, gray.shape))
    smoothed = clip_u8(cv2.filter2D(noisy.astype(np.float64), -1, LOW_PASS_3X3))
    print(f"8b. Noise removal: PSNR noisy {_psnr(gray, noisy):.2f} dB -> "
          f"3x3 low-pass {_psnr(gray, smoothed):.2f} dB")
    show_grid([gray, noisy, smoothed],
              ["Original", "Gaussian noise (sigma 15)", "8b. After 3x3 low-pass"],
              PRACTICAL, "8b_noise_removal.png")


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse)


if __name__ == "__main__":
    main()
