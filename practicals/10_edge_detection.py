"""Practical 10 - Edge detection using different edge masks.

Gradient masks estimate the image derivative in x and y; the edge strength is
sqrt(Gx^2 + Gy^2). Roberts, Prewitt and Sobel use first derivatives; the Laplacian uses
second derivatives; Canny adds Gaussian smoothing, non-maximum suppression and hysteresis.
"""

import cv2
import numpy as np

from utils import load_gray, show_grid, to_uint8

PRACTICAL = "10_edge_detection"

# (horizontal mask, vertical mask)
GRADIENT_MASKS = {
    "Roberts": (np.array([[1, 0], [0, -1]], np.float64),
                np.array([[0, 1], [-1, 0]], np.float64)),
    "Prewitt": (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float64),
                np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float64)),
    "Sobel": (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64),
              np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float64)),
}
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float64)


def convolve(img, kernel):
    """Apply a mask (correlation, same as convolution for these masks) keeping the sign.

    2 x 2 masks are anchored at their top-left element so the output is not shifted.
    """
    anchor = (-1, -1) if kernel.shape[0] % 2 == 1 else (0, 0)
    return cv2.filter2D(img.astype(np.float64), cv2.CV_64F, kernel, anchor=anchor)


def main():
    gray = load_gray()
    results = {"Original": gray}

    for name, (kx, ky) in GRADIENT_MASKS.items():
        gx = convolve(gray, kx)
        gy = convolve(gray, ky)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        results[name] = to_uint8(magnitude)
        print(f"10. {name:8s} mask: max edge strength {magnitude.max():8.1f}")

    lap = np.abs(convolve(gray, LAPLACIAN))
    results["Laplacian"] = to_uint8(lap)
    print(f"10. Laplacian mask: max |response| {lap.max():8.1f}")

    canny = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.4), 50, 150)
    results["Canny"] = canny
    print(f"10. Canny (blur + thresholds 50/150): {np.count_nonzero(canny)} edge pixels")

    titles = ["Original", "10. Roberts", "10. Prewitt", "10. Sobel",
              "10. Laplacian", "10. Canny"]
    show_grid(list(results.values()), titles, PRACTICAL, "10_edge_masks.png", cols=3)


if __name__ == "__main__":
    main()
