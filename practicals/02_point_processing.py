"""Practical 2 - Point processing: negative, flip, thresholding, contrast stretching.

Point processing changes each pixel using only its own value: s = T(r).
"""

import cv2
import numpy as np

from utils import load_color, load_gray, show_grid

PRACTICAL = "02_point_processing"


def negative(img):
    """a. Negative: s = 255 - r. Works for gray and colour images."""
    return 255 - img


def flip(img):
    """b. Flip: mirror about the vertical axis, horizontal axis, or both."""
    return {
        "horizontal": cv2.flip(img, 1),
        "vertical": cv2.flip(img, 0),
        "both": cv2.flip(img, -1),
    }


def threshold(gray, t):
    """c. Thresholding: s = 255 if r > t, else 0."""
    return np.where(gray > t, 255, 0).astype(np.uint8)


def contrast_stretch(gray):
    """d. Linear contrast stretching: s = (r - rmin) * 255 / (rmax - rmin)."""
    r = gray.astype(np.float64)
    rmin, rmax = r.min(), r.max()
    return ((r - rmin) * 255.0 / (rmax - rmin)).astype(np.uint8)


def piecewise_stretch(gray, r1, s1, r2, s2):
    """d. Piecewise-linear stretch through the control points (r1, s1) and (r2, s2).

    Requires 0 < r1 < r2 < 255.
    """
    r = gray.astype(np.float64)
    s = np.select(
        [r < r1, r <= r2],
        [s1 / r1 * r, (s2 - s1) / (r2 - r1) * (r - r1) + s1],
        default=(255 - s2) / (255 - r2) * (r - r2) + s2,
    )
    return np.clip(s, 0, 255).astype(np.uint8)


def main():
    gray = load_gray()
    color = load_color()

    # a. Negative -----------------------------------------------------------
    neg_gray = negative(gray)
    neg_color = negative(color)
    print(f"2a. Negative verified against cv2.bitwise_not: "
          f"{np.array_equal(neg_gray, cv2.bitwise_not(gray))}")
    show_grid([gray, neg_gray, color, neg_color],
              ["Gray original", "2a. Negative (gray)", "Colour original", "2a. Negative (colour)"],
              PRACTICAL, "2a_negative.png")

    # b. Flip ---------------------------------------------------------------
    flips = flip(color)
    print("2b. Flipped horizontally, vertically and both ways")
    show_grid([color, flips["horizontal"], flips["vertical"], flips["both"]],
              ["Original", "2b. Horizontal flip", "2b. Vertical flip", "2b. Flip both"],
              PRACTICAL, "2b_flip.png")

    # c. Thresholding -------------------------------------------------------
    t_values = [60, 127, 200]
    thresholded = [threshold(gray, t) for t in t_values]
    same_as_cv = np.array_equal(thresholded[1],
                                cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1])
    print(f"2c. Threshold at T=127 matches cv2.threshold: {same_as_cv}")
    show_grid([gray] + thresholded,
              ["Gray original"] + [f"2c. T = {t}" for t in t_values],
              PRACTICAL, "2c_threshold.png")

    # d. Contrast stretching -----------------------------------------------
    low = (gray * 0.3 + 80).astype(np.uint8)  # deliberately low-contrast image
    stretched = contrast_stretch(low)
    r1, r2 = np.percentile(low, [5, 95])
    piece = piecewise_stretch(low, r1, 0, r2, 255)
    print(f"2d. Low-contrast range {low.min()}-{low.max()} -> linear stretch "
          f"{stretched.min()}-{stretched.max()}; piecewise (r1={r1:.0f}, r2={r2:.0f}) "
          f"{piece.min()}-{piece.max()}")
    show_grid([low, stretched, piece],
              ["Low-contrast input", "2d. Linear min-max stretch",
               f"2d. Piecewise stretch ({r1:.0f}, {r2:.0f})"],
              PRACTICAL, "2d_contrast_stretch.png")


if __name__ == "__main__":
    main()
