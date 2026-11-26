"""Practical 3 - Image arithmetic: addition, subtraction, mean value, brightness change."""

import cv2
import numpy as np

from utils import load_color, load_pair_gray, show_grid

PRACTICAL = "03_image_arithmetic"


def change_brightness(img, target_mean):
    """d. Shift every pixel by a constant so the image mean becomes target_mean."""
    shift = target_mean - img.mean()
    return np.clip(img.astype(np.float64) + shift, 0, 255).astype(np.uint8)


def main():
    a, b = load_pair_gray()
    print(f"Images A and B: {a.shape[1]} x {a.shape[0]} px grayscale")

    # a. Addition -----------------------------------------------------------
    add_saturated = cv2.add(a, b)                       # clipped at 255
    add_wrapped = a + b                                 # NumPy wraps around (mod 256)
    add_blend = cv2.addWeighted(a, 0.5, b, 0.5, 0)      # (A + B) / 2
    print(f"3a. cv2.add saturates: max = {add_saturated.max()}; "
          f"NumPy + wraps: min value = {add_wrapped.min()} (overflow visible)")
    show_grid([a, b, add_saturated, add_wrapped, add_blend],
              ["Image A", "Image B", "3a. A + B (cv2.add, saturated)",
               "3a. A + B (NumPy, wraps)", "3a. (A + B) / 2"],
              PRACTICAL, "3a_addition.png")

    # b. Subtraction --------------------------------------------------------
    sub_ab = cv2.subtract(a, b)     # negative results become 0
    sub_ba = cv2.subtract(b, a)
    abs_diff = cv2.absdiff(a, b)    # |A - B|
    print(f"3b. A - B: non-zero pixels = {np.count_nonzero(sub_ab)}; "
          f"|A - B| max = {abs_diff.max()}")
    show_grid([sub_ab, sub_ba, abs_diff],
              ["3b. A - B (clipped at 0)", "3b. B - A (clipped at 0)", "3b. |A - B|"],
              PRACTICAL, "3b_subtraction.png")

    # c. Mean value ---------------------------------------------------------
    color = load_color()
    print(f"3c. Mean of gray image A: {a.mean():.2f} (cv2.mean: {cv2.mean(a)[0]:.2f})")
    b_mean, g_mean, r_mean = cv2.mean(color)[:3]
    print(f"3c. Mean of colour image per channel: R = {r_mean:.2f}, "
          f"G = {g_mean:.2f}, B = {b_mean:.2f}")

    # d. Brightness by changing the mean -----------------------------------
    m = a.mean()
    targets = [max(m - 60, 0), m, min(m + 60, 255)]
    variants = [change_brightness(a, t) for t in targets]
    for t, v in zip(targets, variants):
        print(f"3d. Target mean {t:6.1f} -> actual mean {v.mean():6.2f}")
    show_grid([a] + variants,
              ["Original"] + [f"3d. Mean = {t:.0f}" for t in targets],
              PRACTICAL, "3d_brightness.png")


if __name__ == "__main__":
    main()
