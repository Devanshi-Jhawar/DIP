"""Practical 1 - Basic image operations with OpenCV.

1a Read and display an image          1e Image profile (row and column)
1b Resize an image                    1f Split a colour image into R, G, B planes
1c Colour -> grayscale                1g Create a colour image from R, G, B planes
1d Colour/gray -> black & white       1h Write 2-D data to an image file
"""

import os

import cv2
import numpy as np

from utils import load_color, out_dir, show_grid, to_uint8
import matplotlib.pyplot as plt

PRACTICAL = "01_basic_operations"


def main():
    # 1a. Read and display ---------------------------------------------------
    color = load_color()
    h, w = color.shape[:2]
    print(f"1a. Read image: {w} x {h} px, {color.shape[2]} channels, dtype {color.dtype}")
    show_grid([color], ["1a. Original colour image"], PRACTICAL, "1a_read_display.png", cols=1)

    # 1b. Resize -------------------------------------------------------------
    half = cv2.resize(color, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    fixed = cv2.resize(color, (200, 150), interpolation=cv2.INTER_AREA)  # (width, height)
    print(f"1b. Resized by 50%: {half.shape[1]} x {half.shape[0]} px; "
          f"fixed size: {fixed.shape[1]} x {fixed.shape[0]} px")
    show_grid([color, half, fixed],
              ["Original", "1b. 50% (fx = fy = 0.5)", "1b. Fixed 200 x 150"],
              PRACTICAL, "1b_resize.png")

    # 1c. Colour -> grayscale ------------------------------------------------
    b, g, r = cv2.split(color)  # OpenCV stores channels as B, G, R
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    manual = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)
    max_diff = np.max(np.abs(gray.astype(int) - manual.astype(int)))
    print(f"1c. Grayscale image {gray.shape}; max difference from "
          f"0.299R + 0.587G + 0.114B = {max_diff} (rounding only)")
    show_grid([color, gray], ["Colour", "1c. Grayscale"], PRACTICAL, "1c_grayscale.png")

    # 1d. Black & white (binary) with Otsu's automatic threshold -------------
    otsu_t, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"1d. Black & white image using Otsu threshold T = {otsu_t:.0f}")
    show_grid([gray, bw], ["Grayscale", f"1d. Black & white (T = {otsu_t:.0f})"],
              PRACTICAL, "1d_black_white.png")

    # 1e. Image profile (intensity along one row and one column) -------------
    row, col = h // 2, w // 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(gray[row, :], color="tab:blue")
    ax1.set_title(f"Horizontal profile (row {row})")
    ax1.set_xlabel("Column index")
    ax1.set_ylabel("Intensity")
    ax2.plot(gray[:, col], color="tab:orange")
    ax2.set_title(f"Vertical profile (column {col})")
    ax2.set_xlabel("Row index")
    ax2.set_ylabel("Intensity")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 255)
    fig.suptitle("1e. Image profile")
    fig.tight_layout()
    from utils import save_fig
    save_fig(fig, PRACTICAL, "1e_profile.png")
    print(f"1e. Row {row} intensity range: {gray[row].min()} to {gray[row].max()}")

    # 1f. Split into R, G, B planes ------------------------------------------
    show_grid([r, g, b], ["1f. Red plane", "1f. Green plane", "1f. Blue plane"],
              PRACTICAL, "1f_rgb_planes.png")
    print("1f. Split into R, G, B planes (shown as grayscale intensity maps)")

    # 1g. Create a colour image from separate planes -------------------------
    rebuilt = cv2.merge([b, g, r])  # merge expects B, G, R order
    zeros = np.zeros_like(r)
    red_only = cv2.merge([zeros, zeros, r])
    print(f"1g. Rebuilt image identical to original: {np.array_equal(rebuilt, color)}")
    show_grid([rebuilt, red_only], ["1g. Rebuilt from R, G, B", "1g. Red plane only"],
              PRACTICAL, "1g_merge_planes.png")

    # 1h. Write 2-D data to an image file -----------------------------------
    y, x = np.mgrid[0:256, 0:256]
    data_2d = 127.5 + 127.5 * np.sin(x / 12.0) * np.cos(y / 20.0)  # any 2-D array
    img_2d = to_uint8(data_2d)
    path = os.path.join(out_dir(PRACTICAL), "1h_data_as_image.png")
    cv2.imwrite(path, img_2d)
    readback = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print(f"1h. 2-D data {data_2d.shape} written to outputs/{PRACTICAL}/1h_data_as_image.png; "
          f"read back shape {readback.shape}")


if __name__ == "__main__":
    main()
