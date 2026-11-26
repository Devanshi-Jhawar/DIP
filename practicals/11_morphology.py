"""Practical 11 - Morphological operations: erosion and dilation (plus opening, closing, boundary).

Binary images: foreground = white (255). Erosion shrinks the foreground,
dilation grows it. Opening = erosion then dilation; closing = dilation then erosion.
"""

import cv2
import numpy as np

from utils import load_gray, show_grid

PRACTICAL = "11_morphology"


def binarize(gray):
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def main():
    gray = load_gray("coins")  # coins gives clear, separated blobs
    bw = binarize(gray)
    white = np.count_nonzero(bw)
    print(f"Binary image: {white} foreground px ({100.0 * white / bw.size:.1f}%)")

    rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ellipse5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    eroded3 = cv2.erode(bw, rect3)
    dilated3 = cv2.dilate(bw, rect3)
    eroded5 = cv2.erode(bw, ellipse5)
    dilated5 = cv2.dilate(bw, ellipse5)
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ellipse5)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ellipse5)
    boundary = cv2.subtract(dilated3, eroded3)

    print(f"Erosion 3x3 (square):    {np.count_nonzero(eroded3)} px  (shrinks)")
    print(f"Dilation 3x3 (square):   {np.count_nonzero(dilated3)} px  (grows)")
    print(f"Erosion 5x5 (ellipse):   {np.count_nonzero(eroded5)} px")
    print(f"Dilation 5x5 (ellipse):  {np.count_nonzero(dilated5)} px")
    print(f"Ordering check: erosion <= original <= dilation: "
          f"{bool(np.all(eroded3 <= bw) and np.all(bw <= dilated3))}")
    print(f"Opening: {np.count_nonzero(opened)} px; closing: {np.count_nonzero(closed)} px")

    show_grid(
        [bw, eroded3, dilated3, eroded5, dilated5, opened, closed, boundary],
        ["Binary (Otsu)", "11. Erosion 3x3", "11. Dilation 3x3",
         "11. Erosion 5x5 ellipse", "11. Dilation 5x5 ellipse",
         "Opening (5x5 ellipse)", "Closing (5x5 ellipse)", "Boundary = dilation - erosion"],
        PRACTICAL, "11_erosion_dilation.png", cols=4,
    )

    # Grayscale erosion (local minimum) and dilation (local maximum) --------
    g_eroded = cv2.erode(gray, rect3)
    g_dilated = cv2.dilate(gray, rect3)
    show_grid([gray, g_eroded, g_dilated],
              ["Gray original", "11. Grayscale erosion 3x3 (min)",
               "11. Grayscale dilation 3x3 (max)"],
              PRACTICAL, "11_grayscale_erosion_dilation.png")


if __name__ == "__main__":
    main()
