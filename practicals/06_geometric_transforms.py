"""Practical 6 - Geometric transformations: translation, scaling, rotation, shrinking, zooming."""

import cv2
import numpy as np

from utils import load_color, show_grid

PRACTICAL = "06_geometric_transforms"


def translate(img, tx, ty):
    """a. Translation: x' = x + tx, y' = y + ty."""
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def scale(img, factor):
    """b. Scaling about the image centre (factor > 1 enlarges; canvas size is kept)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, factor)  # zero rotation, uniform scale
    return cv2.warpAffine(img, M, (w, h))


def rotate(img, angle):
    """c. Rotation about the centre, counter-clockwise by `angle` degrees."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def shrink(img, factor):
    """d. Shrinking (factor < 1), using area averaging to avoid aliasing."""
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


def zoom(img, factor, method=cv2.INTER_CUBIC):
    """e. Zooming (factor > 1) with the chosen interpolation method."""
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=method)


def main():
    img = load_color()
    h, w = img.shape[:2]

    moved = translate(img, 80, 40)
    print("6a. Translation by (tx=80, ty=40) px")
    scaled = scale(img, 1.4)
    print("6b. Scaling by 1.4 about the centre (canvas kept, edges cropped)")
    rotated = rotate(img, 30)
    print("6c. Rotation by 30 degrees about the centre")
    small = shrink(img, 0.5)
    print(f"6d. Shrinking by 0.5: {w} x {h} -> {small.shape[1]} x {small.shape[0]} px")
    show_grid([img, moved, scaled, rotated, small],
              ["Original", "6a. Translation (80, 40)", "6b. Scaling x1.4",
               "6c. Rotation 30 deg", "6d. Shrink x0.5"],
              PRACTICAL, "6abcd_geometric.png")

    # e. Zooming: compare nearest-neighbour and cubic on a small region ------
    crop = img[h // 3: h // 3 + 100, w // 3: w // 3 + 100]
    zoom_nn = zoom(crop, 4, cv2.INTER_NEAREST)
    zoom_cubic = zoom(crop, 4, cv2.INTER_CUBIC)
    zoomed_full = zoom(img, 2)
    print(f"6e. Zoom x4 of a 100 x 100 crop -> {zoom_cubic.shape[1]} x {zoom_cubic.shape[0]} px; "
          f"full image x2 -> {zoomed_full.shape[1]} x {zoomed_full.shape[0]} px")
    show_grid([crop, zoom_nn, zoom_cubic],
              ["Crop (100 x 100)", "6e. Zoom x4, nearest neighbour", "6e. Zoom x4, cubic"],
              PRACTICAL, "6e_zoom.png")


if __name__ == "__main__":
    main()
