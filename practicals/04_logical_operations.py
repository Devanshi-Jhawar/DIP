"""Practical 4 - Logical operations: AND, OR, intersection, XOR watermarking, NOT."""

import cv2
import numpy as np

from utils import load_pair_gray, show_grid

PRACTICAL = "04_logical_operations"


def make_test_masks(size=256):
    """Two binary test images: a filled circle and a filled rectangle (0 or 255)."""
    circle = np.zeros((size, size), np.uint8)
    cv2.circle(circle, (size // 2 - 30, size // 2), 80, 255, -1)
    rect = np.zeros((size, size), np.uint8)
    cv2.rectangle(rect, (size // 2 - 20, size // 4), (size - 30, size - 60), 255, -1)
    return circle, rect


def watermark(host, logo):
    """d. Embed a binary logo into the host image with XOR (in the least significant bit).

    Returns (watermarked_image, extracted_logo). Extraction needs the original host.
    """
    logo_bits = (logo > 0).astype(np.uint8)              # 1 where the logo is white
    watermarked = np.bitwise_xor(host, logo_bits)        # flips the LSB only where logo = 1
    extracted = np.bitwise_xor(watermarked, host) * 255  # XOR again to recover the logo
    return watermarked, extracted.astype(np.uint8)


def main():
    # a, b. AND and OR on two binary test images ----------------------------
    circle, rect = make_test_masks()
    and_img = cv2.bitwise_and(circle, rect)
    or_img = cv2.bitwise_or(circle, rect)
    print(f"4a. AND: {np.count_nonzero(and_img)} white px; "
          f"4b. OR: {np.count_nonzero(or_img)} white px")
    show_grid([circle, rect, and_img, or_img],
              ["Image 1 (circle)", "Image 2 (rectangle)", "4a. AND", "4b. OR"],
              PRACTICAL, "4ab_and_or_masks.png")

    # a, b. The same operations on two real grayscale images ----------------
    a, b = load_pair_gray()
    gray_and = cv2.bitwise_and(a, b)
    gray_or = cv2.bitwise_or(a, b)
    show_grid([a, b, gray_and, gray_or],
              ["Image A", "Image B", "4a. A AND B", "4b. A OR B"],
              PRACTICAL, "4ab_and_or_images.png")

    # c. Intersection: pixels that are foreground in BOTH binary images -------
    _, bw_a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bw_b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    intersection = cv2.bitwise_and(bw_a, bw_b)
    pct = 100.0 * np.count_nonzero(intersection) / intersection.size
    print(f"4c. Intersection of the two binary images covers {pct:.1f}% of the frame")
    show_grid([bw_a, bw_b, intersection],
              ["Binary A", "Binary B", "4c. Intersection (AND)"],
              PRACTICAL, "4c_intersection.png")

    # d. Watermarking with XOR -------------------------------------------------
    host = a
    logo = np.zeros_like(host)
    cv2.putText(logo, "DIP", (110, 300), cv2.FONT_HERSHEY_SIMPLEX, 3.5, 255, 12)
    watermarked, extracted = watermark(host, logo)
    recovered = np.array_equal(extracted > 0, logo > 0)
    print(f"4d. Watermark embedded (PSNR host vs watermarked = "
          f"{10 * np.log10(255.0 ** 2 / np.mean((host.astype(float) - watermarked) ** 2)):.1f} dB); "
          f"logo recovered exactly: {recovered}")
    show_grid([host, logo, watermarked, extracted],
              ["Host image", "Watermark logo", "4d. Watermarked (XOR)", "4d. Extracted logo"],
              PRACTICAL, "4d_watermark_xor.png")

    # e. NOT (negative) --------------------------------------------------------
    not_img = cv2.bitwise_not(a)
    print(f"4e. NOT verified equal to 255 - A: {np.array_equal(not_img, 255 - a)}")
    show_grid([a, not_img, cv2.bitwise_not(circle)],
              ["Original", "4e. NOT (negative)", "4e. NOT of circle mask"],
              PRACTICAL, "4e_not.png")


if __name__ == "__main__":
    main()
