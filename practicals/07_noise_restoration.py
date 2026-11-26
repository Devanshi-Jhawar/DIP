"""Practical 7 - Image noise models and restoration.

7a. Noise models (Gaussian, salt & pepper, speckle, Poisson) and the noisy baseline
7b. Remove salt & pepper noise (median filter, compared with mean filter)
7c. Minimise Gaussian noise (Gaussian smoothing)
7d. Median filter and Wiener filter compared on both noise types

Quality is reported as PSNR in dB against the clean image (higher is better).
"""

import cv2
import numpy as np
from scipy.signal import wiener
from skimage.util import random_noise

from utils import load_gray, psnr, show_grid

PRACTICAL = "07_noise_restoration"


def add_noise(img, mode, seed=0, **kwargs):
    """Add noise with skimage's noise models and return an 8-bit image."""
    noisy = random_noise(img, mode=mode, rng=seed, **kwargs)  # float in [0, 1]
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


def wiener_filter(img, size=5):
    """Adaptive Wiener filter (local mean and variance estimated in a size x size window)."""
    with np.errstate(divide="ignore", invalid="ignore"):  # flat regions give 0/0 (harmless)
        out = wiener(img.astype(np.float64), mysize=size)
    out = np.nan_to_num(out)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    clean = load_gray()

    # 7a. Noise models -------------------------------------------------------
    gauss = add_noise(clean, "gaussian", var=0.01)
    sp = add_noise(clean, "s&p", amount=0.05)
    speckle = add_noise(clean, "speckle", var=0.04)
    poisson = add_noise(clean, "poisson")
    print("7a. Noise models - PSNR of noisy image vs clean image:")
    for name, img in [("Gaussian", gauss), ("Salt & pepper", sp),
                      ("Speckle", speckle), ("Poisson", poisson)]:
        print(f"    {name:14s} {psnr(clean, img):6.2f} dB")
    show_grid([clean, gauss, sp, speckle, poisson],
              ["Original", "7a. Gaussian (var 0.01)", "7a. Salt & pepper (5%)",
               "7a. Speckle (var 0.04)", "7a. Poisson"],
              PRACTICAL, "7a_noise_models.png")

    # 7b. Remove salt & pepper noise ----------------------------------------
    med_sp = cv2.medianBlur(sp, 3)
    mean_sp = cv2.blur(sp, (3, 3))
    print(f"7b. Salt & pepper: noisy {psnr(clean, sp):.2f} dB | "
          f"3x3 median {psnr(clean, med_sp):.2f} dB | 3x3 mean {psnr(clean, mean_sp):.2f} dB")
    show_grid([sp, med_sp, mean_sp],
              ["Salt & pepper noisy", "7b. 3x3 median filter", "3x3 mean filter (comparison)"],
              PRACTICAL, "7b_remove_salt_pepper.png")

    # 7c. Minimise Gaussian noise -------------------------------------------
    gauss_smooth = cv2.GaussianBlur(gauss, (5, 5), 1.2)
    print(f"7c. Gaussian noise: noisy {psnr(clean, gauss):.2f} dB | "
          f"5x5 Gaussian smoothing {psnr(clean, gauss_smooth):.2f} dB")
    show_grid([gauss, gauss_smooth],
              ["Gaussian noisy", "7c. Gaussian smoothing (5x5, sigma 1.2)"],
              PRACTICAL, "7c_minimise_gaussian.png")

    # 7d. Median vs Wiener filter -------------------------------------------
    med_gauss = cv2.medianBlur(gauss, 5)
    wien_gauss = wiener_filter(gauss, 5)
    med_sp5 = cv2.medianBlur(sp, 5)
    wien_sp = wiener_filter(sp, 5)
    print("7d. Filter comparison (PSNR, dB):")
    print(f"    {'':22s}{'Gaussian':>10s}{'Salt&pepper':>14s}")
    print(f"    {'Noisy':22s}{psnr(clean, gauss):10.2f}{psnr(clean, sp):14.2f}")
    print(f"    {'Median 5x5':22s}{psnr(clean, med_gauss):10.2f}{psnr(clean, med_sp5):14.2f}")
    print(f"    {'Wiener 5x5':22s}{psnr(clean, wien_gauss):10.2f}{psnr(clean, wien_sp):14.2f}")
    show_grid([gauss, med_gauss, wien_gauss, sp, med_sp5, wien_sp],
              ["Gaussian noisy", "7d. Median 5x5", "7d. Wiener 5x5",
               "Salt & pepper noisy", "7d. Median 5x5", "7d. Wiener 5x5"],
              PRACTICAL, "7d_median_vs_wiener.png", cols=3)


if __name__ == "__main__":
    main()
