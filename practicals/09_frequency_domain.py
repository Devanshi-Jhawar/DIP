"""Practical 9 - Frequency-domain filtering: FFT, low-pass / high-pass filters, IFFT.

Pipeline:  f(x,y) --FFT--> F(u,v) --multiply by H(u,v)--> G(u,v) --IFFT--> g(x,y)
The zero frequency is moved to the centre with fftshift before filtering.
"""

import numpy as np

from utils import clip_u8, load_gray, show_grid, to_uint8

PRACTICAL = "09_frequency_domain"
D0 = 30  # cut-off frequency (distance from the centre of the spectrum, in samples)


def distance_from_centre(shape):
    rows, cols = shape
    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2
    U, V = np.meshgrid(u, v, indexing="ij")
    return np.sqrt(U ** 2 + V ** 2)


def ideal_lowpass(shape, d0):
    return (distance_from_centre(shape) <= d0).astype(np.float64)


def gaussian_lowpass(shape, d0):
    return np.exp(-distance_from_centre(shape) ** 2 / (2 * d0 ** 2))


def ideal_highpass(shape, d0):
    return 1.0 - ideal_lowpass(shape, d0)


def gaussian_highpass(shape, d0):
    return 1.0 - gaussian_lowpass(shape, d0)


def fft_spectrum(img):
    """a. 2-D FFT with the zero frequency centred. Returns (F, log-magnitude)."""
    F = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    return F, np.log1p(np.abs(F))


def filter_in_frequency(img, H):
    """b + c. Multiply the centred spectrum by H, then IFFT back to the spatial domain."""
    F = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    G = F * H
    g = np.fft.ifft2(np.fft.ifftshift(G))
    return np.real(g)


def main():
    gray = load_gray()
    shape = gray.shape

    # a. FFT ----------------------------------------------------------------
    F, log_mag = fft_spectrum(gray)
    dc = abs(F[shape[0] // 2, shape[1] // 2])
    print(f"9a. FFT of {shape[1]} x {shape[0]} image: DC term {dc:.0f} "
          f"equals sum of pixels {gray.sum():.0f}")
    show_grid([gray, to_uint8(log_mag)],
              ["Original", "9a. Log magnitude spectrum (centred)"],
              PRACTICAL, "9a_fft_spectrum.png")

    # b. Low-pass and high-pass filtering -----------------------------------
    lp_ideal = filter_in_frequency(gray, ideal_lowpass(shape, D0))
    lp_gauss = filter_in_frequency(gray, gaussian_lowpass(shape, D0))
    hp_ideal = filter_in_frequency(gray, ideal_highpass(shape, D0))
    hp_gauss = filter_in_frequency(gray, gaussian_highpass(shape, D0))
    print(f"9b. Filters with D0 = {D0}: ideal LP, Gaussian LP, ideal HP, Gaussian HP")
    show_grid([gray, clip_u8(lp_ideal), clip_u8(lp_gauss), to_uint8(hp_ideal), to_uint8(hp_gauss)],
              ["Original", "9b. Ideal low-pass", "9b. Gaussian low-pass",
               "9b. Ideal high-pass", "9b. Gaussian high-pass"],
              PRACTICAL, "9b_lowpass_highpass.png")

    # Effect of the cut-off frequency (Gaussian low-pass) -------------------
    cutoffs = [5, 15, 40]
    results = [clip_u8(filter_in_frequency(gray, gaussian_lowpass(shape, d))) for d in cutoffs]
    show_grid([gray] + results,
              ["Original"] + [f"9b. Gaussian LP, D0 = {d}" for d in cutoffs],
              PRACTICAL, "9b_cutoff_effect.png")

    # c. IFFT reconstruction ------------------------------------------------
    recon = filter_in_frequency(gray, np.ones(shape))  # identity filter: FFT then IFFT
    err = np.max(np.abs(recon - gray))
    print(f"9c. IFFT reconstruction with no filtering: max error {err:.2e} grey levels")
    show_grid([gray, clip_u8(recon)], ["Original", "9c. Reconstructed with IFFT"],
              PRACTICAL, "9c_ifft_reconstruction.png")


if __name__ == "__main__":
    main()
