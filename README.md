# Digital Image Processing (DIP) Practicals

Python implementations of the DIP practicals using **OpenCV**, **NumPy**, **scikit-image**, **SciPy** and **Matplotlib**.

## Setup

```bash
pip install -r requirements.txt
```

## Running

Run everything (figures are saved to `practicals/outputs/`):

```bash
cd practicals
python run_all.py
```

Run a single practical, optionally with your own image:

```bash
python 05_histogram.py
python 05_histogram.py path/to/your_image.jpg
```

Set `DIP_SHOW=1` to open the figures in windows instead of only saving them.

If no image is given, sample images from scikit-image are used (astronaut for colour, camera and coins for gray).

## Practicals

| File | Practical | Contents |
|------|-----------|----------|
| `01_basic_operations.py` | 1 | Read/display, resize, colour to gray, black & white, profile, R/G/B split and merge, write 2-D data to an image |
| `02_point_processing.py` | 2 | Negative, flip, thresholding, contrast stretching (linear and piecewise) |
| `03_image_arithmetic.py` | 3 | Addition (saturated and wrapped), subtraction, mean value, brightness change by mean |
| `04_logical_operations.py` | 4 | AND, OR, intersection, XOR watermarking (embed and extract), NOT |
| `05_histogram.py` | 5 | Histogram and equalisation: OpenCV built-ins (5a) and NumPy-only implementation (5b) |
| `06_geometric_transforms.py` | 6 | Translation, scaling, rotation, shrinking, zooming (nearest vs cubic) |
| `07_noise_restoration.py` | 7 | Noise models (Gaussian, salt & pepper, speckle, Poisson), salt & pepper removal, Gaussian smoothing, median vs Wiener filter (PSNR) |
| `08_spatial_filtering.py` | 8 | 1-D and 2-D convolution (explicit loops, checked against NumPy/SciPy), 3x3 low-pass and high-pass masks |
| `09_frequency_domain.py` | 9 | FFT and log spectrum, ideal and Gaussian low/high-pass filters, IFFT reconstruction |
| `10_edge_detection.py` | 10 | Roberts, Prewitt, Sobel, Laplacian and Canny edge detection |
| `11_morphology.py` | 11 | Erosion and dilation (binary and grayscale), plus opening, closing and boundary |

Shared helpers (image loading, figure saving, PSNR) are in `practicals/utils.py`.

## Notes

- OpenCV loads colour images in **BGR** order; the code converts to RGB only when plotting.
- Results are reported as printed numbers (for example PSNR in dB, or pixel counts) and as saved figures in `practicals/outputs/<practical>/`.
- The lab sheet allowed Python or Scilab; this repository uses Python only.
