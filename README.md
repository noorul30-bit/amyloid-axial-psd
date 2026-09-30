# Amyloid Axial PSD

A Python tool for estimating the axial cross-beta repeat from selected cryo-EM 2D class averages using one-dimensional power spectral density (PSD) analysis.

The script is designed for MRC/MRCS stacks of selected 2D class averages, including those exported from cryoSPARC.

## Method

For each selected 2D class average, the script:

1. subtracts the mean image intensity;
2. applies a two-dimensional Hann window;
3. calculates the two-dimensional Fourier transform;
4. calculates the power spectrum as |FFT|²;
5. extracts a narrow Fourier-space profile along the fibril axis;
6. averages positive and negative spatial-frequency components;
7. normalizes and averages PSD profiles across selected classes;
8. identifies the dominant cross-beta peak; and
9. converts the peak spatial frequency to real-space spacing using:

`d = 1 / q_peak`

The calculated spacing is intended as an initial estimate of the axial cross-beta repeat for subsequent helical refinement.

## Installation

Python 3 is required.

Install the dependencies using:

```bash
python3 -m pip install numpy matplotlib mrcfile
```
## Usage

For vertically aligned fibrils:

`python3 axial_psd.py selected_classes.mrc`

For horizontally aligned fibrils:

`python3 axial_psd.py selected_classes.mrc --axis horizontal`

If the effective pixel size needs to be specified manually:

`python3 axial_psd.py selected_classes.mrc --pixel-size 1.25`

To display individual class PSD profiles:

`python3 axial_psd.py selected_classes.mrc --show-individuals`

## Output

The script generates:

- mean axial PSD data as CSV;
- individual class PSD profiles as CSV;
- an analysis summary as TXT;
- a publication-quality PDF plot; and
- a 600 dpi PNG plot.

## Notes

The detected PSD peak provides an initial estimate of the axial cross-beta repeat and should not automatically be treated as the final refined helical rise.

The default cross-beta peak search range is 0.18–0.23 Å⁻¹, which covers typical amyloid axial repeats of approximately 4.35–5.56 Å.

The effective pixel size stored in the MRC header is used by default. If the header value is missing or incorrect, the pixel size can be supplied manually using the `--pixel-size` option.

## Citation

If you use this software, please cite:

Huda, N. (2026). *Amyloid Axial PSD* (Version 1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23045948

Version 1.0.0 DOI: https://doi.org/10.5281/zenodo.23045948

All versions: https://doi.org/10.5281/zenodo.23045947
## License

License information will be added before the first public software release.
