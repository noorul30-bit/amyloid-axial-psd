#!/usr/bin/env python3

"""
axial_psd.py

Calculate a 1D axial power spectral density (PSD) profile from
cryoSPARC-selected 2D class averages stored in an MRC/MRCS file.

Intended use:
    Selected 2D classes
        -> mean subtraction
        -> 2D Hann window
        -> 2D FFT
        -> power spectrum |FFT|^2
        -> axial strip averaging
        -> average +q and -q directions
        -> normalize individual class PSDs
        -> average across classes
        -> identify cross-beta peak
        -> convert spatial frequency to real-space spacing

Example:
    python3 axial_psd.py selected_classes.mrc

For horizontally oriented fibrils:
    python3 axial_psd.py selected_classes.mrc --axis horizontal

If the MRC header has an incorrect/missing pixel size:
    python3 axial_psd.py selected_classes.mrc --pixel-size 1.25

Outputs:
    <input>_axial_PSD.csv
    <input>_individual_PSD.csv
    <input>_axial_PSD.pdf
    <input>_axial_PSD.png
    <input>_PSD_summary.txt
"""

import argparse
from pathlib import Path
import sys

import mrcfile
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# COMMAND-LINE OPTIONS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Calculate a 1D axial PSD profile from selected "
        "cryoSPARC 2D class averages."
    )
)

parser.add_argument(
    "input",
    help="Input MRC/MRCS file containing selected 2D class averages."
)

parser.add_argument(
    "--axis",
    choices=["vertical", "horizontal"],
    default="vertical",
    help=(
        "Orientation of the fibril axis in the 2D class averages. "
        "Default: vertical."
    )
)

parser.add_argument(
    "--pixel-size",
    type=float,
    default=None,
    help=(
        "Effective pixel size of the class averages in Angstrom/pixel. "
        "Default: read from MRC header."
    )
)

parser.add_argument(
    "--strip-half-width",
    type=int,
    default=4,
    help=(
        "Half-width, in Fourier pixels, of the axial strip that is "
        "averaged to generate the 1D PSD. Default: 4."
    )
)

parser.add_argument(
    "--search-min",
    type=float,
    default=0.18,
    help=(
        "Minimum spatial frequency in A^-1 used to search for "
        "the cross-beta peak. Default: 0.18."
    )
)

parser.add_argument(
    "--search-max",
    type=float,
    default=0.23,
    help=(
        "Maximum spatial frequency in A^-1 used to search for "
        "the cross-beta peak. Default: 0.23."
    )
)

parser.add_argument(
    "--normalize-min",
    type=float,
    default=0.05,
    help=(
        "Minimum spatial frequency used for PSD normalization. "
        "Default: 0.05 A^-1."
    )
)

parser.add_argument(
    "--normalize-max",
    type=float,
    default=0.30,
    help=(
        "Maximum spatial frequency used for PSD normalization. "
        "Default: 0.30 A^-1."
    )
)

parser.add_argument(
    "--plot-min",
    type=float,
    default=0.03,
    help="Minimum x-axis spatial frequency for plotting. Default: 0.03 A^-1."
)

parser.add_argument(
    "--plot-max",
    type=float,
    default=0.30,
    help="Maximum x-axis spatial frequency for plotting. Default: 0.30 A^-1."
)

parser.add_argument(
    "--output-dir",
    default=None,
    help=(
        "Directory for output files. Default: same directory "
        "as the input MRC."
    )
)

parser.add_argument(
    "--show-individuals",
    action="store_true",
    help="Plot individual class PSD profiles as faint background lines."
)

parser.add_argument(
    "--no-window",
    action="store_true",
    help="Disable the 2D Hann window. Not recommended for routine use."
)

args = parser.parse_args()


# ============================================================
# INPUT / OUTPUT PATHS
# ============================================================

input_path = Path(args.input)

if not input_path.exists():
    sys.exit(f"ERROR: Input file not found: {input_path}")

if args.output_dir is None:
    output_dir = input_path.parent
else:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

output_prefix = input_path.stem


# ============================================================
# READ MRC STACK
# ============================================================

with mrcfile.open(input_path, permissive=True) as mrc:
    data = np.asarray(mrc.data, dtype=np.float32)

    try:
        header_pixel_size = float(mrc.voxel_size.x)
    except Exception:
        header_pixel_size = 0.0


# If a single 2D image was supplied, convert to a 1-image stack
if data.ndim == 2:
    data = data[np.newaxis, :, :]

if data.ndim != 3:
    sys.exit(
        "ERROR: Expected a 2D image or 3D stack of 2D class averages. "
        f"Found data shape: {data.shape}"
    )

n_classes, ny, nx = data.shape


# ============================================================
# PIXEL SIZE
# ============================================================

if args.pixel_size is not None:
    pixel_size = args.pixel_size
else:
    pixel_size = header_pixel_size

if not np.isfinite(pixel_size) or pixel_size <= 0:
    sys.exit(
        "ERROR: Pixel size is missing or invalid in the MRC header.\n"
        "Supply it manually, for example:\n"
        "python3 axial_psd.py classes.mrc --pixel-size 1.25"
    )


# ============================================================
# BASIC SAMPLING INFORMATION
# ============================================================

if args.axis == "vertical":
    axis_length = ny
else:
    axis_length = nx

delta_q = 1.0 / (axis_length * pixel_size)
nyquist = 1.0 / (2.0 * pixel_size)

print()
print("==============================================")
print("Axial PSD analysis")
print("==============================================")
print(f"Input file        : {input_path.name}")
print(f"Number of classes : {n_classes}")
print(f"Image dimensions  : {nx} x {ny} pixels")
print(f"Pixel size        : {pixel_size:.6f} A/pixel")
print(f"Fibril axis       : {args.axis}")
print(f"Frequency sampling: {delta_q:.6f} A^-1/pixel")
print(f"Nyquist frequency : {nyquist:.4f} A^-1")
print(f"Nyquist resolution: {1.0 / nyquist:.2f} A")
print("==============================================")
print()


if args.search_max >= nyquist:
    print(
        "WARNING: The requested peak-search range extends to or beyond "
        "the Nyquist frequency."
    )


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_axial_psd(img):
    """
    Calculate one normalized 1D axial PSD profile from a single
    2D class average.

    Returns
    -------
    q : ndarray
        Positive spatial-frequency coordinates in A^-1.
    profile : ndarray
        Normalized linear PSD profile.
    """

    # Remove DC/background offset
    img = img.astype(np.float64)
    img = img - np.mean(img)

    # Apply 2D Hann window to reduce edge discontinuities
    if not args.no_window:
        wy = np.hanning(img.shape[0])
        wx = np.hanning(img.shape[1])
        window = np.outer(wy, wx)
        img = img * window

    # 2D Fourier transform, centered
    fft2 = np.fft.fftshift(np.fft.fft2(img))

    # Power spectrum
    psd2d = np.abs(fft2) ** 2

    cy = psd2d.shape[0] // 2
    cx = psd2d.shape[1] // 2

    hw = args.strip_half_width

    if args.axis == "vertical":

        # Fibril runs vertically in real space.
        # Average a narrow strip around qx = 0,
        # leaving intensity as a function of qy.
        strip = psd2d[:, cx - hw:cx + hw + 1]
        axial_profile = np.mean(strip, axis=1)

        freq = np.fft.fftshift(
            np.fft.fftfreq(psd2d.shape[0], d=pixel_size)
        )

    else:

        # Fibril runs horizontally in real space.
        # Average around qy = 0,
        # leaving intensity as a function of qx.
        strip = psd2d[cy - hw:cy + hw + 1, :]
        axial_profile = np.mean(strip, axis=0)

        freq = np.fft.fftshift(
            np.fft.fftfreq(psd2d.shape[1], d=pixel_size)
        )

    # Separate positive and negative spatial frequencies
    positive = freq > 0
    negative = freq < 0

    q_pos = freq[positive]
    p_pos = axial_profile[positive]

    q_neg = np.abs(freq[negative][::-1])
    p_neg = axial_profile[negative][::-1]

    # Ensure matching lengths
    n = min(len(p_pos), len(p_neg))

    q = q_pos[:n]

    # Average centrosymmetric +q and -q information
    profile = 0.5 * (
        p_pos[:n] +
        p_neg[:n]
    )

    # Normalize each class so one high-intensity class
    # does not dominate the final average.
    norm_region = (
        (q >= args.normalize_min) &
        (q <= min(args.normalize_max, nyquist))
    )

    if np.any(norm_region):
        baseline = np.median(profile[norm_region])
    else:
        baseline = np.median(profile)

    if np.isfinite(baseline) and baseline > 0:
        profile = profile / baseline

    return q, profile


# ============================================================
# CALCULATE PSD FOR ALL CLASSES
# ============================================================

profiles = []
q_reference = None

for i, img in enumerate(data):

    q, profile = calculate_axial_psd(img)

    if q_reference is None:
        q_reference = q
    else:
        if len(q) != len(q_reference):
            sys.exit(
                "ERROR: PSD profile lengths are inconsistent across classes."
            )

    profiles.append(profile)

profiles = np.asarray(profiles)
q = q_reference


# ============================================================
# AVERAGE PSD ACROSS SELECTED CLASSES
# ============================================================

mean_psd = np.mean(profiles, axis=0)
std_psd = np.std(profiles, axis=0, ddof=1) if n_classes > 1 else np.zeros_like(mean_psd)
sem_psd = std_psd / np.sqrt(n_classes) if n_classes > 1 else np.zeros_like(mean_psd)


# Log transform only for plotting
log_mean = np.log10(np.maximum(mean_psd, 1e-12))

log_lower = np.log10(
    np.maximum(mean_psd - std_psd, 1e-12)
)

log_upper = np.log10(
    np.maximum(mean_psd + std_psd, 1e-12)
)


# ============================================================
# IDENTIFY CROSS-BETA PEAK
# ============================================================

search_region = (
    (q >= args.search_min) &
    (q <= min(args.search_max, nyquist))
)

if not np.any(search_region):
    sys.exit(
        "ERROR: No Fourier samples fall inside the requested peak-search range."
    )

search_indices = np.where(search_region)[0]

local_index = np.argmax(mean_psd[search_region])
peak_index = search_indices[local_index]

peak_q = q[peak_index]
peak_spacing = 1.0 / peak_q


print("Cross-beta peak:")
print(f"  Spatial frequency : {peak_q:.6f} A^-1")
print(f"  Real-space spacing: {peak_spacing:.3f} A")
print()


# ============================================================
# SAVE SUMMARY PSD CSV
# ============================================================

summary_csv = output_dir / f"{output_prefix}_axial_PSD.csv"

summary_array = np.column_stack(
    [
        q,
        mean_psd,
        std_psd,
        sem_psd
    ]
)

np.savetxt(
    summary_csv,
    summary_array,
    delimiter=",",
    header=(
        "spatial_frequency_A^-1,"
        "mean_PSD,"
        "std_PSD,"
        "sem_PSD"
    ),
    comments=""
)


# ============================================================
# SAVE INDIVIDUAL CLASS PSDs
# ============================================================

individual_csv = output_dir / f"{output_prefix}_individual_PSD.csv"

individual_array = np.column_stack(
    [q] + [profiles[i] for i in range(n_classes)]
)

header_columns = ["spatial_frequency_A^-1"] + [
    f"class_{i + 1}_PSD"
    for i in range(n_classes)
]

np.savetxt(
    individual_csv,
    individual_array,
    delimiter=",",
    header=",".join(header_columns),
    comments=""
)


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

summary_txt = output_dir / f"{output_prefix}_PSD_summary.txt"

with open(summary_txt, "w") as f:

    f.write("Axial PSD analysis summary\n")
    f.write("==========================\n\n")

    f.write(f"Input file: {input_path.name}\n")
    f.write(f"Number of selected classes: {n_classes}\n")
    f.write(f"Image dimensions: {nx} x {ny} pixels\n")
    f.write(f"Effective pixel size: {pixel_size:.6f} A/pixel\n")
    f.write(f"Fibril orientation: {args.axis}\n")
    f.write(f"Strip half-width: {args.strip_half_width} Fourier pixels\n")
    f.write(f"Frequency sampling: {delta_q:.6f} A^-1\n")
    f.write(f"Nyquist frequency: {nyquist:.6f} A^-1\n")
    f.write(f"Nyquist resolution: {1.0 / nyquist:.3f} A\n\n")

    f.write(
        f"Peak search range: "
        f"{args.search_min:.4f} - "
        f"{args.search_max:.4f} A^-1\n"
    )

    f.write(f"Detected peak frequency: {peak_q:.6f} A^-1\n")
    f.write(f"Corresponding spacing: {peak_spacing:.3f} A\n")


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(6.5, 4.2))


# Optional individual class profiles
if args.show_individuals:
    for profile in profiles:
        ax.plot(
            q,
            np.log10(np.maximum(profile, 1e-12)),
            linewidth=0.7,
            alpha=0.20
        )


# Mean PSD
ax.plot(
    q,
    log_mean,
    linewidth=2.0,
    label="Mean PSD"
)


# ±1 SD envelope
ax.fill_between(
    q,
    log_lower,
    log_upper,
    alpha=0.15,
    linewidth=0
)


# Peak marker
ax.axvline(
    peak_q,
    linestyle="--",
    linewidth=1.2
)


# Peak annotation
ax.annotate(
    (
        f"{peak_q:.4f} Å$^{{-1}}$\n"
        f"{peak_spacing:.2f} Å"
    ),
    xy=(peak_q, log_mean[peak_index]),
    xytext=(
        peak_q + 0.02,
        log_mean[peak_index] + 0.6
    ),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=0.8
    ),
    fontsize=9
)


# Axis limits
plot_max = min(args.plot_max, nyquist)

ax.set_xlim(
    args.plot_min,
    plot_max
)


# Labels
ax.set_xlabel(
    r"Spatial frequency ($\AA^{-1}$)"
)

ax.set_ylabel(
    r"log$_{10}$(PSD)"
)


# Publication-style cleanup
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    direction="out",
    length=4,
    width=0.8
)

plt.tight_layout()


# ============================================================
# SAVE FIGURES
# ============================================================

pdf_file = output_dir / f"{output_prefix}_axial_PSD.pdf"
png_file = output_dir / f"{output_prefix}_axial_PSD.png"

plt.savefig(
    pdf_file,
    bbox_inches="tight"
)

plt.savefig(
    png_file,
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FINAL OUTPUT MESSAGE
# ============================================================

print("Files written:")
print(f"  {summary_csv}")
print(f"  {individual_csv}")
print(f"  {summary_txt}")
print(f"  {pdf_file}")
print(f"  {png_file}")
print()