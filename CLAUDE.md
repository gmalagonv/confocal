# Confocal Imaging Analysis Project

## What this project does
Analysis of Leica confocal microscopy data (`.lif` files) from *Drosophila* brain samples.
The main workflow is:
1. Load `.lif` files → extract Z-stacks per series and channel → save as `.tif`
2. Apply ROI masks (created externally, e.g. in Fiji) to extract mean signal intensities per compartment
3. Compare signal ratios across conditions and brain regions (Mushroom Body subregions MBa, Antennal Lobe AL, etc.)
4. Optionally deconvolve images before signal extraction

## Environment
Always use the `leica-env` conda environment:
```
/Users/gerard/miniforge3/envs/leica-env/bin/python
```
Key packages: `aicsimageio`, `tifffile`, `scipy`, `scikit-image`, `numpy`, `matplotlib`.

On Linux (server): same environment name, activated normally via `conda activate leica-env`.

## Data layout
```
~/data/confocal/<date>_<user>/
    Project.lif               # raw Leica file
    series_<i>/
        <date>_s<i>_ch<c>.tif          # full ZYX stack per channel
        channel_<c>/
            <date>_s<i>_ch<c>_f<z>.tif  # individual Z-frames (split_frames mode)
        masks/
            msk_<region>.tif  # integer-labeled ROI masks (0=background, 1..N=ROIs)
        projections/
            <date>_s<i>_ch<c>_<region>.tif
        channel_<c>/
            s<i>_ch<c>_deconv_iter_<n>.tif  # deconvolved output
```

## Main source file
[src/data_processing.py](src/data_processing.py) — all functions live here.

### Key functions
| Function | Purpose |
|---|---|
| `loader2(date, user, split_frames, server)` | Load `.lif` → save `.tif` stacks with embedded ImageJ voxel metadata |
| `signal_extractor(mask_path, file_path, ...)` | Extract mean intensity per ROI per Z-frame |
| `signals(date, user, series_n, masks_suffixes, ...)` | Batch signal extraction for multiple masks/channels |
| `parse_lif_psf_params(lif_path, scene)` | Read NA, voxel sizes, emission wavelengths from `.lif` metadata |
| `deconvolve(stack, lif_path, channel, scene, num_iter, emission_nm)` | Richardson-Lucy deconvolution; returns `(result, sigma_xy_px)` |
| `plot_bars_with_sem3(groups, labels, ...)` | Bar plot with SEM and overlaid data points |
| `raw_values(series_n)` | Raw mean intensities for 2025_11_29 dataset (hardcoded paths) |
| `basic_ratios(series_n)` | AL-normalized ratios for 2025_11_29 dataset (hardcoded paths) |

#### Recent function changes (2026-06-03)
- **`loader2`**: TIFFs now saved with ImageJ-compatible resolution metadata (`resolution=(1/vxy, 1/vxy)`, `spacing=vz`). Previously saved without spatial metadata.
- **`deconvolve`**: Added `emission_nm` parameter (float, nm) to override metadata-derived wavelength — useful when detection bands are wide and the midpoint is far from the true fluorophore peak. Now returns a **tuple** `(result, sigma_xy_px)` instead of just `result`.
- **`parse_lif_psf_params`**: Now uses `DyeName` field from `.lif` metadata to look up the true fluorophore emission peak (via a hardcoded `_dye_emission` dict) instead of always using the detection-band midpoint. Falls back to midpoint when `DyeName` is absent or unrecognised. Also reworked to parse per `ATLConfocalSettingDefinition` block (one per acquisition sequence), which correctly handles sequential acquisitions where the same detector channel number is reused for different dyes. Sequences with >2 simultaneous lasers are skipped (live-view/alignment scans). Sequences are sorted by minimum active laser wavelength before flattening into the channel list.

## Deconvolution
- Algorithm: Richardson-Lucy (`skimage.restoration.richardson_lucy`)
- PSF: Gaussian, parameters derived from `.lif` metadata
- PSF formulas: `σ_xy = 0.21 × λ / NA`, `σ_z = 0.66 × λ × n / NA²`
- **Nyquist-based mode selection** (automatic): if `σ_z_px ≥ 2` → 3D deconvolution; if `σ_z_px < 2` (Z undersampled) → 2D Richardson-Lucy applied per frame using XY PSF only. Nyquist criterion: pixel ≤ σ/2, i.e. σ/pixel ≥ 2.
- Hot pixel removal: selective median filter (only replaces pixels > 5× std above local median)
- Output: `float32`, saved as ImageJ-compatible TIFF with voxel size metadata
- Accepts both full ZYX stacks and single 2D frames
- `num_iter=15` default; use 5–10 for a first check, 10–30 typical range
- **Return value**: `(result, sigma_xy_px)` — the deconvolved array and the XY PSF sigma in pixels (useful for quality checks)

### Quality check used in notebooks
Re-blur the deconvolved result with `gaussian_filter(result[z], sigma=sigma_xy_px)` and compute the residual vs the original frame. Flag frames where the background noise ratio (`result[bg].std() / original[bg].std()`) exceeds 1.5× — those frames may have noise amplification. The background mask is defined as pixels below the 10th percentile of the original frame.

### Known microscope parameters (2025-11-29 dataset)
- Objective: HC PL APO CS2 40x/1.30 OIL (DMI8-CS)
- NA = 1.3, n = 1.518 (oil)
- Voxel size: Z = 1.0 µm, XY = 0.71 µm
- Channels: Ch0 ~529 nm (BRP), Ch1 ~659 nm (mito), Ch2 ~775 nm (HSP)
- Images are coarsely sampled (~7× below Nyquist in XY) → σ_xy_px ≈ 0.12, deconvolution effect is subtle; σ_z_px ≈ 0.44 → 2D-per-frame mode will be selected automatically

### Known microscope parameters (2026-05-26 dataset)
- Objective: HC PL APO CS2 40x/1.40 OIL (NA = 1.4, different from 2025-11-29)
- NA = 1.4, n = 1.518 (oil)
- Channels: Ch0 ~529 nm (BRP), Ch1 ~659 nm (mito), Ch2 ~796.5 nm (HSP)
- 3 series: series_0 and series_1 = 110 Z-frames, series_2 = 18 Z-frames
- Scene 0 voxel size: Z = 0.534 µm, XY = 0.361 µm
- Scene 1 voxel size: Z = 0.292 µm, XY = 0.023 µm (fine XY zoom)
- Scene 2 voxel size: Z = 0.661 µm, XY = 0.023 µm
- All scenes → σ_z_px < 2 → 2D-per-frame deconvolution is selected automatically

## Analysis notebooks
- `analysis_notebooks/` — one notebook per imaging session date (format `YY_MM_DD.ipynb`)
- Notebooks load functions from `src/data_processing.py`

### `26_05_26.ipynb` — current active notebook (as of 2026-06-03)
This notebook has two main sections:

**1. Deconvolution testing (2026_05_26_Gerardo data)**
Runs `deconvolve()` across scenes, channels, and iteration counts (3 and 4 iterations tested last). Computes residual reprojection quality check per frame and prints fraction of frames with potential noise amplification.

**2. Signal analysis (2025_11_29_Gerardo data)**
Compares **5X vs 1X training** conditions across Mushroom Body alpha lobe subcompartments (α1, α2, α3):
- **5X brains**: series 3–7 (5 brains)
- **1X brains**: series 8–11 (4 brains)
- Series map:
  - series_0–2: brains 1–3, Antennal Lobe (ALL)
  - series_3–7: brains 3,2,1,4,5 — Mushroom Body (5X training)
  - series_8–11: brains 6–9 — Mushroom Body (1X training)
- Normalization: BRP signal in the Antennal Lobe (Ch0/AL) used as denominator (left/right hemisphere separately)
- Measurements: BRP (Ch0), mito (Ch1), HSP70 (Ch2) in α1/α2/α3 ROIs of MBa mask
- Helper functions `basic_ratios(series_n)` and `raw_values(series_n)` are defined in `data_processing.py` — they use hardcoded paths to `2025_11_29_Gerardo` data and the mask/projection file structure
- Frames used per brain for AL and MBa masks are noted in the notebook markdown cell

## Channels in 2025-11-29 dataset
| Channel | Marker | Detection band |
|---|---|---|
| Ch0 | BRP (active zones) | 501–557 nm |
| Ch1 | Mito (mitochondria) | 557–761 nm |
| Ch2 | HSP (heat shock protein) | 761–789 nm |

## Colocalization analysis

Target dataset: **2026_05_26_Gerardo** (no hand-drawn ROIs available — masks are generated by thresholding).
Primary channel pair: **Ch1 (Mito) ↔ Ch2 (HSP70)**. Other pairs (Ch0↔Ch1, Ch0↔Ch2) may follow the same workflow.

### Step 1 — Generate a foreground mask via thresholding
No Fiji-drawn ROI masks exist for this dataset. Use Otsu thresholding on each channel to define foreground, then take the union as the analysis region:

```python
from skimage.filters import threshold_otsu

thresh1 = threshold_otsu(ch1)
thresh2 = threshold_otsu(ch2)
mask = (ch1 > thresh1) | (ch2 > thresh2)  # union: anywhere either signal is present
```

Alternative: use only one channel's mask as reference (e.g., Mito mask to ask "within mitochondria, how much HSP70 is there?"). Choice depends on the biological question.

### Step 2 — Choose Z mode

**Frame-by-frame**
Compute the colocalization metric independently for each Z frame, yielding a per-depth profile.
- Pro: reveals whether colocalization varies along the Z axis (e.g., if structures are stratified by depth).
- Con: edge frames often have little signal → noisy estimates; 110 frames per scene means many low-signal slices at the top/bottom of the stack.

**Z-projection (sum)**
Sum all frames along Z first, then compute one colocalization value per stack. Use sum (not max) projection to preserve intensity ratios between channels.
- Pro: integrates all signal, much more robust single estimate per brain/series.
- Con: loses depth information; sum projection can blur structures that are present at only a few Z levels.

Recommended starting point: **sum projection**. Add frame-by-frame as a secondary check to see if there is depth-dependent structure worth investigating.

### Step 3 — Metric options

#### Option A — Pearson R
```python
r = np.corrcoef(ch1[mask].ravel(), ch2[mask].ravel())[0, 1]
```
- **Pro:** No threshold needed beyond the foreground mask; measures intensity co-variation (if Mito signal is high in a pixel, is HSP70 also high?); single intuitive number (−1 to 1).
- **Con:** Sensitive to background — pixels where both channels are near zero artificially inflate R. Background-subtract or apply a strict mask first. Also sensitive to outlier/hot pixels (deconvolution helps but does not fully eliminate them).

#### Option B — Manders coefficients (M1, M2)
```python
# M1: fraction of Ch1 (Mito) signal that overlaps with Ch2 (HSP70) foreground
# M2: fraction of Ch2 (HSP70) signal that overlaps with Ch1 (Mito) foreground
ch2_mask = ch2 > threshold_otsu(ch2)
ch1_mask = ch1 > threshold_otsu(ch1)

M1 = ch1[ch2_mask].sum() / ch1[mask].sum()  # mito signal in HSP70-positive regions
M2 = ch2[ch1_mask].sum() / ch2[mask].sum()  # HSP70 signal in mito-positive regions
```
- **Pro:** Threshold-based overlap — asks "what fraction of the mito signal is in HSP70-positive territory?" Robust to intensity differences between channels. M1 ≠ M2 is informative: if HSP70 is broadly expressed but Mito is sparse, M1 can be high while M2 is low.
- **Con:** Threshold-dependent — Otsu is automatic but still a choice; results can shift if the histogram is not clearly bimodal. Less informative about intensity co-variation (a pixel just above threshold counts the same as a very bright pixel for the binary mask).

### Which to use
Start with **Pearson R on sum projection** — it is the simplest and captures intensity co-variation directly. Follow up with **Manders** if you want to ask the directional overlap question (e.g., "what fraction of mitochondria are in HSP70-positive regions?" vs "what fraction of HSP70 is on mitochondria?"). Both metrics can be computed in the same pass with minimal extra code.

## Notes
- `.lif` metadata contains NA, refractive index, voxel sizes, and per-channel spectral bands — always prefer reading from metadata over hardcoding
- `signal_extractor` returns `results[frame][roi_num]` as strings keys
- `T` = time point (always 0 for these experiments), `C` = channel (0-indexed)
