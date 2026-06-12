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

### Pinhole size and PSF calibration
The PSF formulas (`σ_xy = 0.21λ/NA`, `σ_z = 0.66λn/NA²`) are calibrated for approximately **1 Airy Unit (AU)** pinhole. Actual pinhole size changes the PSF shape — not just brightness:

| Pinhole | Actual PSF vs formula | Consequence for RL |
|---|---|---|
| ~1 AU | Matches | Correct deconvolution |
| < 1 AU (e.g. 0.2 AU) | Narrower (~0.7×) | Formula overestimates blur → **over-deconvolution** (ringing, structure fusion) |
| > 1 AU | Broader | Formula underestimates blur → under-correction |

**Correction for sub-AU pinholes** — approximate scaling factor:
```python
def _pinhole_psf_factor(pinhole_au):
    p = min(max(pinhole_au, 0.0), 1.0)
    return 1.0 / np.sqrt(2) + (1.0 - 1.0 / np.sqrt(2)) * p  # 0.707 at 0 AU → 1.0 at 1 AU

factor = _pinhole_psf_factor(pinhole_au)
sigma_xy_px *= factor
sigma_z_px  *= factor
```
Gold standard: image 100 nm fluorescent beads and fit PSF directly — accounts for all aberrations.

**Practical guidance on pinhole choice:**
- **1 AU**: best balance — formula is well-calibrated, ~84% of focal-plane signal collected, good SNR. Recommended for quantitative intensity/colocalization measurements.
- **< 1 AU** (e.g. 0.2 AU): maximum lateral resolution (narrower PSF by ~√2), better optical sectioning, but ~10× less light, noisier images, PSF mismatch degrades deconvolution. Only justified when resolving closely spaced sub-diffraction structures (e.g. individual BRP puncta) in a thin, bright sample.
- **> 1 AU**: more light, approaches widefield (less optical sectioning). Useful for thick/scattering tissue or very weak signals.

### 3D vs 2D deconvolution — practical threshold
The code switches to 3D RL at `σ_z_px ≥ 2.0`. In practice, **3D RL is unstable near this threshold**:
- At σ_z ≈ 2 px (barely Nyquist): zero-padding boundary artifacts appear in the first/last ~3 frames (they become progressively brighter with more iterations); structures near each other can fuse.
- Reliable 3D RL requires **σ_z ≥ 3 px**. Consider raising the threshold in the code:
  ```python
  use_3d_psf = is_3d and (sigma_z_px >= 3.0) and not forced2d
  ```
- When in doubt, use `forced2d=True`. 2D-per-frame is more conservative and avoids Z-boundary artifacts entirely.

### Quality check used in notebooks
Re-blur the deconvolved result with `gaussian_filter(result[z], sigma=sigma_xy_px)` and compute the residual vs the original frame. Flag frames where the background noise ratio (`result[bg].std() / original[bg].std()`) exceeds 1.5× — those frames may have noise amplification. The background mask is defined as pixels below the 10th percentile of the original frame.

**Known limitations of the quality check:**
- Skipped frames (dark edge slices where `bg_orig_std == 0`) append `0` (coded as "no problem") and are included in the denominator — artificially lowering the reported fraction. Track skipped frames separately for an accurate count.
- The peak-ratio check (`result.max() / stack.max() > 5`) rarely fires because `_process` normalises by `img.max()` before RL and rescales after. It can be dropped.
- The check is global (pixel values) and does **not** detect ringing/structure fusion — those require visual inspection of individual frames.

### Parallelizing deconvolution loops
`AICSImage.set_scene()` is stateful — parallel calls corrupt each other's scene pointer. Two-phase approach:
```python
from joblib import Parallel, delayed

# Phase 1: load stacks sequentially (img is stateful)
stacks = {}
for scene in scenes:
    img.set_scene(img.scenes[scene])
    for channel in channels:
        stacks[(scene, channel)] = img.get_image_data("ZYX", T=0, C=channel)

# Phase 2: parallel deconvolution (threading shares imports & sys.path)
def run_one(stack, scene, channel, num_iter, forced2d):
    result, sigma_xy_px = deconvolve(stack, path, channel=channel, scene=scene,
                                     num_iter=num_iter, forced2d=forced2d)
    return analyze_deconvolution_results(result, stack)

results = Parallel(n_jobs=8, backend='threading')(
    delayed(run_one)(stacks[(s, c)], s, c, n, f2d)
    for s in scenes for c in scene_channels[s]
    for n in iterations for f2d in [False, True]
)
```
Use `backend='threading'` (not `'loky'`): threads share the notebook's `sys.path` so imports work; scipy's FFT releases the GIL so real CPU parallelism is achieved. With 16 cores, `n_jobs=8` is a safe starting point.

### Known microscope parameters (2025-11-29 dataset)
- Objective: HC PL APO CS2 40x/1.30 OIL (DMI8-CS)
- NA = 1.3, n = 1.518 (oil)
- Voxel size: Z = 1.0 µm, XY = 0.71 µm
- Channels: Ch0 ~529 nm (BRP), Ch1 ~659 nm (mito), Ch2 ~775 nm (HSP)
- Images are coarsely sampled (~7× below Nyquist in XY) → σ_xy_px ≈ 0.12, deconvolution effect is subtle; σ_z_px ≈ 0.44 → 2D-per-frame mode will be selected automatically

### Known microscope parameters (2026-05-26 dataset)
- Objective: HC PL APO CS2 **63x/1.40 OIL** (NA = 1.4, different from 2025-11-29 which was 40x/1.30)
- NA = 1.4, n = 1.518 (oil)
- Antibodies: Alexa 488 (BRP), Alexa 546 (mitochondria), Cy5 (HSP70)

| Scene index | Scene name | Z-frames | XY (µm/px) | Z (µm/step) | Notes |
|---|---|---|---|---|---|
| 0 | `1zoom_all` | 110 | 0.361 | 0.534 | 1× digital zoom — coarse XY, σ_xy ≈ 0.22 px (below Nyquist) |
| 1 | `16zoom_vac` | 110 | 0.023 | 0.292 | 16× digital zoom — well sampled XY, σ_xy ≈ 3.5 px |
| 2 | `16zoom_a3` | 18 | 0.023 | 0.661 | 16× zoom, α3 subregion only |

- All scenes → σ_z_px < 2 → 2D-per-frame deconvolution is selected automatically
- Scene 0 is coarsely sampled in XY (same situation as 2025-11-29 dataset); deconvolution has negligible effect
- Scenes 1 and 2 are well sampled in XY; deconvolution is meaningful

#### Acquisition structure (from .lif metadata `ATLConfocalSettingDefinition` blocks)
Sequential acquisition with two sequences per Z-stack:

| Sequence | Active lasers | Detector | Detection band | Dye | Image channel |
|---|---|---|---|---|---|
| Seq 1 (sequential) | 552 nm only | PMT2 | 557–761 nm | Alexa 546 | ch2 |
| Seq 2 (simultaneous) | 488 + 638 nm | PMT1 | 501–610 nm | Alexa 488 | ch0 |
| Seq 2 (simultaneous) | 488 + 638 nm | PMT2 | 643–788 nm | Cy5 | ch1 |

**ch0 (BRP/Alexa 488) and ch1 (HSP70/Cy5) are acquired simultaneously in Seq 2; ch2 (Mito/Alexa 546) is acquired separately in Seq 1.**

#### Spectral bleed-through
Determined by cross-referencing active lasers with detection bands:
- **Alexa 546 → ch0 (BRP channel)**: During Seq 2, the 488 nm laser cross-excites Alexa 546 (~5–15% efficiency). Alexa 546 emits at 573 nm, which falls within the ch0 detection band (501–610 nm). **Mitochondria (Alexa 546) bleed into the BRP channel.** This should be considered when interpreting BRP signals.
- **ch1 ↔ ch2 (HSP70/Mito pair)**: Clean — acquired in separate sequences with non-overlapping excitation. No significant bleed-through. This pair is suitable for colocalization analysis.
- Bleed-through was identified by parsing `ATLConfocalSettingDefinition` XML blocks from the `.lif` file and checking whether emission peaks of non-target dyes fall within each detector's active wavelength band.

### Known microscope parameters (2026-06-05 dataset)
- Dataset: `2026_06_05_Carmina` — 10 scenes, alpha prime and gamma Mushroom Body regions (left/right hemispheres, multiple brains)
- Objective: HC PL APO CS2 **63x/1.40 OIL** (NA = 1.4, same objective as 2026-05-26)
- NA = 1.4, n = 1.518 (oil)
- Voxel size: XY = 0.0226 µm/px, Z = 0.1307 µm/step
- **Pinhole: 20 µm = 0.209 AU** — very small, nearly ideal confocal. Actual PSF is ~30–40% narrower than formula. Use `forced2d=True` or ≤ 2 iterations to avoid over-deconvolution.
- Antibodies: Alexa 488 (BRP/ch0), Cy5 (HSP70/ch1), Alexa 546 (Mito/ch2)
- Acquisition: two sequences — Seq1: 488+638 nm simultaneous (ch0 BRP + ch1 HSP70); Seq2: 552 nm only (ch2 Mito)
- σ values at 0.209 AU (formula overestimates): formula gives σ_xy=3.45 px, σ_z=2.03 px for ch0 (Alexa 488); true values ~2.4 px and ~1.4 px respectively → actual PSF is below Nyquist in Z
- Bleed-through: Alexa 546 → ch0 (MEDIUM); Cy5 → ch2 (LOW)

## Analysis notebooks
- `analysis_notebooks/` — one notebook per imaging session date (format `YY_MM_DD.ipynb`)
- Notebooks load functions from `src/data_processing.py`

### `26_06_05_gerardo.ipynb` — deconvolution testing on 2026_06_05_Carmina data
Tests RL deconvolution across all 10 scenes, 3 channels, multiple iteration counts (2, 4, 6, 10, 15, 20), comparing automatic mode vs `forced2d=True`. Uses parallelized approach (joblib threading). Key finding: at 0.209 AU pinhole, 3D mode causes boundary artifacts and structure fusion even at 2 iterations; `forced2d=True` with ≤ 4 iterations is more appropriate for this dataset.

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
Primary channel pair: **Ch1 (HSP70) ↔ Ch2 (Mito)**. Other pairs (Ch0↔Ch1, Ch0↔Ch2) may follow the same workflow.

### Step 1 — Generate a foreground mask via thresholding
No Fiji-drawn ROI masks exist for this dataset. Use thresholding on each channel to define foreground, then take the union as the analysis region.

**You do not need to use the same algorithm for every channel.** Choose based on each channel's histogram shape.

#### Algorithm options

| Algorithm | `skimage` function | Best when |
|---|---|---|
| Otsu | `threshold_otsu` | Histogram is clearly bimodal (distinct foreground/background) |
| Li (min. cross-entropy) | `threshold_li` | Foreground is sparse — Otsu pulls threshold too low because background dominates |
| Fixed percentile | `np.percentile(img, p)` | Need consistency across samples regardless of histogram shape |

**Li (minimum cross-entropy):** finds threshold `t` minimising `Σ_{i≤t}[i·log(i/μ_bg)] + Σ_{i>t}[i·log(i/μ_fg)]`. Less sensitive to class imbalance than Otsu — tends to place threshold closer to foreground mean, better for sparse bright structures like HSP70 puncta.

```python
from skimage.filters import threshold_otsu, threshold_li

# Start with Otsu on both; switch Li for a channel if mask looks over/under-segmented
thresh1 = threshold_otsu(ch1)   # or threshold_li(ch1)
thresh2 = threshold_otsu(ch2)   # or threshold_li(ch2)
mask = (ch1 > thresh1) | (ch2 > thresh2)  # union: anywhere either signal is present
```

**Visual check before committing to an algorithm:**
```python
import matplotlib.pyplot as plt

for img, name in [(ch1, 'Ch1 Mito'), (ch2, 'Ch2 HSP70')]:
    plt.figure()
    plt.hist(img.ravel(), bins=256, log=True)
    plt.axvline(threshold_otsu(img), color='b', label='Otsu')
    plt.axvline(threshold_li(img),   color='r', label='Li')
    plt.title(name); plt.legend()
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
