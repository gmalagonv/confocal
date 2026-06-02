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
| `loader2(date, user, split_frames, server)` | Load `.lif` → save `.tif` stacks |
| `signal_extractor(mask_path, file_path, ...)` | Extract mean intensity per ROI per Z-frame |
| `signals(date, user, series_n, masks_suffixes, ...)` | Batch signal extraction for multiple masks/channels |
| `parse_lif_psf_params(lif_path, scene)` | Read NA, voxel sizes, emission wavelengths from `.lif` metadata |
| `deconvolve(stack, lif_path, channel, scene, num_iter)` | Richardson-Lucy deconvolution with metadata-derived PSF |
| `plot_bars_with_sem3(groups, labels, ...)` | Bar plot with SEM and overlaid data points |

## Deconvolution
- Algorithm: Richardson-Lucy (`skimage.restoration.richardson_lucy`)
- PSF: 3D (or 2D for single frames) Gaussian, parameters derived from `.lif` metadata
- PSF formulas: `σ_xy = 0.21 × λ / NA`, `σ_z = 0.66 × λ × n / NA²`
- Hot pixel removal: selective median filter (only replaces pixels > 5× std above local median)
- Output: `float32`, saved as ImageJ-compatible TIFF with voxel size metadata
- Accepts both full ZYX stacks and single 2D frames
- `num_iter=15` default; use 5–10 for a first check, 10–30 typical range

### Known microscope parameters (2025-11-29 dataset)
- Objective: HC PL APO CS2 40x/1.30 OIL (DMI8-CS)
- NA = 1.3, n = 1.518 (oil)
- Voxel size: Z = 1.0 µm, XY = 0.71 µm
- Channels: Ch0 ~529 nm (BRP), Ch1 ~659 nm (mito), Ch2 ~775 nm (HSP)
- Images are coarsely sampled (~7× below Nyquist in XY) → deconvolution effect is subtle

## Analysis notebooks
- `analysis_notebooks/` — one notebook per imaging session date (format `YY_MM_DD.ipynb`)
- Notebooks load functions from `src/data_processing.py`

## Channels in 2025-11-29 dataset
| Channel | Marker | Detection band |
|---|---|---|
| Ch0 | BRP (active zones) | 501–557 nm |
| Ch1 | Mito (mitochondria) | 557–761 nm |
| Ch2 | HSP (heat shock protein) | 761–789 nm |

## Notes
- `.lif` metadata contains NA, refractive index, voxel sizes, and per-channel spectral bands — always prefer reading from metadata over hardcoding
- `signal_extractor` returns `results[frame][roi_num]` as strings keys
- `T` = time point (always 0 for these experiments), `C` = channel (0-indexed)
