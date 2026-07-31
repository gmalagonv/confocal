## Understanding the colocalization metrics

To ask whether two proteins occupy the same pixels in the image, we threshold each channel to obtain a binary mask — pixels are either positive (signal present) or negative (signal absent). All three metrics below are computed within a defined region of interest (e.g. the Mushroom Body).

---

### Directional overlap fractions (binary Manders coefficients)

These answer: *of all pixels positive for protein A, what fraction are also positive for protein B?* And vice versa.

- **Fraction of A in B** = |A∩B| / |A| — "what fraction of A pixels have B signal?"
- **Fraction of B in A** = |A∩B| / |B| — "what fraction of B pixels have A signal?"

These are directional: they can be different from each other. For example, if mitochondria are large and HSP70 is a small subset that sits on them, the fraction of HSP in Mito will be high (most HSP is on mitochondria), while the fraction of Mito in HSP will be lower (mitochondria are big, and only part of their surface has HSP).

---

### Enrichment over random

The overlap fractions alone do not tell you whether the co-localization is *more than expected by chance*. If 80% of pixels in a region are Mito-positive and 70% are HSP-positive, you would expect ~56% overlap purely by random spatial overlap, regardless of any physical association.

The enrichment corrects for this:

**Enrichment = observed overlap / expected random overlap**

where the expected overlap is calculated assuming the two signals are distributed independently within the region. An enrichment of 1.0 means no association beyond chance; an enrichment of 6× means the two proteins co-localize six times more than would be predicted if they were independently distributed.

---

### Odds ratio

The odds ratio (OR) is the most statistically rigorous of the three. It asks: *how much more likely is a pixel to be B-positive if it is A-positive, compared to if it is A-negative?*

#### How it is computed: the 2×2 contingency table

Every pixel inside the analysis region (MASK) is classified into one of four categories:

```
                    Mito-positive      Mito-negative
HSP-positive          A_and_B            A_not_B
HSP-negative          B_not_A            neither
```

```python
A_and_B = (mask_chA_MB &  mask_chB_MB).sum()   # positive for both
A_not_B = (mask_chA_MB & ~mask_chB_MB).sum()   # HSP only
B_not_A = (~mask_chA_MB &  mask_chB_MB).sum()  # Mito only
neither = MASK.sum() - A_and_B - A_not_B - B_not_A  # neither (within MASK)
```

The `neither` cell is computed by subtraction rather than directly as `(~mask_chA_MB & ~mask_chB_MB).sum()`. Even though `mask_chA_MB` has the MASK baked in, taking its complement undoes the restriction — by De Morgan's law:

```
~mask_chA_MB = ~(mask_chA & MASK) = ~mask_chA | ~MASK
```

The `| ~MASK` term means every pixel *outside* the MASK evaluates to True, so they all get counted in `neither`, inflating it by the entire image background. The subtraction approach avoids any complement and is always correct, since the four cells must exactly partition the MASK:

```python
neither = MASK.sum() - A_and_B - A_not_B - B_not_A
```

The equivalent direct fix (complement before restricting) would be `(MASK & ~mask_chA & ~mask_chB).sum()`, but the subtraction form is simpler.

Fisher's exact test then asks: are the rows (HSP status) and columns (Mito status) independent? The OR it returns is:

```
OR = (A_and_B × neither) / (A_not_B × B_not_A)
```

If HSP and Mito are spatially independent, a HSP-positive pixel is no more likely to be Mito-positive than a HSP-negative pixel → OR = 1. OR > 1 means co-occurrence is above chance.

`alternative='greater'` tests the one-sided hypothesis OR > 1. With microscopy images (tens of thousands of pixels), p-values are almost always near zero — what matters is the size of the OR, not the p-value.

- **OR = 1**: knowing a pixel is A-positive gives no information about B — the two signals are independent
- **OR > 1**: A-positive pixels are more likely to carry B signal — positive co-localization
- **OR = 52**: an A-positive pixel is 52 times more likely to also be B-positive than an A-negative pixel

The OR is accompanied by a p-value. With microscopy images (thousands of pixels), p-values are almost always effectively zero — what matters is the **size of the OR**, not whether p < 0.05. An OR of 2 is meaningful; an OR of 900 is dramatic.

---

### Using the three metrics together

| Metric | What it tells you |
|---|---|
| Fraction A in B | How much of A is "covered" by B |
| Fraction B in A | How much of B is "covered" by A |
| Enrichment | Is co-localization above what chance would produce? |
| Odds ratio | How strongly are A and B spatially associated? |

A complete interpretation uses all three: the fractions describe the *extent* of overlap, the enrichment shows whether it is *above chance*, and the odds ratio quantifies the *strength of association* independently of how prevalent each signal is.

### Why the two fractions can be very different — and why that matters

The two fractions are not symmetric and will often differ substantially. This asymmetry is biologically informative, not an artifact.

**Example from data (MB_total, all triangle thresholding):**

| Region | Fraction HSP in Mito | Fraction Mito in HSP |
|---|---|---|
| alphaprime | 0.87 | 0.10 |
| gamma | 0.54 | 0.11 |
| alpha | 0.30 | 0.08 |

Interpretation: most HSP70 puncta sit on mitochondria (high fraction of HSP in Mito), but mitochondria are so abundant that only a small fraction of them carry HSP70 at any given time (low fraction of Mito in HSP). This asymmetry reflects the biology — HSP70 is a sparse, punctate stress-response signal; mitochondria are a large, continuous network.

**The OR and enrichment tell you the association is above chance. The fractions tell you the extent:**
- High fraction of HSP in Mito → most HSP70 is mito-associated
- Low fraction of Mito in HSP → mitochondria are only partially covered by HSP70
- Both can be true simultaneously and together give a fuller picture than either metric alone

**Caveat:** fractions are more sensitive to threshold choice than OR or enrichment, because the denominator (|A| or |B|) changes directly with mask size. With a consistent thresholding algorithm applied across all movies, fractions are valid to compare. With mixed algorithms, interpret with caution.

---

### Density: single-channel voxel occupancy

Not a joint two-channel metric like the three above — `density_HSP_syn`, `density_Mito_non_syn` etc. ask a simpler question about *one* channel at a time: what fraction of a given compartment's voxels are positive for that channel?

```python
density_HSP_syn = (hsp_mask & SYN_mask).sum() / SYN_mask.sum()
```

**This is not a physical density.** No physical unit is involved — both numerator and denominator are voxel counts from the same image, so the ratio is a dimensionless occupancy/coverage fraction (0 to 1), not a concentration (e.g. molecules/µm³) or an object count (e.g. puncta/µm³). It also carries no intensity information: a voxel counts as "1" if it crosses the mask threshold, regardless of how far above threshold it sits, so two regions with very different absolute brightness but the same fraction of voxels crossing threshold give identical density values.

**Units: voxels, not pixels.** These masks come from full 3D Z-stacks (`ZYX`), so each unit being counted is a 3D volume element (voxel — with a physical volume set by the acquisition's X×Y×Z voxel size), not a 2D picture element (pixel, no thickness).

**A useful invariance:** because numerator and denominator are both voxel counts from the *same* image, the physical volume of one voxel cancels out of the ratio — this metric stays comparable across sessions/acquisitions with different voxel sizes (e.g. different Z-step), unlike a raw voxel count would.

**Relationship to `prop_MB_*`:** `density_HSP_syn`/`density_HSP_non_syn` are the same calculation as `prop_MB_HSP`, just restricted to the synaptic/non-synaptic sub-compartment instead of the whole MB — together they show how the whole-MB occupancy splits between the two compartments.

---

## Pooling data across replicates

### Experimental structure

Measurements are nested: multiple brain regions within each brain, multiple brains per condition. This hierarchy determines what is independent and what is not.

| Comparison | Independent? |
|---|---|
| Brain 1 alpha vs Brain 2 alpha | Yes — different animals |
| Left alpha vs Right alpha within the same brain | No — same animal, same staining |
| Alpha vs gamma within the same brain | No — same animal, but answers a different biological question |

**Rule:** left and right hemisphere versions of the same region are pseudoreplicates — average them within each brain first, then treat each brain as one data point. Different region types (alpha, gamma, alpha prime) are kept separate; they answer different questions and can be compared as a region effect.

### Comparing two factors at once: mixed paired/unpaired designs

A common case: you want to compare two compartments (e.g. synaptic vs non-synaptic) *and* two conditions (e.g. two training groups) in the same figure — four groups total. The independence structure is now mixed, not uniform:

- **Within a condition, the two compartments are paired** — both computed from the same animal/image.
- **Across conditions, samples are independent** — different animals.

A single statistical test can't reflect both at once. Naively running one test across all four groups picks the wrong structure either way:
- Treating all four as **paired** (e.g. repeated-measures/Friedman) assumes every value in a row comes from the same subject — false, since the two conditions are different animals.
- Treating all four as **independent** (e.g. one-way ANOVA/Kruskal-Wallis) ignores the real within-animal pairing between the two compartments.

**The fix: only test the comparisons that are actually meaningful and correctly matched, individually:**
- **Compartment A vs compartment B, within condition 1** — paired test (Wilcoxon signed-rank / paired t-test).
- **Compartment A vs compartment B, within condition 2** — paired test.
- **Condition 1 vs condition 2, at compartment A** — unpaired test (Mann-Whitney U / independent t-test).
- **Condition 1 vs condition 2, at compartment B** — unpaired test.

That's 4 of the 6 possible pairwise comparisons among 4 groups — the remaining 2 (e.g. condition-1-compartment-A vs condition-2-compartment-B) mix both factors at once and don't answer a single, well-posed question, so they're best left untested rather than added as extra multiple-comparison noise.

**Caveat carried over from "Experimental structure" above:** each of these four groups must itself already be de-duplicated to one value per animal (hemispheres/regions averaged first) before running any of these tests — if a group is actually dominated by one animal's multiple regions, the four tests above don't fix that. See the case study below for a concrete example of this being knowingly accepted (not fixed) in a preliminary analysis of this project's own data.

### How to pool each metric

**Fractions (Manders coefficients)**

Average per-brain values, report mean ± SEM across brains. Fractions are bounded [0, 1], so if values cluster near 0 or 1 use a logit transform before statistical testing:

```python
import numpy as np

logit_m = np.log(m / (1 - m))   # transform
# ... t-test or Mann-Whitney on logit_m across brains
```

**Enrichment**

Average per-brain enrichment values directly, report mean ± SEM. The natural null is 1.0 (random co-distribution), so the mean is directly interpretable. Enrichment ≈ 1 in a given brain is real data — it means co-distribution is not above chance in that animal and should not be excluded.

**Odds ratio**

OR must be pooled on the **log scale** because it is right-skewed (range 0 to ∞, null at 1). OR = 0.5 and OR = 2 are equally distant from random on a multiplicative scale, but look asymmetric on a linear scale.

```python
log_ors = np.log(or_values)          # one value per brain

mean_log_or = np.mean(log_ors)
sem_log_or  = np.std(log_ors, ddof=1) / np.sqrt(len(log_ors))

pooled_OR = np.exp(mean_log_or)
ci_low    = np.exp(mean_log_or - 1.96 * sem_log_or)
ci_high   = np.exp(mean_log_or + 1.96 * sem_log_or)
```

Report as: **OR = X (95% CI: Y–Z)**.

**Edge case:** if any brain has a zero cell in the 2×2 contingency table (e.g. no pixels that are A-positive and B-negative), OR is undefined. Add 0.5 to each cell before computing (Haldane-Anscombe correction):

```python
a, b, c, d = AB + 0.5, A_only + 0.5, B_only + 0.5, neither + 0.5
or_val = (a * d) / (b * c)
```

**Flag extreme ORs** (e.g. > 500): these usually mean one contingency table cell is near zero, often in a small sub-region. Check the raw counts before including them in the pooled mean — they will dominate log-OR averages.

#### Reading a log(OR) plot in practice

- **Zero is the reference, not an arbitrary baseline** — `log(OR=1) = 0`, and OR=1 means "no association beyond chance." Read each bar relative to zero:
  - **Above 0** → OR > 1 → more co-occurrence than chance in that group/compartment.
  - **At/near 0**, especially if the SEM error bar crosses zero → not distinguishable from independence, regardless of what any other bar shows.
  - **Below 0** → OR < 1 → less overlap than chance predicts (active exclusion, a different story from "no association").
- **Converting a bar height back to an actual OR**: `OR = exp(log(OR))`. Report *this* number in text/results, not the log value — the log scale is for the plot and the statistics, not for reporting.
- **Reading the gap between two bars**: subtracting two log(OR) values gives `log(OR_a / OR_b)` — a **ratio**, not a difference in absolute strength. A gap of 0.7 between two bars means one OR is `e^0.7 ≈ 2.0×` the other, not "0.7 units" stronger.
- **What a significance bracket between two bars does and doesn't tell you**: it tells you those two groups' OR values reliably differ from each other. It does **not** tell you whether either one individually is significantly above chance (OR ≠ 1) — that's a separate question, answered by whether each bar (with its SEM) sits clearly away from the zero line.

### Should you exclude replicates with enrichment ≈ 1?

No — not based on the enrichment value itself. Enrichment ≈ 1 means co-distribution is not above chance in that brain. Excluding those samples would inflate your reported colocalization and is circular reasoning. Legitimate exclusion criteria are quality-based and blind to the result:

- One channel fell below a minimum pixel count above threshold (mask too sparse to be reliable)
- Obvious imaging artifact visible on inspection
- Signal-to-background ratio below a pre-defined threshold

### Sensitivity of each metric to threshold choice

Threshold choice affects all three metrics, but not equally:

| Metric | Sensitivity to threshold variability |
|---|---|
| Fractions (Manders) | High — denominator changes directly with mask size |
| Enrichment | Medium — expected term partially compensates for mask size changes |
| Odds ratio | Lowest — ratio of ratios, partially self-correcting |

When thresholding algorithms are inconsistent across movies, rely primarily on OR (log-scale) and enrichment as the main reported metrics. Treat fractions as descriptive/secondary.

---

## Thresholding strategy across movies

### Within a movie: different algorithms per channel are fine

Choose the algorithm based on each channel's histogram shape:

| Algorithm | Best when |
|---|---|
| Otsu | Histogram is clearly bimodal |
| Li (min. cross-entropy) | Foreground is sparse — Otsu pulls threshold too low |
| Triangle | One dominant background peak with a long tail |
| Fixed percentile | Need consistency across movies regardless of histogram shape |

### Across movies: the algorithm must be consistent per channel

If movie 1 uses Otsu and movie 2 uses Li for the same channel, the fractions become non-comparable — the threshold shift changes mask size, which changes all three metrics. Options in order of preference:

1. **Fix one algorithm per channel** across all movies. Choose based on what fits the typical histogram across the dataset.
2. **Fixed percentile**: `thresh = np.percentile(channel_img, p)` — consistent regardless of histogram shape. Set the percentile based on the expected biological foreground fraction for that channel (e.g. 87th percentile if ~13% of pixels should be foreground). Use a different percentile per channel, fixed across all movies:

```python
THRESHOLDS = {
    'ch1_mito':  65,   # ~35% foreground expected
    'ch2_hsp70': 87,   # ~13% foreground expected
}
thresh_mito  = np.percentile(ch1, THRESHOLDS['ch1_mito'])
thresh_hsp70 = np.percentile(ch2, THRESHOLDS['ch2_hsp70'])
```

3. **Conservative rule**: `thresh = max(threshold_otsu(img), threshold_li(img))` — always take the more restrictive of two algorithms. The decision criterion is fixed even if the winning algorithm varies per movie. Defensible if described explicitly.

**Caveat for fixed percentile:** this approach assumes the foreground *fraction* is roughly constant across movies. If the amount of signal is itself a biological variable (e.g. more HSP70 puncta in stressed brains), a fixed percentile will erase that difference. In that case use an intensity-based threshold (Otsu/Li) and fix the algorithm choice instead.

**Sanity check after thresholding:** verify foreground fractions are in a similar range across movies for the same channel:

```python
frac = mask.sum() / mask.size
print(f"{movie}: {frac:.2%} foreground")
```

Large movie-to-movie variation in foreground fraction (e.g. 5% vs 30%) signals a staining or imaging variability problem that thresholding cannot fix.

---

## Reporting uncertainty: the 95% confidence interval

When you pool OR values across brains, the result is an estimate — your best guess at the true OR for that region and condition, given the brains you measured. The **95% confidence interval (CI)** quantifies the precision of that estimate.

**What it means:** if you repeated the experiment many times and computed the CI each time, 95% of those intervals would contain the true population value. In practice: it tells you how wide the range of plausible true values is, given your sample size.

**How to read it:**

- **OR = 7.4 (95% CI 2.7–20.2)**: best estimate is 7.4, but with n=3 brains the true value could plausibly be anywhere from 2.7 to 20.2. All values are above 1, so colocalization above chance is certain — but the exact magnitude is uncertain.
- **OR = 2.3 (95% CI 1.8–2.8)**: tight interval — three brains gave very similar values, so the estimate is reliable.

**How it is computed** (for log-OR pooling across brains):

```python
log_ors = np.log(or_per_brain)
mean_log_or = log_ors.mean()
sem_log_or  = log_ors.std(ddof=1) / np.sqrt(len(log_ors))

pooled_OR = np.exp(mean_log_or)
ci_low    = np.exp(mean_log_or - 1.96 * sem_log_or)
ci_high   = np.exp(mean_log_or + 1.96 * sem_log_or)
```

The 1.96 comes from the normal distribution — it captures the middle 95% of values. The CI is computed on the log scale and then back-transformed, so it is asymmetric around the point estimate (wider on the high side), which is correct for a ratio.

**CI width depends on:**
- **n** (number of brains): the main driver. Doubling n roughly halves the CI width on the log scale.
- **Variability between brains**: if brains give very different ORs, the CI will be wide even with a reasonable n.

---

## Signal-to-noise: is there enough SNR to trust these metrics?

Colocalization metrics are computed on binary masks derived from a threshold. If a channel's threshold doesn't clearly separate real signal from background noise, the "positive" mask is capturing noise, not structure — and every downstream metric (fractions, enrichment, OR) is describing noise co-occurrence, not biology.

This matters most for a **fixed-percentile threshold** (e.g. `np.percentile(stack, 95)`): by construction it always keeps the top 5% of voxels, whether the channel has crisp bright puncta on a clean background or is barely above the noise floor. Mask size alone never tells you whether that cut is meaningful.

### An SNR proxy for percentile-thresholded masks

```python
def snr_proxy(stack, p95_thresh=None):
    """(p95 - median) / std of background (voxels <= median)."""
    stack = np.asarray(stack)
    median = np.median(stack)
    if p95_thresh is None:
        p95_thresh = np.percentile(stack, 95)
    bg = stack[stack <= median]
    std_bg = bg.std()
    return round((p95_thresh - median) / std_bg, 3)
```

**Rationale for each term:**

- **`median`** — a robust stand-in for the "typical background level." The mean is not used because a stack is mostly background with a small fraction of bright foreground, which drags the mean upward; the median stays anchored to the bulk of the histogram as long as foreground occupies well under 50% of voxels.
- **`std(bottom 50%)`** — an estimate of pure noise spread that deliberately excludes the brightest half of the stack, so real signal can't inflate it. Using the whole-stack std instead would let the very thing you're trying to detect (bright foreground) contaminate the noise estimate, making every channel look artificially noisier than it is.
- **`p95 − median`** — how far above typical background the actual mask threshold sits, in raw intensity units.
- **Dividing by `std(bottom 50%)`** — converts that gap into background-noise standard deviations, analogous to the classic imaging SNR definition `(signal − background) / noise_std`.

**How to read it:** a large value (roughly >5–8) means the threshold sits many background-sigmas above typical noise — the mask is capturing real bright structure. A small value (roughly <2–3) means the cutoff is within a couple of background standard deviations of "typical" — for a roughly Gaussian background, ~2.3% of voxels sit >2σ above the mean by chance alone, so "top 5% brightest" is barely distinguishable from "top 5% noisiest," and colocalization computed on that mask is largely noise-driven.

**Caveat:** this is a ratio computed within a single image, so a pure linear rescaling of pixel values (e.g. a different PMT gain applied uniformly) leaves it unchanged — numerator and background std scale together. It is *not* immune to changes in the underlying noise process itself (e.g. genuinely fewer photons collected, which changes shot-noise statistics rather than just rescaling values).

### Checking whether SNR is confounding a colocalization comparison

Compute `snr_proxy` per channel/sample, take the minimum across the channel pair being compared (`snr_min`, since a joint overlap metric is only as trustworthy as its noisier input channel), and correlate it against OR/enrichment across samples (Spearman, since the relationship need not be linear):

```python
from scipy.stats import spearmanr
r, p = spearmanr(df['snr_min'], df['MB_total_odds_ratio'])
```

A positive correlation here is **ambiguous by itself** — it is equally consistent with two very different explanations:

1. **Real biology, revealed better at high SNR**: the true association differs between samples, and only high-SNR samples measure it accurately.
2. **Pure noise artifact (attenuation toward the null)**: non-differential misclassification of both channel masks systematically biases OR/enrichment toward 1 as noise increases. If every sample shared the *same* true colocalization, lower-SNR samples would still show a lower measured OR purely from noise — a well-known effect in measurement-error statistics, not specific to this pipeline.

Correlation alone cannot distinguish these.

### Noise-injection calibration: testing the "pure artifact" explanation

To test explanation (2) directly: take the highest-SNR sample (the one trusted most), progressively add synthetic Gaussian noise to its two channels (scaled as multiples of that channel's own background std), and recompute the same mask → OR/enrichment pipeline at each noise level. Because the true underlying image never changes, any change in the measured OR across noise levels is coming from noise alone — this traces out what "colocalization" looks like as a function of SNR when true biology is held perfectly constant.

```python
def add_gaussian_noise(stack, sigma):
    if sigma == 0:
        return stack.astype(np.float32)
    return stack.astype(np.float32) + np.random.normal(0, sigma, size=stack.shape).astype(np.float32)
```

Interpolating each real sample's own SNR onto this noise-only curve gives an expected OR — "what this sample's OR would be if it shared the reference's true biology and its lower value were pure artifact." The ratio of the observed OR to this expectation (`excess_ratio`) is the diagnostic:

| `excess_ratio` | Interpretation |
|---|---|
| ≈ 1 | Fully consistent with noise-only attenuation — no evidence this sample differs biologically from the reference |
| > 1 | More colocalization than noise alone predicts at this SNR — evidence of a real signal beyond the artifact |
| < 1 | Less colocalization than even pure noise would produce — a systematic (not random) explanation is needed: real biological difference, or an unrelated sample-specific issue |

**What this rules out, and what it doesn't:**

- If `excess_ratio` scatters randomly around 1 across samples, the "pure noise" null (explanation 2) is a sufficient explanation — don't read biological meaning into the spread.
- If `excess_ratio` shows a **systematic, one-sided pattern clustered by animal/sample** (e.g. every subregion of one brain sits well below 1, every subregion of another sits near/above 1), that pattern rules out pure random per-image noise as the explanation — noise degrading a shared signal would scatter independently per image, not consistently by animal. This is evidence that *some* real, systematic difference exists between samples.
- **It cannot distinguish *what kind* of real difference.** True biological variation between animals and a technical/staining-batch effect (e.g. weaker antibody penetration, different fixation quality — anything that lowers true signal in every channel for that whole prep) produce the identical pattern in this test. Resolving that requires information outside the colocalization pipeline itself (e.g. whether samples were stained in the same batch/session, or an independent quality marker unrelated to the channels being compared).

**Caveat on the noise model:** the calibration adds simple additive Gaussian noise on top of the reference sample's own stack. If gain/laser power are tuned per-sample specifically to avoid saturation (common when raw fluorescence brightness varies a lot between samples), the real degradation in a dim sample is closer to genuine **photon-limited (shot) noise from fewer collected photons** — signal-dependent, not simple additive Gaussian. The calibration is therefore a useful sanity check and rough lower bound on how much noise alone can distort the metric, not an exact model of what a dimmer sample "really" looks like.

### Before comparing conditions (e.g. training groups): check SNR balance first

Averaging colocalization metrics across several biological replicates per condition is the right way to average out animal-to-animal variability. But if gain/laser power are adjusted per-sample, and that adjustment correlates with condition (e.g. one training condition tends to produce dimmer samples, prompting systematically higher gain for that whole group), a between-group comparison can re-measure an SNR difference rather than a true biological one.

Practical check: compute `snr_proxy` per sample and compare its distribution between conditions (e.g. Mann-Whitney U, given typically small n per group) before trusting a group-level OR/enrichment comparison.
- **Balanced SNR across groups** → proceed with the group comparison; a difference found is harder to explain away as an acquisition artifact.
- **Unbalanced SNR across groups** → either include SNR as a covariate in the group comparison, or use the noise-corrected `excess_ratio` metric instead of the raw OR/enrichment.

---

## Case study: diagnosing a confounded training-condition comparison

The following is a record of a real multi-session debugging process (2026-06-05/15/25 datasets, 5X vs 1X training, HSP↔Mito colocalization). Kept here because the pitfalls are generic and will recur with any new pooled dataset.

### Interpretation: what the diagnostics actually showed

- **Pinhole size differed by imaging session** (0.209 AU vs ~1.0 AU — a ~5x difference in physical diameter), and this tracked almost perfectly with which training condition was imaged in which session. Detected by tabulating `describe_acquisition()` output across all dates in the pooled dataset (pinhole, voxel size, NA, bit depth side by side) — any acquisition parameter that varies by date rather than by animal is a candidate confound.
- **Deconvolution reliability differed drastically between those same sessions.** The existing quality-control metric (fraction of frames per scene/channel where post-deconvolution background noise std exceeds 1.5x the pre-deconvolution value) was ~0% at 0.209 AU but a **median of 100%** (essentially every frame flagged) at ~1.0 AU — plausibly because a larger pinhole collects more out-of-focus light, violating the in-focus-blur assumption Richardson-Lucy relies on.
- **An accidental duplicate scan** (same tissue, nothing deliberately changed) showed an **11x odds-ratio discrepancy on deconvolved data**, shrinking to **~1.2x on raw (undeconvolved) data** — direct evidence that deconvolution, not the acquisition itself, was the dominant source of that instability.
- Recomputing the whole dataset from raw instead of deconvolved images collapsed odds ratios from an implausible 1.2–77 range down to a far more believable 1.0–2.4. Most of the dramatic "colocalization" signal seen in deconvolved data was very likely a deconvolution artifact (structure fusion from PSF mismatch, since the PSF was never bead-calibrated), not real biology.
- Even after switching to raw data, the pinhole/session split **persisted, just at a much smaller magnitude** (mean OR 1.10 at 0.209 AU vs 1.58 at ~1.0 AU) — so deconvolution was inflating the effect, but not fabricating the underlying pattern on its own.
- The observed (not statistically supported) direction, on raw data: 1X-trained animals showed a higher OR than 5X in every subregion tested (alpha, alphaprime, gamma), both pooled across sessions and in the smaller pinhole-matched subset.

### Why no conclusion about training condition could be drawn

- **Confounding.** Pinhole and training condition are nearly collinear in this dataset — only one 5X animal happened to share the pinhole setting used for all 1X animals. No statistical adjustment recovers a clean comparison from data this confounded; it requires new, deconfounded acquisitions.
- **Power.** Even the best available (pinhole-matched) subset was n=2 vs n=2. A Mann-Whitney test at that sample size cannot reach conventional significance under any arrangement of the data (its minimum achievable p-value is ~0.33) — this is a structural limit, not a "no effect found this time" result.
- **Subregion splitting made it worse, not better.** Dividing an already-too-small comparison by anatomical subregion (alpha/alphaprime/gamma) shrinks each subgroup's n further, and in the pinhole-matched subset the direction of the effect wasn't even consistent across regions — a sign of noise dominating, not a real region-specific effect.
- Net result: the data can describe a direction and rough magnitude, but cannot support a statistical claim about whether training condition affects colocalization.

### Conclusions about the pipeline (the genuine, reusable output of this exercise)

1. **Local (in-MB) thresholding is only valid for MB-restricted markers.** Restricting a channel's percentile threshold to `stack[MB_mask]` is correct only when that channel's true positive population is entirely contained in the ROI — true for Mito here (MB-driver-restricted labeling), false for BRP and HSP70 (both expressed throughout the brain). Applying it to a broadly-expressed marker forces a roughly fixed X% of the ROI to be called "positive" regardless of real signal level, destroying the biological variation you're trying to measure — the same flattening effect already visible in `prop_MB_Mito` sitting nearly constant (~0.15–0.17) across every sample in this dataset.
2. **Deconvolution reliability must be checked per acquisition condition, not assumed constant.** Tabulate the existing flagged-fraction QC metric by session before trusting deconvolved output; it can vary from ~0% to ~100% between sessions using the identical pipeline code.
3. **When deconvolution reliability is in doubt, prefer raw images for mask/threshold-based analysis.** Validate this choice using any available natural-experiment pairs (duplicate acquisitions, bit-depth pairs) — if raw data reproduces far better than deconvolved data on a known duplicate, that's direct evidence to trust raw over deconvolved until the PSF is properly validated (e.g. bead calibration).
4. **Use accidental duplicate acquisitions as a free noise-floor calibration.** If a dataset contains even one, compare it against the between-animal spread you're trying to interpret — if the technical noise floor is comparable to or larger than the apparent biological effect, no amount of statistics on the existing data will resolve the question.
5. **Never assume what a naming convention or scene-name suffix means — check the metadata directly, or ask.** A wrong guess here (in this case, assuming a numeric suffix meant zoom level when it meant bit depth) can send an entire analysis down the wrong path.

### Statistical power for small-n comparisons

Before reporting any group comparison from a small pilot dataset, check explicitly whether the sample size can, even in principle, reach significance:

```python
from scipy.stats import mannwhitneyu
u, p = mannwhitneyu(group_a, group_b, alternative='two-sided')
# with n=2 vs n=2, the minimum achievable p-value is ~0.33 regardless of how
# cleanly separated the groups are — the test is structurally underpowered,
# not just "not significant this time"
```

A rough forward-looking sample size estimate, from an observed (or assumed) effect size:

```python
import numpy as np
log_a, log_b = np.log(group_a), np.log(group_b)   # use log scale for OR — see pooling section above
pooled_sd = np.sqrt(((len(log_a)-1)*log_a.var(ddof=1) + (len(log_b)-1)*log_b.var(ddof=1))
                     / (len(log_a)+len(log_b)-2))
d = (log_b.mean() - log_a.mean()) / pooled_sd      # Cohen's d, log scale
n_per_group = 2 * ((1.96 + 0.84) / d) ** 2         # 80% power, alpha=0.05, two-sided
```

Treat the resulting n as a rough order-of-magnitude planning number, not a precise target — an effect size estimated from 2-4 animals per group is itself highly unstable. Compute it for a range of plausible effect sizes (the observed one, plus more conservative medium/small ones) rather than a single number.

**A related pitfall: splitting an already-small comparison into subgroups (e.g. by anatomical subregion) makes it worse, not better.** Each subgroup inherits the parent comparison's small n, and subgroup-level "patterns" (e.g. one region showing a bigger apparent gap than another) are usually just which animals happened to land in which subgroup, not a real region-specific effect — especially telling if the direction of the effect isn't even consistent across subgroups in a more tightly-matched subset of the data. If subregion effects matter to the biological question, they belong in the experimental design (e.g. as a within-animal repeated factor, since each animal already contributes multiple regions) rather than as a post-hoc split of an underpowered top-level comparison.

### Reporting a confounded, underpowered comparison (e.g. for a thesis)

When the biological question cannot be answered from the available data — due to confounding, small n, or both — the honest and still-valuable framing is:

1. **State plainly that the comparison is not statistically supportable**, and show why (confound structure, achievable power/significance at the current n).
2. **Report the observed direction and magnitude as a preliminary/descriptive observation**, explicitly labeled as such, not as a conclusion.
3. **Present the methodological findings as the actual contribution**: a validated pipeline, and concretely characterized sources of measurement error (acquisition confounds, deconvolution reliability, reproducibility floor) — these are real, demonstrable results in their own right.
4. **Use the pilot data to design the properly powered follow-up** (matched acquisition settings across conditions, interleaved sessions, deliberate technical replicates, a sample size informed by the power estimate above).

Example framing:
> "This work establishes a validated pipeline for quantifying [X] colocalization, and identifies [specific confound] as the dominant source of measurement uncertainty in the current dataset. Preliminary data suggest [direction/magnitude], but this cannot be confirmed given [confounding structure; n insufficient for power, estimated at N≈X-Y per group]. These findings directly inform the design of a follow-up study with acquisition parameters held constant across conditions and adequate replication."

Finding and clearly explaining why a plausible-looking effect isn't yet trustworthy is a genuine research skill and a defensible thesis contribution — often more so than an unvalidated positive result.

---

## Future improvements

### Bead-based PSF calibration

The deconvolution pipeline currently relies on a theoretical Gaussian PSF (`σ_xy = 0.21λ/NA`, `σ_z = 0.66λn/NA²`) plus an approximate pinhole correction factor, not a measured PSF. Given the deconvolution reliability problems found in the case study above (concentrated specifically in the ~1.0 AU sessions), bead calibration would let the pinhole correction be replaced with an empirically measured value instead of the current approximation.

**Bead size: 100 nm diameter.** The rule of thumb is that bead diameter should be well below (roughly ≤1/3–1/2 of) the system's diffraction-limited resolution, so the bead approximates a true point source rather than a resolvable object whose own size blurs the measurement. With NA=1.4 and the shortest-wavelength channel here (Alexa488, 519 nm), the theoretical lateral PSF is `σ_xy ≈ 78 nm`, FWHM ≈ 183 nm — a 100 nm bead sits comfortably below that, contributing only a small, usually-ignorable broadening. Smaller beads (e.g. 40 nm) get closer to a true point source but are dimmer, trading calibration accuracy for worse SNR in the calibration measurement itself. 100 nm is the standard compromise used in commercial calibration bead kits.

**Practical recommendations:**
- Use **TetraSpeck beads** (100 nm, four-color) rather than single-color beads — lets you measure the PSF in all three channels (Alexa488/BRP, Cy5/HSP70, Alexa546/Mito) from one prep, and doubles as a chromatic-registration check across channels.
- Dilute enough to get well-isolated single beads; a cluster of beads will masquerade as an artificially larger/asymmetric "PSF."
- Image with the **same objective, immersion oil, and pinhole setting** used for real acquisitions — PSF depends on pinhole, and this project now has two distinct pinhole settings in use across sessions (0.209 AU and ~1.0 AU), so **calibrate at both**, not just one.
- Sample finely: XY pixel size well below Nyquist for the PSF (e.g. ~30–40 nm/px) and Z-steps of ~50–100 nm, so the bead's 3D profile is actually resolved, not just detected.
- Fit a 3D Gaussian (or the actual RL PSF model) to the resulting bead stack to extract empirical `sigma_xy_px`/`sigma_z_px` directly, and compare against what the formula + pinhole-factor currently predicts. This would tell definitively whether the ~1.0 AU sessions' deconvolution instability (found in the case study above) is a PSF-mismatch problem or something else.
