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

## Pooling data across replicates

### Experimental structure

Measurements are nested: multiple brain regions within each brain, multiple brains per condition. This hierarchy determines what is independent and what is not.

| Comparison | Independent? |
|---|---|
| Brain 1 alpha vs Brain 2 alpha | Yes — different animals |
| Left alpha vs Right alpha within the same brain | No — same animal, same staining |
| Alpha vs gamma within the same brain | No — same animal, but answers a different biological question |

**Rule:** left and right hemisphere versions of the same region are pseudoreplicates — average them within each brain first, then treat each brain as one data point. Different region types (alpha, gamma, alpha prime) are kept separate; they answer different questions and can be compared as a region effect.

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
