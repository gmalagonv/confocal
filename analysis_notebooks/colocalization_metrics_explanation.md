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
