import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import (
    shapiro, levene, f_oneway,
    mannwhitneyu, wilcoxon, kruskal, friedmanchisquare, rankdata, ttest_rel, ttest_ind,
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations


def plot_bars_with_sem3_test(groups, labels=None, ylabel="Value", title= None, title_font_size=14, figsize=(3,6),
                       bar_color="lightgray", pattern='', dot_color="black", spine_width=2, legend=False, lengend_font_size=14):
    """
    Plot multiple groups as bars with SEM and overlay individual data points.

    Parameters
    ----------
    groups : list of array-like
        List of numeric arrays/lists, one per group.
    labels : list of str, optional
        Labels for each group (x-axis).
    ylabel : str, optional
        Label for y-axis.
    figsize : tuple, optional
        (width, height) of the figure in inches.
    bar_color : str, optional
        Color of the bars.
    pattern : str or list of str, optional
        Hatch pattern(s) for the bars (e.g. '', '///'). Single string applies to all
        bars; a list is repeated to match n_groups the same way bar_color is, mirroring
        `plot_bars_with_sem3`.
    dot_color : str, optional
        Color of the overlaid dots.
    spine_width : float, optional
        Thickness of the axis spines.
    """

    groups = [np.asarray(g) for g in groups]
    n_groups = len(groups)

    # Allow single color or list of colors
    if isinstance(bar_color, (list, tuple, np.ndarray)):
        if len(bar_color) != n_groups:
            raise ValueError("Length of bar_color must match number of groups")
        bar_colors = bar_color
    else:
        bar_colors = [bar_color] * n_groups

    # Allow single pattern or list of patterns (same repeat-to-fit convention as
    # plot_bars_with_sem3's `pattern` handling)
    if isinstance(pattern, (list, tuple, np.ndarray)):
        patterns = list(pattern)
        if len(patterns) != n_groups:
            if len(patterns) == 0:
                patterns = [''] * n_groups
            else:
                num_repeats = n_groups / len(patterns)
                patterns = patterns * int(num_repeats)
    else:
        patterns = [pattern] * n_groups

    means = [np.nanmean(g) for g in groups]
    sems  = [np.nanstd(g, ddof=1) / np.sqrt(np.sum(~np.isnan(g))) for g in groups]


    if labels is None:
        labels = [f"Group {i+1}" for i in range(n_groups)]
    for i,lbl in enumerate(labels):
        print(f"group: {lbl}, mean: {means[i]}, sem: {sems[i]}")

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_groups)

    # Bars + SEM
    ax.bar(x, means, yerr=sems, capsize=8,
           color=bar_colors, edgecolor="black", hatch=patterns, linewidth=1.5)

    # Overlay dots
    for i, vals in enumerate(groups):
        # jitter = (np.random.rand(len(vals)) - 0.5) * 0.2
        jitter = 0
        # ax.scatter(np.full(len(vals), x[i]) + jitter, vals,
        #            color=dot_color, s=40, alpha=0.8, zorder=3)
        ax.scatter(np.full(len(vals), x[i]) + jitter, vals,
           facecolors='none', edgecolors=dot_color, linewidths=1.2, s=40, zorder=3)

    # Axis labels & limits
    if len(labels[0]) >= 20:
       angle = 45
    elif len(labels[0]) >= 15 and len(labels[0]) < 20:
       angle = 30

    else:
       angle = 0
    
    ax.set_xticks(x)
    if legend:
       ax.set_xticklabels([""] * n_groups)
       handles = [Patch(facecolor=c, edgecolor="black", hatch=p, label=l)
                  for c, p, l in zip(bar_colors, patterns, labels)]
       ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05),
                 ncol=2,fontsize=lengend_font_size)
    else:
       ax.set_xticklabels(labels, rotation=angle)

    ax.set_ylabel(ylabel)

    # --- Style adjustments ---
    # Thicker left & bottom spines, remove top/right
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(spine_width)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Ticks only on left/bottom
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if title is not None:
      ax.set_title(title, fontsize=title_font_size, pad=30)


    # Add baseline at y = 0
    ax.axhline(0, color="black", linewidth=1.2)

    plt.tight_layout()
    return ax

def add_sig_bar(ax, x1, x2, y, h, text, fontsize=16):
    ax.plot([x1, x1, x2, x2],
            [y, y+h, y+h, y],
            lw=1.5, color="black")
    ax.text((x1+x2)/2, y+h, text,
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold")
    



###### STATS #############



def shapiro_wilk(list_arrays, list_labels, alpha=0.05, quiet=False, min_n=3):
  all_normal = True
  for name, group in zip(list_labels, list_arrays):
    if len(group) < min_n:
      if not quiet:
        print(f"{name}")
        print(f"  Skipped: only {len(group)} sample(s) (need >= {min_n} for Shapiro-Wilk)\n")
      continue

    stat, p = shapiro(group)
    if not quiet:
      print(f"{name}")
      print(f"  W statistic = {stat:.4f}")
      print(f"  p-value     = {p:.4f}")

    if p > alpha:
        if not quiet:
          print("  → Data look approximately normal\n")
    else:
        if not quiet:
          print("  → Significant deviation from normality\n")
        all_normal = False

  return all_normal


def levene_and_brown_forsythe_test(list_arrays, test2run=0, quiet=False):
  test = [["Brown-Forsythe test", "median"], ["Levene's test", "mean"]]

  pass_similar_var = False
  if test2run < 2:
    test = [test[test2run]]


  for i in range(len(test)):
    stat, p_val = levene(*list_arrays, center=test[i][1])
    if not quiet:
      print(f'{test[i][0]}')
      print(f"Statistic = {stat:.4f}")
      print(f"p-value   = {p_val:.4f}")

    if p_val > 0.05:
      if not quiet:
        print("→ Variances are not significantly different\n")
      if test2run < 2:
        pass_similar_var = True

    else:
      if not quiet:
        print("→ Variances differ significantly\n")

  return pass_similar_var


def anova_test(list_arrays, quiet=False):
  f_stat, p_value = f_oneway(*list_arrays)
  if not quiet:
    print("One-way ANOVA")
    print(f"F statistic = {f_stat:.4f}")
    print(f"p-value     = {p_value:.4f}")

  if p_value < 0.05:
      if not quiet:
        print("→ Significant differences exist between groups\n")
      pass_anova = True
  else:
      if not quiet:
        print("→ No significant differences detected\n")
      pass_anova = False
  return pass_anova


def ETA_SQUARED(list_arrays, quiet=False):
  # Flatten data
  all_data = np.concatenate(list_arrays)

  # Grand mean
  grand_mean = np.mean(all_data)

  # Between-group sum of squares
  ss_between = sum(
      len(group) * (np.mean(group) - grand_mean)**2
      for group in list_arrays
  )

  # Total sum of squares
  ss_total = sum(
      (x - grand_mean)**2
      for x in all_data
  )

  # Eta squared
  eta_squared = ss_between / ss_total

  if not quiet:
    print("\nEffect size")
    print(f"Eta squared (η²) = {eta_squared:.4f}")


  # Interpretation guideline
  if eta_squared < 0.01:
      interpretation = "very small"
  elif eta_squared < 0.06:
      interpretation = "small"
  elif eta_squared < 0.14:
      interpretation = "medium"
  else:
      interpretation = "large"
  if not quiet:
    print(f"Effect size interpretation: {interpretation}")


def tukey_test(list_arrays, list_labels, quiet=False):

  # Flatten all values into one array
  all_data = np.concatenate(list_arrays)

  # Create matching labels
  labels = np.concatenate([
      np.repeat(name, len(group))
      for name, group in zip(list_labels, list_arrays)
  ])

  # Run Tukey HSD
  tukey = pairwise_tukeyhsd(
      endog=all_data,
      groups=labels,
      alpha=0.05
  )
  if not quiet:
    print(tukey)
  return tukey


def p_to_stars(p_adj):
  if p_adj < 0.001:
    return "***"
  elif p_adj < 0.01:
    return "**"
  elif p_adj < 0.05:
    return "*"
  return "ns"


def tukey_significant_pairs(tukey, list_labels):
  """Map Tukey's (possibly re-sorted) group labels back to the original
  list_labels order, so bar positions in fast_plotter stay correct
  regardless of how many groups are being compared."""
  label_to_index = {str(name): i for i, name in enumerate(list_labels)}
  pairs = []
  for row in tukey.summary().data[1:]:
    group1, group2, meandiff, p_adj, lower, upper, reject = row
    if reject:
      i1, i2 = label_to_index[str(group1)], label_to_index[str(group2)]
      pairs.append((min(i1, i2), max(i1, i2), p_adj))
  return sorted(pairs, key=lambda p: p[1] - p[0])


def cohens_d(x, y):
  nx = len(x)
  ny = len(y)

    # pooled standard deviation
  pooled_sd = np.sqrt(
      ((nx - 1) * np.var(x, ddof=1) +
        (ny - 1) * np.var(y, ddof=1))
      / (nx + ny - 2)
  )

  d = (np.mean(x) - np.mean(y)) / pooled_sd
  return d


def cohens_all(list_arrays, list_labels, quiet=False):
  for (i, g1), (j, g2) in combinations(enumerate(list_arrays), 2):

    d = cohens_d(g1, g2)

    # interpretation
    abs_d = abs(d)

    if abs_d < 0.2:
        size = "very small"
    elif abs_d < 0.5:
        size = "small"
    elif abs_d < 0.8:
        size = "medium"
    else:
        size = "large"

    if not quiet:
      print(f"{list_labels[i]} vs {list_labels[j]}")
      print(f"  Cohen's d = {d:.4f}")
      print(f"  Effect size: {size}\n")


###### NON-PARAMETRIC STATS #############
# Used in place of the ANOVA/Tukey/Cohen's-d block above when shapiro_wilk flags
# non-normal data. `paired=True` means the groups are repeated measurements on the
# same subjects (e.g. same brains' syn vs non-syn fraction) rather than independent
# samples, which determines whether the signed-rank or rank-sum family is valid.

def mannwhitneyu_test(list_arrays, quiet=False):
  """Non-parametric alternative to an independent two-sample t-test."""
  stat, p_value = mannwhitneyu(list_arrays[0], list_arrays[1], alternative="two-sided")
  if not quiet:
    print("Mann-Whitney U test")
    print(f"U statistic = {stat:.4f}")
    print(f"p-value     = {p_value:.4f}")

  significant = p_value < 0.05
  if not quiet:
    print("→ Significant difference between groups\n" if significant
          else "→ No significant difference detected\n")
  return significant, p_value


def wilcoxon_test(list_arrays, quiet=False):
  """Non-parametric alternative to a paired t-test."""
  stat, p_value = wilcoxon(list_arrays[0], list_arrays[1])
  if not quiet:
    print("Wilcoxon signed-rank test")
    print(f"W statistic = {stat:.4f}")
    print(f"p-value     = {p_value:.4f}")

  significant = p_value < 0.05
  if not quiet:
    print("→ Significant difference between groups\n" if significant
          else "→ No significant difference detected\n")
  return significant, p_value


def paired_ttest_test(list_arrays, quiet=False):
  """Parametric paired test (2 groups, same subjects) -- used instead of wilcoxon_test
  when the differences pass the normality check, same way anova_test is preferred over
  kruskal_test for independent groups that pass normality."""
  stat, p_value = ttest_rel(list_arrays[0], list_arrays[1])
  if not quiet:
    print("Paired t-test")
    print(f"t statistic = {stat:.4f}")
    print(f"p-value     = {p_value:.4f}")

  significant = p_value < 0.05
  if not quiet:
    print("→ Significant difference between groups\n" if significant
          else "→ No significant difference detected\n")
  return significant, p_value


def unpaired_ttest_test(list_arrays, quiet=False):
  """Parametric independent-samples test (2 groups) -- used instead of mannwhitneyu_test
  when both groups pass normality (and variance homogeneity). Gives the same p-value as
  a 2-group one-way ANOVA (f_oneway), just without the Tukey/eta-squared machinery that
  only makes sense for >2 groups."""
  stat, p_value = ttest_ind(list_arrays[0], list_arrays[1])
  if not quiet:
    print("Independent-samples t-test")
    print(f"t statistic = {stat:.4f}")
    print(f"p-value     = {p_value:.4f}")

  significant = p_value < 0.05
  if not quiet:
    print("→ Significant difference between groups\n" if significant
          else "→ No significant difference detected\n")
  return significant, p_value


def two_group_test(list_arrays, labels=None, paired=False, quiet=False):
  """Shared entry point for 'compare exactly 2 groups, correctly' -- picks the test the
  same way fast_plotter's own 2-group branch does (normality -> parametric, else
  non-parametric; paired vs unpaired selects which test family), so any caller doing a
  standalone 2-group comparison (e.g. one leg of a mixed paired/unpaired multi-group
  design that fast_plotter itself can't express) gets identical logic instead of an
  independently-hardcoded, potentially inconsistent choice. Returns (significant,
  p_value).
  """
  if labels is None:
    labels = ['group1', 'group2']
  pass_normality = shapiro_wilk(list_arrays, labels, quiet=quiet)

  if paired:
    if pass_normality:
      return paired_ttest_test(list_arrays, quiet=quiet)
    return wilcoxon_test(list_arrays, quiet=quiet)

  pass_similar_var = levene_and_brown_forsythe_test(list_arrays, quiet=quiet)
  if pass_normality and pass_similar_var:
    return unpaired_ttest_test(list_arrays, quiet=quiet)
  return mannwhitneyu_test(list_arrays, quiet=quiet)


def kruskal_test(list_arrays, quiet=False):
  """Non-parametric alternative to one-way ANOVA (>2 independent groups)."""
  stat, p_value = kruskal(*list_arrays)
  if not quiet:
    print("Kruskal-Wallis test")
    print(f"H statistic = {stat:.4f}")
    print(f"p-value     = {p_value:.4f}")

  significant = p_value < 0.05
  if not quiet:
    print("→ Significant differences exist between groups\n" if significant
          else "→ No significant differences detected\n")
  return significant


def friedman_test(list_arrays, quiet=False):
  """Non-parametric alternative to repeated-measures ANOVA (>2 paired groups)."""
  stat, p_value = friedmanchisquare(*list_arrays)
  if not quiet:
    print("Friedman test")
    print(f"Chi-square statistic = {stat:.4f}")
    print(f"p-value               = {p_value:.4f}")

  significant = p_value < 0.05
  if not quiet:
    print("→ Significant differences exist between groups\n" if significant
          else "→ No significant differences detected\n")
  return significant


def rank_biserial_effect_size(x, y, quiet=False, label=""):
  """Effect size for Mann-Whitney U (non-parametric analogue of Cohen's d)."""
  u_stat, _ = mannwhitneyu(x, y, alternative="two-sided")
  n1, n2 = len(x), len(y)
  r = 1 - (2 * u_stat) / (n1 * n2)

  abs_r = abs(r)
  if abs_r < 0.1:
    size = "very small"
  elif abs_r < 0.3:
    size = "small"
  elif abs_r < 0.5:
    size = "medium"
  else:
    size = "large"

  if not quiet:
    prefix = f"{label}\n" if label else ""
    print(f"{prefix}  Rank-biserial r = {r:.4f}")
    print(f"  Effect size: {size}\n")
  return r


def matched_pairs_rank_biserial(x, y, quiet=False, label=""):
  """Effect size for Wilcoxon signed-rank (non-parametric analogue of Cohen's d)."""
  diffs = np.asarray(x) - np.asarray(y)
  diffs = diffs[diffs != 0]

  ranks = rankdata(np.abs(diffs))
  r_plus = ranks[diffs > 0].sum()
  r_minus = ranks[diffs < 0].sum()
  r = (r_plus - r_minus) / ranks.sum()

  abs_r = abs(r)
  if abs_r < 0.1:
    size = "very small"
  elif abs_r < 0.3:
    size = "small"
  elif abs_r < 0.5:
    size = "medium"
  else:
    size = "large"

  if not quiet:
    prefix = f"{label}\n" if label else ""
    print(f"{prefix}  Matched-pairs rank-biserial r = {r:.4f}")
    print(f"  Effect size: {size}\n")
  return r


def pairwise_nonparametric_posthoc(list_arrays, list_labels, paired=False, quiet=False):
  """Post-hoc pairwise comparisons for >2 groups, substituting for Tukey HSD.

  Uses Wilcoxon (paired) or Mann-Whitney U (independent) per pair with
  Holm-Bonferroni correction for multiple comparisons (no extra dependency
  needed, unlike Dunn's test which would require scikit-posthocs).
  """
  pairs = list(combinations(range(len(list_arrays)), 2))
  raw_pvals = []
  for i, j in pairs:
    if paired:
      _, p = wilcoxon(list_arrays[i], list_arrays[j])
    else:
      _, p = mannwhitneyu(list_arrays[i], list_arrays[j], alternative="two-sided")
    raw_pvals.append(p)

  # Holm-Bonferroni step-down correction
  m = len(raw_pvals)
  order = np.argsort(raw_pvals)
  adj_pvals = [None] * m
  running_max = 0.0
  for rank, idx in enumerate(order):
    adj = min(raw_pvals[idx] * (m - rank), 1.0)
    running_max = max(running_max, adj)
    adj_pvals[idx] = running_max

  sig_pairs = []
  for (i, j), p_adj in zip(pairs, adj_pvals):
    reject = p_adj < 0.05
    if not quiet:
      print(f"{list_labels[i]} vs {list_labels[j]}: p_adj = {p_adj:.4f} "
            f"{'*' if reject else 'ns'}")
    if reject:
      sig_pairs.append((min(i, j), max(i, j), p_adj))

  return sorted(sig_pairs, key=lambda p: p[1] - p[0])


################################################


def fast_plotter(dates,  df=None, figsize=(6,6), ylabel="Performance Index", ylim=None, title=None, title_font_size=14, bar_color="lightgray", pattern='', labels = [], quietStats=False, legend=False, lengend_font_size=14,  paired=False):
    

    vals_arrays = []

    if df is None:
      sheet_url = "https://docs.google.com/spreadsheets/d/1YpXxk7YYIIJh5XcIkcv7oECVNmW58WHS_IbCw-qfRhc/export?format=csv&gid=1997495720"
      
      df = pd.read_csv(sheet_url, decimal=",")
    else:
      df = df.T
      df = df.reset_index() 
      df = df.rename(columns={'index': 'date'})
    #print(df)
    for date in dates:
        
        vals = df[df["date"] == date].drop(columns=["date"])
        #print(date, '<------------------')
        #print(vals)
        #print(f'num columns: {vals.shape[1]}, num rows: {vals.shape[0]}, not nans: {vals.notna().sum().sum()}')
      # vals_array = vals.dropna().values.flatten().tolist()

        # Convert all remaining columns to numeric, coercing errors to NaN
        for col in vals.columns:
          #for idx, val in enumerate(vals[col]):
          # print(f"Column: {col}, Row: {idx}, Value: {val}, Type: {type(val)}")
          vals[col] = pd.to_numeric(vals[col], errors="coerce")

        #print(f'AFTER: num columns: {vals.shape[1]}, num rows: {vals.shape[0]}, not nans: {vals.notna().sum().sum()}')

        vals_array = vals.values.flatten()
        vals_array = vals_array[~np.isnan(vals_array)]

        #print(date, vals_array, type(vals_array))


        vals_arrays.append(vals_array)

    # STATS ######################################
    if not quietStats:
      print('\n---------- STATS ----------')
    pass_normality = shapiro_wilk(vals_arrays, dates, quiet=quietStats)

    if len(dates) > 1:
      pass_similar_var = levene_and_brown_forsythe_test(vals_arrays, quiet=quietStats)
    else:
      pass_similar_var = False

    # `paired` must gate which test family runs BEFORE the normality check, not only
    # as a fallback once normality/variance-homogeneity fails. Previously, any data
    # that happened to pass both checks fell into the `pass_normality and
    # pass_similar_var` branch unconditionally and ran an unpaired one-way ANOVA even
    # when the caller explicitly passed paired=True -- silently discarding known
    # within-subject pairing (e.g. two compartments measured in the same animal) and
    # understating significance, since an unpaired test can't cancel out
    # between-subject variability the way a paired one does. Normality now only
    # chooses parametric vs non-parametric *within* the paired/unpaired path the
    # caller actually asked for.
    sig_pairs = []
    if paired:
      if len(dates) == 2:
        # two_group_test re-runs shapiro_wilk internally (already computed above as
        # pass_normality) so its printed output stays self-contained when called
        # standalone elsewhere -- a second, identical call here is cheap and keeps
        # this branch as a thin wrapper around the single shared implementation
        # rather than a second copy of the paired-test-selection logic to drift.
        significant, p_value = two_group_test(vals_arrays, labels=dates, paired=True, quiet=quietStats)
        matched_pairs_rank_biserial(vals_arrays[0], vals_arrays[1], quiet=quietStats)

        if significant:
          sig_pairs = [(0, 1, p_value)]

      elif len(dates) > 2:
        # Repeated-measures ANOVA isn't implemented here; Friedman (non-parametric) is
        # used for paired designs with >2 groups regardless of normality -- still a
        # valid test when the data is normal, just somewhat less powerful than a true
        # RM-ANOVA would be. Preferable to silently running an unpaired test.
        significant = friedman_test(vals_arrays, quiet=quietStats)
        if significant:
          sig_pairs = pairwise_nonparametric_posthoc(vals_arrays, dates, paired=paired, quiet=quietStats)

    else:
      if pass_normality and pass_similar_var:
        anova_test(vals_arrays, quiet=quietStats)
        ETA_SQUARED(vals_arrays, quiet=quietStats)

        tukey = tukey_test(vals_arrays, dates, quiet=quietStats)
        cohens_all(vals_arrays, dates, quiet=quietStats)
        sig_pairs = tukey_significant_pairs(tukey, dates)

      elif len(dates) == 2:
        significant, p_value = mannwhitneyu_test(vals_arrays, quiet=quietStats)
        rank_biserial_effect_size(vals_arrays[0], vals_arrays[1], quiet=quietStats)

        if significant:
          sig_pairs = [(0, 1, p_value)]

      elif len(dates) > 2:
        significant = kruskal_test(vals_arrays, quiet=quietStats)
        if significant:
          sig_pairs = pairwise_nonparametric_posthoc(vals_arrays, dates, paired=paired, quiet=quietStats)



    # PLOT ######################################
    # updated version
    if len(labels) == 0:
       labels = dates
    
    ax = plot_bars_with_sem3_test(
        vals_arrays,
        labels=labels,
        ylabel=ylabel,
        title = title,
        title_font_size = title_font_size,
        bar_color=bar_color,
        pattern=pattern,
        figsize=figsize,
        legend=legend,
        lengend_font_size = lengend_font_size
    )

    data_max = max(v.max() for v in vals_arrays)
    data_min = min(v.min() for v in vals_arrays)
    # Significance-bracket geometry, scaled to the data range. These offsets/heights were
    # hardcoded for ~0-1-scale data (+0.04 above data_max, 0.04 step, 0.015 bracket height,
    # +0.1 headroom); on any larger y-scale (raw intensities, counts, ...) the brackets landed
    # right on top of the tallest bar and the axes had no room above them, so the stars were
    # hidden behind the title. Now everything is a fraction of the plotted span.
    span = (data_max - min(data_min, 0.0)) or 1.0
    first_off = 0.08 * span      # gap between the tallest bar and the first bracket
    bar_step  = 0.09 * span      # vertical spacing between stacked brackets
    bar_h     = 0.025 * span     # bracket "tick" height
    for k, (i1, i2, p_adj) in enumerate(sig_pairs):
        y = data_max + first_off + k * bar_step
        add_sig_bar(ax, i1, i2, y, bar_h, p_to_stars(p_adj))

    if ylim == None:
      bottom = data_min - 0.05 * span if data_min < 0 else 0
      n_bars = max(len(sig_pairs), 1)
      top = data_max + first_off + (n_bars - 1) * bar_step + bar_h + 0.14 * span
      ax.set_ylim(bottom, top)
    else:
      ax.set_ylim(ylim[0], ylim[1])
    ax.tick_params(axis="both", labelsize=14)
    ax.yaxis.label.set_size(16)

    plt.show()
