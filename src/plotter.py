import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import shapiro, levene, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations


def plot_bars_with_sem3_test(groups, labels=None, ylabel="Value", title= None, figsize=(3,6),
                       bar_color="lightgray", dot_color="black", spine_width=2, legend=False):
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
           color=bar_colors, edgecolor="black", linewidth=1.5)

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
       handles = [Patch(facecolor=c, edgecolor="black", label=l)
                  for c, l in zip(bar_colors, labels)]
       ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15),
                 ncol=1)
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
      ax.set_title(title)

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




################################################


def fast_plotter(dates,  df=None, figsize=(6,6), ylabel="Performance Index", title=None, bar_color="lightgray", labels = [], quietStats=False, legend=False):
    

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

    sig_pairs = []
    if pass_normality and pass_similar_var:
      anova_test(vals_arrays, quiet=quietStats)
      ETA_SQUARED(vals_arrays, quiet=quietStats)

      tukey = tukey_test(vals_arrays, dates, quiet=quietStats)
      cohens_all(vals_arrays, dates, quiet=quietStats)
      sig_pairs = tukey_significant_pairs(tukey, dates)



    # PLOT ######################################
    # updated version
    if len(labels) == 0:
       labels = dates
    
    ax = plot_bars_with_sem3_test(
        vals_arrays,
        labels=labels,
        ylabel=ylabel,
        title = title, 
        bar_color=bar_color,
        figsize=figsize,
        legend=legend,
    )

    data_max = max(v.max() for v in vals_arrays)
    bar_step = 0.04
    for k, (i1, i2, p_adj) in enumerate(sig_pairs):
        y = data_max + 0.04 + k * bar_step
        add_sig_bar(ax, i1, i2, y, 0.015, p_to_stars(p_adj))

    data_min = min(v.min() for v in vals_arrays)
    bottom = data_min - 0.05 if data_min < 0 else 0
    top = data_max + 0.1 + bar_step * len(sig_pairs)
    ax.set_ylim(bottom, top)
    ax.tick_params(axis="both", labelsize=14)
    ax.yaxis.label.set_size(16)

    plt.show()
