import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools
from scipy.stats import skew, kurtosis

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

sns.set(style="whitegrid", font_scale=1.1)

PLOT_DIR = "plots/univariate"
PAIR_DIR = "plots/pairwise"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(PAIR_DIR, exist_ok=True)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("debutanizer_full.csv")

print("Initial shape:", df.shape)

df.columns = df.columns.str.strip()
df = df.apply(pd.to_numeric, errors="coerce")

print("\nNaNs after type coercion:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ---------------------------------------------------------
# REMOVE CONSTANT COLUMNS
# ---------------------------------------------------------

variance = df.var()
constant_cols = variance[variance == 0].index.tolist()

print("\nDropping constant columns:", constant_cols)
df = df.drop(columns=constant_cols)

# ---------------------------------------------------------
# DROP NaNs
# ---------------------------------------------------------

initial_rows = df.shape[0]
df = df.dropna()
final_rows = df.shape[0]

print(f"\nDropped {initial_rows - final_rows} rows due to NaNs")
print("Final shape:", df.shape)

# ---------------------------------------------------------
# UNIVARIATE STATISTICS
# ---------------------------------------------------------

numeric_cols = df.columns

univariate_stats = pd.DataFrame(index=numeric_cols)

univariate_stats["mean"] = df.mean()
univariate_stats["median"] = df.median()
univariate_stats["std"] = df.std()
univariate_stats["variance"] = df.var()
univariate_stats["min"] = df.min()
univariate_stats["max"] = df.max()
univariate_stats["skewness"] = df.apply(skew)
univariate_stats["kurtosis"] = df.apply(kurtosis)

univariate_stats = univariate_stats.sort_values("variance", ascending=False)

print("\nUnivariate Statistics:")
print(univariate_stats.round(4))

univariate_stats.to_csv(f"{PLOT_DIR}/univariate_statistics.csv")

# ---------------------------------------------------------
# UNIVARIATE PLOTS
# ---------------------------------------------------------

for col in numeric_cols:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.histplot(df[col], bins=30, kde=True, ax=axes[0])
    axes[0].set_title(f"Histogram: {col}")

    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f"Boxplot: {col}")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{col}_univariate.png", dpi=300)
    plt.close()

print("\nUnivariate plots saved.")

# ─────────────────────────────────────────────────────────────────────────────
# UNIVARIATE PLOTS — 4x2 grid (histogram + boxplot per variable)
# ─────────────────────────────────────────────────────────────────────────────
numeric_cols = list(df.columns)  # [u1, u2, u3, u4, u5, u6, u7, y]

fig, axes = plt.subplots(
    nrows=len(numeric_cols),  # 8 rows, one per variable
    ncols=2,                  # 2 columns: histogram | boxplot
    figsize=(14, 4 * len(numeric_cols))
)

for i, col in enumerate(numeric_cols):
    # Left: Histogram with KDE
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i][0])
    axes[i][0].set_title(f"Histogram: {col}", fontsize=11)
    axes[i][0].set_xlabel(col)
    axes[i][0].set_ylabel("Count")

    # Right: Boxplot
    sns.boxplot(x=df[col], ax=axes[i][1])
    axes[i][1].set_title(f"Boxplot: {col}", fontsize=11)
    axes[i][1].set_xlabel(col)

fig.suptitle("Univariate Analysis — All Variables", fontsize=14, fontweight="bold", y=1.001)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/histograms_all.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: histograms_all.png")
# =========================================================
# STRIP-STYLE FULL PAIRWISE SCATTER PLOTS
# =========================================================

df_sample = df.sample(min(1200, len(df)), random_state=42)

all_cols = df.columns.tolist()
pairs = list(itertools.combinations(all_cols, 2))

print("\nGenerating strip-style full pairwise scatter plots...")

n_pairs = len(pairs)
n_cols = 3
n_rows = int(np.ceil(n_pairs / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()

for i, (x_var, y_var) in enumerate(pairs):
    sns.scatterplot(
        data=df_sample,
        x=x_var,
        y=y_var,
        ax=axes[i],
        alpha=0.4,
        s=18
    )
    axes[i].set_title(f"{x_var} vs {y_var}")

# Remove unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(f"{PAIR_DIR}/strip_full_pairwise.png", dpi=300)
plt.close()

print("Strip-style full pairwise scatter plot saved.")

# =========================================================
# TARGET STRIP PLOT (INPUTS → y)
# =========================================================

if "y" in df.columns:

    print("\nGenerating target strip plot (Inputs → y)...")

    input_cols = [col for col in df.columns if col != "y"]

    n_inputs = len(input_cols)
    fig, axes = plt.subplots(n_inputs, 1, figsize=(6, 4*n_inputs))

    if n_inputs == 1:
        axes = [axes]

    for i, col in enumerate(input_cols):
        sns.scatterplot(
            data=df_sample,
            x=col,
            y="y",
            ax=axes[i],
            alpha=0.5,
            s=20
        )
        axes[i].set_title(f"{col} vs y")

    plt.tight_layout()
    plt.savefig(f"{PAIR_DIR}/target_strip_plot.png", dpi=300)
    plt.close()

    print("Target strip plot saved.")

print("\nPairwise analysis complete.")