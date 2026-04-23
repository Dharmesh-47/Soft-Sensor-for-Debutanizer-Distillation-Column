"""
static_model_acf
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PATHS
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH  = "/mnt/user-data/uploads/1772699447991_debutanizer_full.csv"
OUT_DIR   = "plots/static model"

import os
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("debutanizer_full.csv")
df.columns = df.columns.str.strip()

print("Shape:", df.shape)
print("Columns:", list(df.columns))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE CONSTRUCTION
#     u_avg = (u6 + u7) / 2  →  6 inputs: u1, u2, u3, u4, u5, u_avg
#
#     If you want u6 and u7 as SEPARATE inputs instead, change the two lines
#     marked  [SEPARATE]  below:
#       [SEPARATE] comment out:  df["u_avg"] = (df["u6"] + df["u7"]) / 2
#       [SEPARATE] change INPUT_COLS to: ["u1","u2","u3","u4","u5","u6","u7"]
# ─────────────────────────────────────────────────────────────────────────────
df["u_avg"] = (df["u6"] + df["u7"]) / 2          # [SEPARATE] comment out
INPUT_COLS  = ["u1", "u2", "u3", "u4", "u5", "u_avg"]   # [SEPARATE] replace
TARGET_COL  = "y"

# ALL variable names (for ACF/CCF section) — keep original 8 columns
ALL_COLS = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "y"]

X = df[INPUT_COLS].values
y = df[TARGET_COL].values
N = len(df)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAIN / TEST SPLIT  (chronological, 75 / 25)
# ─────────────────────────────────────────────────────────────────────────────
split      = int(0.75 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  STATIC MLP MODEL
#     Architecture mirrors the paper's spirit but without time lags:
#       • 1 hidden layer, 13 neurons, sigmoid activation, L-BFGS solver
#       • 10 random restarts — keep best
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining static MLP (10 restarts)...")

best_model   = None
best_train_r2 = -np.inf

for seed in range(10):
    model = MLPRegressor(
        hidden_layer_sizes = (13,),
        activation         = "logistic",   # sigmoid
        solver             = "lbfgs",
        max_iter           = 2000,
        random_state       = seed,
        tol                = 1e-6,
    )
    model.fit(X_train, y_train)
    r2 = r2_score(y_train, model.predict(X_train))
    if r2 > best_train_r2:
        best_train_r2 = r2
        best_model    = model

model = best_model

# ── Metrics ──────────────────────────────────────────────────────────────────
y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

train_r2   = r2_score(y_train, y_train_pred)
test_r2    = r2_score(y_test,  y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse  = np.sqrt(mean_squared_error(y_test,  y_test_pred))
train_mae  = mean_absolute_error(y_train, y_train_pred)
test_mae   = mean_absolute_error(y_test,  y_test_pred)

print(f"\n{'─'*40}")
print(f"  STATIC MODEL RESULTS")
print(f"{'─'*40}")
print(f"  Train  R²   : {train_r2:.6f}")
print(f"  Test   R²   : {test_r2:.6f}")
print(f"  Train  RMSE : {train_rmse:.6f}")
print(f"  Test   RMSE : {test_rmse:.6f}")
print(f"  Train  MAE  : {train_mae:.6f}")
print(f"  Test   MAE  : {test_mae:.6f}")
print(f"{'─'*40}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  STATIC MODEL PLOTS
# ─────────────────────────────────────────────────────────────────────────────
residuals = y_test - y_test_pred
test_idx  = np.arange(len(y_test))

# ── Fig A: Actual vs Predicted overlay ───────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
axes[0].plot(test_idx, y_test,      color="#1B6CA8", lw=0.9, label="Actual")
axes[0].set_title("(a) Actual C4 Concentration", fontsize=10)
axes[0].set_ylabel("C4 (norm.)")
axes[0].legend(fontsize=8)

axes[1].plot(test_idx, y_test_pred, color="#E65100", lw=0.9, label="Static MLP estimate")
axes[1].set_title(f"(b) Static MLP Prediction  "
                  f"[RMSE={test_rmse:.4f}  R²={test_r2:.4f}]", fontsize=10)
axes[1].set_ylabel("C4 (norm.)")
axes[1].legend(fontsize=8)

axes[2].plot(test_idx, residuals, color="#C62828", lw=0.7)
axes[2].axhline(0, color="black", lw=1, ls="--")
axes[2].set_title("(c) Residual", fontsize=10)
axes[2].set_ylabel("Error (norm.)")
axes[2].set_xlabel("Sample index (test set)")

fig.suptitle("Static MLP — Actual vs Predicted (No Time Lags)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/static_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: static_actual_vs_predicted.png")

# ── Fig B: Online overlay ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(test_idx, y_test,      color="#1B6CA8", lw=1.0, label="Actual (GC)")
ax.plot(test_idx, y_test_pred, color="#E65100", lw=0.9,
        ls="--", label="Static MLP estimate")
ax.set_xlabel("Sample index (test set)")
ax.set_ylabel("C4 concentration (normalised)")
ax.set_title(f"Static MLP — On-line Performance  "
             f"[RMSE={test_rmse:.4f}  R²={test_r2:.4f}  MAE={test_mae:.4f}]",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/static_online_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: static_online_performance.png")

# ── Fig C: Metrics bar chart ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
metrics = {
    "RMSE": (train_rmse, test_rmse),
    "MAE":  (train_mae,  test_mae),
    "R²":   (train_r2,   test_r2),
}
colors = ["#1B6CA8", "#E65100"]
for ax, (name, (tr, te)) in zip(axes, metrics.items()):
    bars = ax.bar(["Train", "Test"], [tr, te], color=colors, width=0.5)
    for bar, val in zip(bars, [tr, te]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(tr, te) * 1.25)
fig.suptitle("Static MLP — Performance Metrics", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/static_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: static_metrics.png")

# ── Fig D: Scatter actual vs predicted ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_test, y_test_pred, alpha=0.3, s=8, color="#1B6CA8")
lims = [0, 1]
ax.plot(lims, lims, "k--", lw=1.2, label="Perfect fit")
ax.set_xlabel("Actual C4")
ax.set_ylabel("Predicted C4")
ax.set_title(f"Scatter  R²={test_r2:.4f}", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/static_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: static_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# LINEAR REGRESSION — same static inputs as MLP
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr  = lr_model.predict(X_test)

train_r2_lr   = r2_score(y_train, y_train_pred_lr)
test_r2_lr    = r2_score(y_test,  y_test_pred_lr)
train_rmse_lr = np.sqrt(mean_squared_error(y_train, y_train_pred_lr))
test_rmse_lr  = np.sqrt(mean_squared_error(y_test,  y_test_pred_lr))
train_mae_lr  = mean_absolute_error(y_train, y_train_pred_lr)
test_mae_lr   = mean_absolute_error(y_test,  y_test_pred_lr)

print(f"\n{'─'*40}")
print(f"  LINEAR REGRESSION RESULTS")
print(f"{'─'*40}")
print(f"  Train  R²   : {train_r2_lr:.6f}")
print(f"  Test   R²   : {test_r2_lr:.6f}")
print(f"  Train  RMSE : {train_rmse_lr:.6f}")
print(f"  Test   RMSE : {test_rmse_lr:.6f}")
print(f"  Train  MAE  : {train_mae_lr:.6f}")
print(f"  Test   MAE  : {test_mae_lr:.6f}")
print(f"{'─'*40}")
print(f"\n  Coefficients:")
for name, coef in zip(INPUT_COLS, lr_model.coef_):
    print(f"    {name:8s}: {coef:+.4f}")
print(f"  Intercept   : {lr_model.intercept_:+.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
residuals_lr = y_test - y_test_pred_lr
test_idx     = np.arange(len(y_test))

# Online performance
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(test_idx, y_test,         color="#1B6CA8", lw=1.0, label="Actual (GC)")
ax.plot(test_idx, y_test_pred_lr, color="#2CA02C", lw=0.9,
        ls="--", label="Linear Regression estimate")
ax.set_xlabel("Sample index (test set)")
ax.set_ylabel("C4 concentration (normalised)")
ax.set_title(f"Linear Regression — On-line Performance  "
             f"[RMSE={test_rmse_lr:.4f}  R²={test_r2_lr:.4f}  MAE={test_mae_lr:.4f}]",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/lr_online_performance.png", dpi=150, bbox_inches="tight")
plt.close()

# Scatter
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_test, y_test_pred_lr, alpha=0.3, s=8, color="#2CA02C")
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect fit")
ax.set_xlabel("Actual C4")
ax.set_ylabel("Predicted C4")
ax.set_title(f"LR Scatter  R²={test_r2_lr:.4f}", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/lr_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# Metrics bar chart
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
metrics_lr = {
    "RMSE": (train_rmse_lr, test_rmse_lr),
    "MAE":  (train_mae_lr,  test_mae_lr),
    "R²":   (train_r2_lr,   test_r2_lr),
}
colors = ["#1B6CA8", "#E65100"]
for ax, (name, (tr, te)) in zip(axes, metrics_lr.items()):
    bars = ax.bar(["Train", "Test"], [tr, te], color=colors, width=0.5)
    for bar, val in zip(bars, [tr, te]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(tr, te) * 1.25 if max(tr, te) > 0 else 0.1)
fig.suptitle("Linear Regression — Performance Metrics", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/lr_metrics.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved: lr_online_performance.png, lr_scatter.png, lr_metrics.png")
# ─────────────────────────────────────────────────────────────────────────────
# 6.  ACF / CCF HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
MAX_LAG = 15   # number of lags to compute

def compute_acf(x, max_lag=MAX_LAG):
    """
    Normalised autocorrelation of signal x at lags 0 … max_lag.
    R_xx(τ) = Σ (x(k) - μ)(x(k+τ) - μ)  /  Σ (x(k) - μ)²
    Using the mean-centred version (standard definition).
    """
    x  = np.asarray(x, dtype=float)
    x  = x - x.mean()
    n  = len(x)
    denom = np.dot(x, x)
    acf   = np.array([np.dot(x[:n-lag], x[lag:]) / denom
                      for lag in range(max_lag + 1)])
    return acf

def compute_ccf(x, y_sig, max_lag=MAX_LAG):
    """
    Normalised cross-correlation of signals x and y at lags 0 … max_lag.
    R_xy(τ) = Σ (x(k) - μx)(y(k+τ) - μy)  /  sqrt(Var(x)*Var(y)) / N
    Returns lags 0 … max_lag (positive lags only, x leads y).
    """
    x     = np.asarray(x, dtype=float) - np.mean(x)
    y_sig = np.asarray(y_sig, dtype=float) - np.mean(y_sig)
    n     = len(x)
    denom = np.sqrt(np.dot(x, x) * np.dot(y_sig, y_sig))
    if denom == 0:
        return np.zeros(max_lag + 1)
    ccf   = np.array([np.dot(x[:n-lag], y_sig[lag:]) / denom
                      for lag in range(max_lag + 1)])
    return ccf

def ci_bound(n, alpha=0.95):
    """95% CI for ACF/CCF under null hypothesis of no correlation."""
    from scipy.stats import norm
    z = norm.ppf((1 + alpha) / 2)
    return z / np.sqrt(n)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  COMPUTE ALL ACF AND CCF VALUES
# ─────────────────────────────────────────────────────────────────────────────
data   = df[ALL_COLS].values       # shape (2394, 8)
n_vars = len(ALL_COLS)
N_data = len(data)
CI     = ci_bound(N_data)
lags   = np.arange(MAX_LAG + 1)

print(f"\nComputing ACF/CCF for {n_vars} variables, max lag = {MAX_LAG}...")
print(f"95% CI bound = ±{CI:.4f}")

# Store results in dict:  results[(i,j)] = ccf array (i==j means ACF)
results = {}
for i in range(n_vars):
    for j in range(n_vars):
        xi = data[:, i]
        xj = data[:, j]
        if i == j:
            results[(i, j)] = compute_acf(xi)
        else:
            results[(i, j)] = compute_ccf(xi, xj)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PLOT — 8×8 GRID
#     Diagonal  = ACF of that variable with itself
#     Off-diag  = CCF(row variable, column variable)
#     Layout: rows = "from" variable, cols = "to" variable
# ─────────────────────────────────────────────────────────────────────────────
COLORS_DIAG = "#1B6CA8"   # blue  for ACF (diagonal)
COLORS_OFF  = "#E65100"   # orange for CCF (off-diagonal)

fig, axes = plt.subplots(n_vars, n_vars,
                         figsize=(36, 36),
                         sharex=True, sharey=True)

for i in range(n_vars):
    for j in range(n_vars):
        ax  = axes[i][j]
        arr = results[(i, j)]
        col = COLORS_DIAG if i == j else COLORS_OFF

        # stem plot
        markerline, stemlines, baseline = ax.stem(
            lags, arr,
            linefmt=col,
            markerfmt=f"o",
            basefmt="k-"
        )
        plt.setp(markerline, color=col, markersize=4)
        plt.setp(stemlines,  color=col, linewidth=1.0)

        # CI bounds
        ax.axhline( CI, color="red", ls="--", lw=1.0, alpha=0.8)
        ax.axhline(-CI, color="red", ls="--", lw=1.0, alpha=0.8)
        ax.axhline(0,   color="black", lw=0.8)

        # y-axis limits
        ax.set_ylim(-0.5, 1.05)

        # Remove all tick labels from every cell
        ax.tick_params(labelbottom=False, labelleft=False,
                       bottom=False, left=False)

        # Column header on top row, row label on left column only
        if i == 0:
            ax.set_title(ALL_COLS[j], fontsize=16, fontweight="bold", pad=8)
        if j == 0:
            ax.set_ylabel(ALL_COLS[i], fontsize=16, fontweight="bold", rotation=0,
                          labelpad=32, va="center")

# Single shared axis labels
fig.text(0.5, -0.005, "Lag τ",
         ha="center", fontsize=18, fontweight="bold")
fig.text(-0.005, 0.5, "Row variable",
         va="center", rotation="vertical", fontsize=18, fontweight="bold")

fig.suptitle(
    "ACF / CCF Matrix — Debutanizer Variables\n"
    "Diagonal = Autocorrelation (ACF)  |  "
    "Off-diagonal = Cross-correlation (CCF)  |  "
    "Red dashed = 95% CI",
    fontsize=13, fontweight="bold", y=1.03
)

plt.tight_layout(h_pad=0.4, w_pad=0.4)
plt.savefig(f"{OUT_DIR}/acf_ccf_matrix.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: acf_ccf_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  INDIVIDUAL ACF PLOTS (one per variable, cleaner view)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col_name in enumerate(ALL_COLS):
    ax  = axes[i]
    acf = results[(i, i)]

    ml, sl, bl = ax.stem(lags, acf, linefmt="#1B6CA8",
                         markerfmt="o", basefmt="k-")
    plt.setp(ml, color="#1B6CA8", markersize=4)
    plt.setp(sl, color="#1B6CA8", linewidth=1.0)

    ax.axhline( CI, color="red", ls="--", lw=1.2, label=f"95% CI (±{CI:.3f})")
    ax.axhline(-CI, color="red", ls="--", lw=1.2)
    ax.axhline(0,   color="black", lw=0.8)

    pct_within = np.mean(np.abs(acf[1:]) < CI) * 100
    ax.set_title(f"ACF — {col_name}\n({pct_within:.0f}% lags within CI)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Lag τ", fontsize=9)
    ax.set_ylabel("Autocorrelation", fontsize=9)
    ax.set_ylim(-0.5, 1.05)
    ax.legend(fontsize=7)

fig.suptitle("Individual ACF for All 8 Variables",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/acf_individual.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: acf_individual.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. CCF OF EACH INPUT WITH TARGET y  (most relevant for the report)
# ─────────────────────────────────────────────────────────────────────────────
y_idx      = ALL_COLS.index("y")
input_idxs = [ALL_COLS.index(c) for c in ALL_COLS if c != "y"]
input_names= [c for c in ALL_COLS if c != "y"]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for ax_i, (inp_idx, inp_name) in enumerate(zip(input_idxs, input_names)):
    ax  = axes[ax_i]
    ccf = results[(inp_idx, y_idx)]

    ml, sl, bl = ax.stem(lags, ccf, linefmt="#E65100",
                         markerfmt="o", basefmt="k-")
    plt.setp(ml, color="#E65100", markersize=4)
    plt.setp(sl, color="#E65100", linewidth=1.0)

    ax.axhline( CI, color="red", ls="--", lw=1.2)
    ax.axhline(-CI, color="red", ls="--", lw=1.2)
    ax.axhline(0,   color="black", lw=0.8)

    pct = np.mean(np.abs(ccf[1:]) < CI) * 100
    ax.set_title(f"CCF — {inp_name} → y\n({pct:.0f}% lags within CI)",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(-0.6, 0.6)

axes[-1].set_visible(False)   # 7 inputs, 8th cell unused

fig.suptitle("Cross-Correlation of Each Input with Target y",
             fontsize=14, fontweight="bold")
fig.text(0.5, -0.01, "Lag τ", ha="center", fontsize=12, fontweight="bold")
fig.text(-0.01, 0.5, "Cross-correlation", va="center", rotation="vertical",
         fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ccf_inputs_vs_y.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: ccf_inputs_vs_y.png")

# ─────────────────────────────────────────────────────────────────────────────
# PACF and PCCF with target y
# PACF  — Partial Autocorrelation of y with itself
# PCCF  — Partial Cross-Correlation of each input with y
#
# "Partial" means: the correlation at lag τ after removing the effect
# of all intermediate lags 1, 2, ... τ-1. This directly tells you
# which specific lags are genuinely informative vs redundant.
# Use this to decide how many lags to include in the NARMAX model.
# ─────────────────────────────────────────────────────────────────────────────
from statsmodels.tsa.stattools import pacf, ccf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAX_LAG = 15   # change this if you want more or fewer lags shown

# Output folder — same as rest of script
PACF_DIR = OUT_DIR   # saves into your existing output folder
import os
os.makedirs(PACF_DIR, exist_ok=True)

y_series      = df["y"].values
input_cols_all = ["u1", "u2", "u3", "u4", "u5", "u6", "u7"]

# 95% CI bound (same formula as ACF/CCF)
N   = len(y_series)
CI  = 1.96 / np.sqrt(N)
lags_arr = np.arange(0, MAX_LAG + 1)

# ── Helper: compute partial cross-correlation (PCCF) ─────────────────────────
def partial_cross_correlation(x, y_sig, max_lag):
    """
    Compute PCCF of x → y at each lag by regressing out lower-order lags.

    At lag τ:
      1. Regress y(k+τ) on x(k+1)...x(k+τ-1)  → get residual ry
      2. Regress x(k)   on x(k+1)...x(k+τ-1)  → get residual rx
      3. PCCF(τ) = Pearson correlation of rx and ry

    At lag 0: just the plain Pearson r between x and y (no conditioning).
    At lag 1: plain CCF at lag 1 (nothing to partial out yet).
    """
    from numpy.linalg import lstsq

    x     = np.asarray(x,     dtype=float)
    y_sig = np.asarray(y_sig, dtype=float)
    n     = len(x)

    def pearson(a, b):
        a = a - a.mean(); b = b - b.mean()
        denom = np.sqrt((a**2).sum() * (b**2).sum())
        return 0.0 if denom == 0 else np.dot(a, b) / denom

    pccf = np.zeros(max_lag + 1)

    for tau in range(max_lag + 1):
        if tau == 0:
            # plain correlation at lag 0
            xi = x[:n]
            yi = y_sig[:n]
            pccf[tau] = pearson(xi, yi)

        elif tau == 1:
            # plain CCF at lag 1 (nothing to condition on)
            xi = x[:n - 1]
            yi = y_sig[1:n]
            pccf[tau] = pearson(xi, yi)

        else:
            # Build conditioning matrix from intermediate lags 1 ... tau-1
            # Rows aligned so x(k) predicts y(k+tau)
            n_eff = n - tau
            # x(k), x(k+1), ..., x(k+tau-1) shifted to align with y(k+tau)
            Z = np.column_stack([x[s: s + n_eff]
                                 for s in range(1, tau)])  # lags 1..tau-1

            xi = x[:n_eff]          # x(k)
            yi = y_sig[tau: n]      # y(k+tau)

            # Residualise xi on Z
            _, rx, _, _ = lstsq(Z, xi, rcond=None)
            xi_res = xi - Z @ lstsq(Z, xi, rcond=None)[0]

            # Residualise yi on Z
            yi_res = yi - Z @ lstsq(Z, yi, rcond=None)[0]

            pccf[tau] = pearson(xi_res, yi_res)

    return pccf


# ── 1. PACF of y ──────────────────────────────────────────────────────────────
pacf_vals = pacf(y_series, nlags=MAX_LAG, method="ywm")

fig, ax = plt.subplots(figsize=(12, 4))
ax.stem(lags_arr, pacf_vals[:MAX_LAG + 1],
        linefmt="#1B6CA8", markerfmt="o", basefmt="k-")
ax.axhline( CI, color="red", ls="--", lw=1.2, label=f"95% CI (±{CI:.3f})")
ax.axhline(-CI, color="red", ls="--", lw=1.2)
ax.axhline(0,   color="black", lw=0.8)

# Annotate last significant lag
sig_lags = np.where(np.abs(pacf_vals[1:MAX_LAG+1]) > CI)[0] + 1
last_sig  = sig_lags[-1] if len(sig_lags) > 0 else 0
ax.axvline(last_sig, color="green", ls=":", lw=1.5,
           label=f"Last significant lag = {last_sig}")

ax.set_title("PACF of Target y (C4 Concentration)\n"
             "Significant lags suggest how many output lags to include in NARMAX",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Lag τ")
ax.set_ylabel("Partial Autocorrelation")
ax.legend(fontsize=9)
ax.set_xlim(-0.5, MAX_LAG + 0.5)
plt.tight_layout()
plt.savefig(f"{PACF_DIR}/pacf_y.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: pacf_y.png")

# ── 2. PCCF of each input with y — individual panels ─────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()

for i, col in enumerate(input_cols_all):
    ax      = axes[i]
    x_series = df[col].values
    pccf_vals = partial_cross_correlation(x_series, y_series, MAX_LAG)

    ax.stem(lags_arr, pccf_vals,
            linefmt="#E65100", markerfmt="o", basefmt="k-")
    ax.axhline( CI, color="red", ls="--", lw=1.2)
    ax.axhline(-CI, color="red", ls="--", lw=1.2)
    ax.axhline(0,   color="black", lw=0.8)

    # First significant lag (tells you minimum lag needed)
    sig = np.where(np.abs(pccf_vals[1:]) > CI)[0]
    first_sig = sig[0] + 1 if len(sig) > 0 else None
    last_sig  = sig[-1] + 1 if len(sig) > 0 else None

    title_str = f"PCCF — {col} → y"
    if first_sig is not None:
        title_str += f"\nSignificant lags: {first_sig} to {last_sig}"
    else:
        title_str += "\nNo significant lags"
    ax.set_title(title_str, fontsize=10, fontweight="bold")
    ax.set_xlabel("Lag τ", fontsize=9)
    ax.set_ylabel("Partial Cross-correlation", fontsize=9)
    ax.set_xlim(-0.5, MAX_LAG + 0.5)
    ax.set_ylim(-0.6, 0.6)

axes[-1].set_visible(False)   # 7 inputs, 8th cell empty

fig.suptitle("Partial Cross-Correlation (PCCF) — Each Input with Target y\n"
             "Shows which specific lags carry unique predictive information",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PACF_DIR}/pccf_inputs_vs_y.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: pccf_inputs_vs_y.png")

# ── 3. Summary print — recommended lags per variable ─────────────────────────
print(f"\n{'─'*50}")
print(f"  LAG SELECTION SUMMARY  (95% CI = ±{CI:.3f})")
print(f"{'─'*50}")

pacf_sig = np.where(np.abs(pacf_vals[1:MAX_LAG+1]) > CI)[0] + 1
print(f"  y  (PACF) : significant lags = {list(pacf_sig)}")
print(f"             → suggested output lags in NARMAX: {len(pacf_sig)}")

for col in input_cols_all:
    x_series  = df[col].values
    pccf_vals = partial_cross_correlation(x_series, y_series, MAX_LAG)
    sig       = np.where(np.abs(pccf_vals[1:]) > CI)[0] + 1
    print(f"  {col:4s} (PCCF): significant lags = {list(sig)}")

print(f"{'─'*50}")
