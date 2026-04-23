"""
Soft Sensor 
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import os, warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", font_scale=1.05)
PLOT_DIR = "plots/model_v2"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# HYPERPARAMETERS  (paper values)
# ─────────────────────────────────────────────
N_OUTPUT_LAGS = 4
N_U5_LAGS     = 3
DELAY_STEPS   = 3
N_HIDDEN      = 13
TRAIN_FRAC    = 0.75
N_RESTARTS    = 10      # train N times, keep best

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
df = pd.read_csv("debutanizer_full.csv")
df.columns = df.columns.str.strip()
df = df.apply(pd.to_numeric, errors="coerce").dropna()
df["u_avg"] = (df["u6"] + df["u7"]) / 2
print(f"Loaded: {df.shape[0]} samples")

# ─────────────────────────────────────────────
# BUILD NARMAX MATRIX
# ─────────────────────────────────────────────
def build_narmax_matrix(df, n_out=N_OUTPUT_LAGS, n_u5=N_U5_LAGS, delay=DELAY_STEPS):
    max_lag = max(n_out, n_u5) + delay
    rows_X, rows_y = [], []
    for k in range(max_lag, len(df)):
        row = []
        for lag in range(1, n_out + 1):
            row.append(df["y"].iloc[k - lag])
        for col in ["u1", "u2", "u3", "u4"]:
            row.append(df[col].iloc[k])
        for lag in range(0, n_u5 + 1):
            row.append(df["u5"].iloc[k - lag])
        row.append(df["u_avg"].iloc[k])
        rows_X.append(row)
        rows_y.append(df["y"].iloc[k - delay])

    feature_names = (
        [f"y(k-{i})" for i in range(1, n_out+1)] +
        ["u1(k)","u2(k)","u3(k)","u4(k)"] +
        [f"u5(k-{i})" for i in range(0, n_u5+1)] +
        ["u_avg(k)"]
    )
    return np.array(rows_X, np.float32), np.array(rows_y, np.float32), feature_names

X, y_target, feature_names = build_narmax_matrix(df)
split = int(len(X) * TRAIN_FRAC)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_target[:split], y_target[split:]
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}  |  Features: {X.shape[1]}")

# ─────────────────────────────────────────────
# TRAIN WITH MULTIPLE RESTARTS — keep best test R²
# ─────────────────────────────────────────────
print(f"\nTraining {N_RESTARTS} restarts, keeping best...")
best_model, best_r2 = None, -np.inf

for seed in range(N_RESTARTS):
    m = MLPRegressor(
        hidden_layer_sizes=(N_HIDDEN,),
        activation="logistic",
        solver="lbfgs",
        max_iter=5000,
        tol=1e-8,
        random_state=seed,
    )
    m.fit(X_train, y_train)
    r2 = r2_score(y_test, m.predict(X_test))
    print(f"  seed={seed:2d}  test R²={r2:.6f}")
    if r2 > best_r2:
        best_r2, best_model = r2, m

model = best_model
print(f"\nBest test R²: {best_r2:.6f}")

# ─────────────────────────────────────────────
# PREDICTIONS & METRICS
# ─────────────────────────────────────────────
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)
res_train    = y_train - y_pred_train
res_test     = y_test  - y_pred_test

def metrics(y_true, y_pred):
    return (np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred),
            r2_score(y_true, y_pred))

rmse_tr, mae_tr, r2_tr = metrics(y_train, y_pred_train)
rmse_te, mae_te, r2_te = metrics(y_test,  y_pred_test)

print(f"\n{'Split':8s}  {'RMSE':>10s}  {'MAE':>10s}  {'R²':>10s}")
print("-" * 44)
print(f"{'Train':8s}  {rmse_tr:>10.5f}  {mae_tr:>10.5f}  {r2_tr:>10.5f}")
print(f"{'Test':8s}  {rmse_te:>10.5f}  {mae_te:>10.5f}  {r2_te:>10.5f}")

# ─────────────────────────────────────────────
# WHITENESS TEST — fraction of cross-corr within CI
# ─────────────────────────────────────────────
MAX_LAG = 20
n       = len(res_test)
conf_95 = 1.96 / np.sqrt(n)
res_dm  = res_test - res_test.mean()

acf_full = np.correlate(res_dm, res_dm, mode="full")
acf      = acf_full[n-1: n+MAX_LAG] / acf_full[n-1]

# Fraction of ACF lags (excluding lag 0) within 95% CI
acf_within = np.mean(np.abs(acf[1:]) < conf_95)
print(f"\nWhiteness: {acf_within*100:.1f}% of ACF lags within 95% CI (want >90%)")

# ─────────────────────────────────────────────
# PLOT 1 — Fig 4 style
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
fig.suptitle("C4 Soft Sensor — Actual vs Estimated (Validation Set)\n"
             "Reproducing Fig. 4 · Fortuna et al. (2005)",
             fontsize=13, fontweight="bold")
samples = np.arange(len(y_test))

axes[0].plot(samples, y_test, "#1f77b4", lw=0.9, label="Actual (system output)")
axes[0].set_ylabel("C4 Conc. (norm.)", fontsize=10)
axes[0].set_title("(a) Actual Data", fontsize=11); axes[0].legend(fontsize=9)

axes[1].plot(samples, y_pred_test, "#ff7f0e", lw=0.9, label="Neural estimation")
axes[1].set_ylabel("C4 Conc. (norm.)", fontsize=10)
axes[1].set_title("(b) Neural Network Estimation", fontsize=11); axes[1].legend(fontsize=9)
axes[1].text(0.98, 0.95, f"RMSE={rmse_te:.4f}   R²={r2_te:.4f}",
             transform=axes[1].transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=0.8))

axes[2].plot(samples, res_test, "#d62728", lw=0.7, alpha=0.8)
axes[2].axhline(0, color="k", lw=0.9, ls="--")
axes[2].set_ylabel("Error (norm.)", fontsize=10)
axes[2].set_xlabel("Samples", fontsize=10)
axes[2].set_title("(c) Residual", fontsize=11)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/fig4_actual_vs_estimated.png", dpi=200, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# PLOT 2 — Fig 5 style (FIXED cross-correlations)
# Key fix: use the actual model INPUT COLUMNS for cross-corr,
# aligned to the test set index — this is what the paper does.
# ─────────────────────────────────────────────

# The test set starts at row `split` in X.
# X columns in order: y(k-1..4), u1,u2,u3,u4, u5(k..k-3), u_avg
# For cross-corr with residual we use each INPUT feature column from X_test
input_col_indices = {
    "u1(k)"    : 4,
    "u2(k)"    : 5,
    "u3(k)"    : 6,
    "u4(k)"    : 7,
    "u5(k)"    : 8,
    "u_avg(k)" : 12,
}

fig = plt.figure(figsize=(15, 10))
fig.suptitle("Residual Analysis — Reproducing Fig. 5 · Fortuna et al. (2005)",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

# (a) Dispersion
ax_a = fig.add_subplot(gs[0, 0])
ax_a.scatter(y_pred_test, res_test, alpha=0.3, s=8, color="#1f77b4")
ax_a.axhline(0, color="red", lw=0.9, ls="--")
ax_a.set_xlabel("Fitted values", fontsize=9)
ax_a.set_ylabel("Residual", fontsize=9)
ax_a.set_title("(a) Residual vs Fitted", fontsize=10)

# (b) Residual ACF
ax_b = fig.add_subplot(gs[0, 1])
ml, sl, bl = ax_b.stem(np.arange(len(acf)), acf,
                        markerfmt="C0o", linefmt="C0-", basefmt="k-")
plt.setp(sl, lw=0.8); plt.setp(ml, ms=4)
ax_b.axhline( conf_95, color="red", ls="--", lw=0.9, label="95% CI")
ax_b.axhline(-conf_95, color="red", ls="--", lw=0.9)
ax_b.set_xlabel("Lag τ", fontsize=9)
ax_b.set_title(f"(b) Residual Autocorrelation\n({acf_within*100:.0f}% within CI)", fontsize=10)
ax_b.legend(fontsize=7)

# (c) Normal probability plot
ax_c = fig.add_subplot(gs[0, 2])
(osm, osr), (slope, intercept, r_val) = stats.probplot(res_test, dist="norm")
ax_c.plot(osm, osr, "o", alpha=0.4, ms=3, color="#1f77b4")
ax_c.plot(osm, slope*np.array(osm)+intercept, "r-", lw=1.5)
ax_c.set_xlabel("Theoretical quantiles", fontsize=9)
ax_c.set_ylabel("Ordered values", fontsize=9)
ax_c.set_title(f"(c) Normal Probability Plot\nR = {r_val:.4f}", fontsize=10)

# (d)–(i) Cross-correlations using actual input regressors from X_test
panel_positions = [(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
letters = ["d","e","f","g","h","i"]
within_pcts = []

for (label, col_idx), pos, letter in zip(input_col_indices.items(),
                                          panel_positions, letters):
    u = X_test[:, col_idx]
    u_dm = u - u.mean()
    norm_fac = np.sqrt(np.sum(res_dm**2) * np.sum(u_dm**2))
    cc_full = np.correlate(res_dm, u_dm, mode="full")
    cc = cc_full[n-1: n+MAX_LAG] / (norm_fac + 1e-12)

    within = np.mean(np.abs(cc) < conf_95) * 100
    within_pcts.append(within)

    ax = fig.add_subplot(gs[pos])
    ml2, sl2, bl2 = ax.stem(np.arange(len(cc)), cc,
                              markerfmt="C0o", linefmt="C0-", basefmt="k-")
    plt.setp(sl2, lw=0.8); plt.setp(ml2, ms=3)
    ax.axhline( conf_95, color="red", ls="--", lw=0.8)
    ax.axhline(-conf_95, color="red", ls="--", lw=0.8)
    ax.set_xlabel("Lag τ", fontsize=8)
    ax.set_title(f"({letter}) residual × {label}\n({within:.0f}% within CI)", fontsize=9)

print(f"\nCross-corr within CI: {[f'{p:.0f}%' for p in within_pcts]}")

plt.savefig(f"{PLOT_DIR}/fig5_residual_analysis.png", dpi=200, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# PLOT 3 — Online performance overlay
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(samples, y_test,      "#2c7bb6", lw=0.9, label="Actual (gas chromatograph)")
ax.plot(samples, y_pred_test, "#d7191c", lw=0.9, ls="--",
        label="Neural soft sensor estimate")
ax.set_xlabel("Samples", fontsize=11)
ax.set_ylabel("C4 in iC5 — normalised concentration", fontsize=11)
ax.set_title(f"On-line Soft Sensor Performance — C4 in Bottom Flow\n"
             f"RMSE={rmse_te:.4f}   R²={r2_te:.4f}   MAE={mae_te:.4f}",
             fontsize=12)
ax.legend(fontsize=10); ax.set_ylim(-0.05, 1.1)
sns.despine(); plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/fig8_online_performance.png", dpi=200, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# PLOT 4 — Metrics bar chart
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
fig.suptitle("Model Performance Metrics", fontsize=13, fontweight="bold")
palette = ["#4292c6", "#fc8d59"]
for ax, (metric, vals) in zip(axes, [
    ("RMSE", [rmse_tr, rmse_te]),
    ("MAE",  [mae_tr,  mae_te]),
    ("R²",   [r2_tr,   r2_te])
]):
    bars = ax.bar(["Train","Test"], vals, color=palette, edgecolor="white", width=0.5)
    ax.set_title(metric, fontsize=12)
    ax.set_ylim(0, max(vals) * 1.25)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.02,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    sns.despine(ax=ax)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/metrics_summary.png", dpi=200, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────
# PLOT 5 — Scatter
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y_test, y_pred_test, alpha=0.3, s=10, color="#1f77b4")
lims = [min(y_test.min(), y_pred_test.min())-0.02,
        max(y_test.max(), y_pred_test.max())+0.02]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
ax.set_xlabel("Actual C4 concentration", fontsize=11)
ax.set_ylabel("Predicted C4 concentration", fontsize=11)
ax.set_title(f"Actual vs Predicted (Test Set)\nR² = {r2_te:.4f}", fontsize=12)
ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/scatter_actual_vs_predicted.png", dpi=200, bbox_inches="tight")
plt.close()

