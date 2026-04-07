# Soft Sensor for Debutanizer Distillation Column

A rigorous data-driven soft-sensing pipeline for real-time C4 quality estimation in a petroleum debutanizer column, integrating **exploratory data analysis**, **temporal structure analysis (ACF/PACF/CCF/PCCF)**, **static baseline benchmarking**, and a **NARMAX-MLP neural network** — replicating and validating Fortuna et al. (2005).

This repository is structured to clearly separate each stage of the soft-sensor development workflow, from raw data understanding through to model validation.

---

## Repository Structure

<pre>
Debutanizer-SoftSensor/
│── main.py                      # EDA — univariate statistics, histograms, pairwise scatter plots
│── model_static.py              # Static baselines (Linear Regression, static MLP) + ACF/CCF/PACF/PCCF
│── model.py                     # NARMAX-MLP model — training, evaluation, residual analysis
│── debutanizer_full.csv         # Industrial dataset (2394 × 8)
│── report/
│   └── soft_sensor_report.pdf   # Full technical report
└── README.md
</pre>

---

## Project Workflow

1. **Exploratory Data Analysis** (`main.py`)
2. **Temporal structure analysis — ACF, PACF, CCF, PCCF** (`model_static.py`)
3. **Static baseline models — empirical justification for dynamics** (`model_static.py`)
4. **NARMAX model development and paper replication** (`model.py`)
5. **Residual analysis and model validation** (`model.py`)

---

## Dataset

- **Source:** ERG Raffineria Mediterranea, Italy (Fortuna et al., 2005)
- **Samples:** 2,394 observations at 6-minute intervals (~240 hours of operation)
- **Inputs (u1–u7):** 7 process variables — reflux flow, feed flow, top/bottom temperatures, pressures
- **Output (y):** Bottom product C4 concentration (gas chromatograph measurement)
- **Split:** 75% training (~1,795 samples), 25% test (~599 samples), chronological

---

## Exploratory Data Analysis

Implemented in **main.py**.

- Univariate statistics: mean, median, std, variance, skewness, kurtosis for all 8 variables
- Per-variable histogram + KDE and boxplot — saved individually and as a combined 8×2 grid
- All 28 pairwise scatter plots across variables
- Input-vs-target scatter plots for each of the 7 inputs against y

### Key Observations
- Bimodal distributions identified in **u4, u6, u7** — suggests two distinct operating regimes
- Remaining variables (u1, u2, u3, u5, y) are approximately bell-shaped

---

## Temporal Structure Analysis

Implemented in **model_static.py**.

Distillation columns exhibit strong process dynamics — the current output depends on past inputs and outputs. This section identifies significant lag windows before any dynamic model is built.

- **ACF:** Autocorrelation of each variable with itself (diagonal of 8×8 matrix)
- **CCF:** Cross-correlation between all variable pairs across lags 0–30 (full 8×8 matrix)
- **PACF:** Partial autocorrelation of y — directly indicates how many output lags to include
- **PCCF:** Partial cross-correlation of each input with y — identifies which specific lags carry unique predictive information

### Key Findings
- y shows strong autocorrelation up to several lags — confirms long process memory
- PACF and PCCF provide data-driven, statistically grounded justification for the lag window used in the NARMAX feature set

---

## Static Baseline Models

Implemented in **model_static.py**.

These models deliberately exclude all time lag features to empirically demonstrate that static modeling is insufficient for this process.

| Model | Train R² | Test R² |
|---|---|---|
| Linear Regression | 0.194 | 0.110 |
| Static MLP (13 neurons, sigmoid, lbfgs, 10 restarts) | — | −2.066 |

- A static linear model explains only ~11% of variance on unseen data
- A static MLP performs worse than a mean predictor (R² < 0)
- **Conclusion:** Temporal lag features are not optional — they are essential for this system

---

## NARMAX Model

Implemented in **model.py**.

### Feature Set (13 inputs)
- Lagged outputs: y(k−1), y(k−2), y(k−3), y(k−4)
- Current inputs: u1(k), u2(k), u3(k), u4(k)
- Lagged input: u5(k), u5(k−1), u5(k−2), u5(k−3)
- Averaged input: u\_avg(k) = (u6 + u7) / 2
- Prediction delay: 3 steps (accounts for the 45-minute GC measurement delay)

### Training
- Architecture: 1 hidden layer, 13 neurons, sigmoid activation, lbfgs solver
- 10 random restarts — best test R² model retained

### Results

| Model | Train R² | Test R² | Test RMSE |
|---|---|---|---|
| Linear Regression | 0.194 | 0.110 | — |
| Static MLP | — | −2.066 | — |
| **NARMAX-MLP (replicated)** | **~1.000** | **0.9998** | **0.0025** |

The replicated model matches the performance reported in Fortuna et al. (2005), fully validating the implementation.

---

## Residual Analysis

- ACF of residuals confirms **whiteness** — no remaining autocorrelation structure
- Residuals verified to be zero-mean and approximately normally distributed (Q-Q plot)
- Cross-correlation between residuals and each input regressor within 95% CI
- Model is deemed adequate — no systematic unexplained dynamics remain

---

## Key Takeaways

- Debutanizer dynamics are **nonlinear, temporal, and regime-dependent**
- Static models fail entirely — this is not an assumption but an empirically demonstrated result
- Bimodal EDA findings inform expectations about model generalization across regimes
- ACF/PACF/CCF/PCCF analysis provides statistically grounded lag selection rather than arbitrary choices
- The NARMAX architecture eliminates the 45-minute GC delay, enabling real-time quality estimation

---

## Reference

Fortuna, L., Graziani, S., Rizzo, A., & Xibilia, M. G. (2005). Soft sensors for monitoring and control of industrial processes. *Control Engineering Practice*, 13(4), 499–508.
