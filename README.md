# ADAPEL: Adaptive Doubly-Robust Pseudo-outcome Ensemble Learner

[![CI](https://github.com/manhthai1706/ADAPEL/actions/workflows/ci.yml/badge.svg)](https://github.com/manhthai1706/ADAPEL/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()

A meta-learning model for **Conditional Average Treatment Effect (CATE)** estimation from observational data. ADAPEL adaptively fuses three complementary meta-learners (DR-Learner, X-Learner, R-Learner) into a single propensity-driven ensemble, with a focus on doubly-robust inference and clinical-grade uncertainty quantification.

Optimised for **weak machines**: parallel OOF stacking, mode presets (fast/balanced/accurate), threading backend.

## Key Innovation

Existing meta-learners have well-known failure modes:

- **T/S-Learner** underperforms when treatment groups are imbalanced.
- **X-Learner** has high variance in regions of strong treatment imbalance.
- **DR-Learner** is optimal only when both outcome and propensity models are correctly specified.
- **R-Learner** is unbiased but suffers from inverse-propensity variance in well-overlapped regions.

**ADAPEL** combines these into a single adaptive pseudo-outcome:

```
Y_fused(x) = alpha(x) * Y_X(x) + (1 - alpha(x)) * Y_DR(x)

alpha(x) = max(min_alpha, clip(1 - 4*e(x)*(1-e(x)), 0, 1)^gamma)
```

- When `e(x) ≈ 0.5` (well-overlapped region): alpha → 0, ADAPEL relies on the **DR-Learner** (lowest variance under good overlap).
- When `e(x) ≈ 0` or `1` (imbalanced region): alpha → 1, ADAPEL falls back to the **X-Learner** (more stable in tails).
- **R-Learner sample weighting** `(T - e(x))^2` further upweights informative samples whose treatment status deviates from the propensity.

The fused pseudo-outcome is then regressed on `X` using **NNLS-constrained positive stacking** over a diverse set of base learners (HistGBM, ExtraTrees, Ridge, DecisionTree, Lasso), with L2 regularization for stability. OOF stacking runs in **parallel** via joblib threading.

## Algorithm (4 Steps)

1. Fit nuisance functions `mu0(x), mu1(x), e(x)` on full data (no cross-fitting, faster on small data).
2. **Adaptive pseudo-outcome** `Y_fused = alpha * Y_X + (1 - alpha) * Y_DR` driven by `e(x)`.
3. **R-Learner sample weights** `w_i = (T_i - e(X_i))^2` to upweight informative samples.
4. **NNLS positive stacking** with L2 regularization on out-of-fold base learner predictions for the final CATE estimator.

## Installation

```bash
pip install git+https://github.com/manhthai1706/ADAPEL.git
```

Or for development:

```bash
git clone https://github.com/manhthai1706/ADAPEL.git
cd ADAPEL
pip install -r requirements.txt
pip install -e .
```

## Mode Presets

| Mode | Outcome iter | Tree depth | ExtraTrees | OOF frac | Boot frac | Use case |
|------|-------------|------------|------------|----------|-----------|----------|
| `fast` | 80 | 4 | 100 | 0.35 | 0.50 | Weak machines, quick exploration |
| `balanced` | 150 | 6 | 200 | 0.50 | 0.70 | Default, general purpose |
| `accurate` | 300 | 8 | 500 | 0.70 | 0.85 | Maximum accuracy, strong machines |

```python
# Fast mode — lighter models, less RAM
model = ADAPEL(mode="fast", verbose=True).fit(X, T, Y)
```

## Quick Start

```python
from learner import ADAPEL
import numpy as np

# X: covariates, T: binary treatment, Y: observed outcome
model = ADAPEL(n_folds=3, mode="fast")
model.fit(X, T, Y)

# CATE estimates per individual
cate = model.predict(X)

# Counterfactual inference
y0, y1 = model.predict_potential_outcomes(X)
y_factual = model.predict_counterfactual(X, T_observed=T)

# ATE, ATT, ATC
ate = model.estimate_ate(X)
att = model.estimate_att(X, T)
atc = model.estimate_atc(X, T)
```

## Bootstrap Confidence Intervals

```python
model.fit_bootstrap(X, T, Y, n_bootstrap=10, random_state=42)
result = model.predict_clinical(X)

# BMA point estimate + 95% percentile CI + overlap check
print(result["cate"], result["lower_ci"], result["upper_ci"])
print("In overlap region:", result["in_overlap"].mean())
```

## Clinical-Grade Diagnostics

- **Overlap check**: `result["in_overlap"]` flags patients with extreme propensity (e(x) < 0.05 or > 0.95) where causal extrapolation is unreliable.
- **E-Value** (VanderWeele & Ding 2017): minimum strength of unmeasured confounding needed to explain away the observed ATE.
- **Surrogate decision tree**: human-readable rules extracted from a shallow tree fit on CATE predictions.

```python
e_val = model.estimate_e_value(X, outcome_type="binary")
rules = model.explain_cate_surrogate(X, feature_names=cols, max_depth=3)
```

## Benchmarks

### Synthetic data (built-in, no download needed)

```bash
python train.py -m fast
```

### Real-data benchmarks (auto-download from GitHub, no setup)

```bash
# IHDP — semi-synthetic benchmark with known ground-truth CATE
python ihdp.py -m fast

# Hillstrom — RCT email marketing, validates ATE vs the trial estimate
python hillstrom.py -m fast --arm mens
python hillstrom.py -m fast --arm womens
```

| Benchmark | Samples | Features | Result |
|-----------|---------|----------|--------|
| IHDP (npci_1) | 747 | 25 | ADAPEL PEHE **0.691**; ATE 4.037 (true 4.016); bootstrap CI coverage **96.7%** |
| Hillstrom (mens arm) | 42,613 | 9 | ADAPEL ATE **0.7613** |
| Hillstrom (womens arm) | 42,693 | 9 | ADAPEL ATE **0.4236** |

## Clinical-Grade Analysis

```python
# Sample size adequacy check
report = model.sample_size_report(X, T)

# Covariate balance (SMD) with propensity weighting
balance = model.balance_check(X, T, feature_names=cols)

# Subgroup analysis (auto-quartile or custom masks)
subgroups = model.subgroup_analysis(X, T, Y, n_bins=4)

# Variable importance (permutation-based)
importance = model.variable_importance(X, feature_names=cols, n_repeats=10)

# Negative control / placebo test
placebo = model.negative_control_test(X, n_permute=100)

# Calibration check (predicted vs observed CATE by quantile)
calib = model.calibration_check(X, n_groups=10)

# Fairness assessment across protected groups
protected = {"race": race_col, "gender": gender_col}
fairness = model.fairness_report(X, protected)

# Save / load model
model.save("learner_model.joblib")
loaded = ADAPEL.load("learner_model.joblib")

# Audit trail
audit = model.get_audit_trail()
```

## API Reference

### Core

| Method | Description |
|--------|-------------|
| `ADAPEL(mode, verbose, ...)` | Init: `mode`='fast'/'balanced'/'accurate'; `verbose` for progress logs. |
| `fit(X, T, Y)` | Train ADAPEL on covariates, treatment, outcome. |
| `predict(X)` | Predict CATE `tau(x)` for each individual. |
| `predict_potential_outcomes(X)` | Return `(Y(0), Y(1))` predictions. |
| `predict_counterfactual(X, T_observed)` | Predict the unobserved potential outcome. |
| `estimate_ate(X)` / `att` / `atc` | ATE, ATT, ATC. |

### Inference & Uncertainty

| Method | Description |
|--------|-------------|
| `fit_bootstrap(X, T, Y, n_bootstrap)` | Bootstrap ensemble for confidence intervals. |
| `predict_clinical(X, alpha)` | BMA point estimate + percentile CI + overlap flags. |

### Sensitivity & Explainability

| Method | Description |
|--------|-------------|
| `estimate_e_value(X, outcome_type)` | E-Value for unmeasured confounding. |
| `explain_cate_surrogate(X, feature_names)` | Surrogate decision tree rules. |
| `get_diagnostics(X)` | Propensity, alpha, stacking weights, ensemble std. |

### Clinical Analysis

| Method | Description |
|--------|-------------|
| `sample_size_report(X, T)` | Sample size adequacy check with warnings. |
| `balance_check(X, T, feature_names)` | Standardised mean difference (SMD) before/after weighting. |
| `subgroup_analysis(X, T, Y, ...)` | CATE heterogeneity across subgroups with p-values. |
| `variable_importance(X, ...)` | Permutation-based feature importance for CATE. |
| `negative_control_test(X, ...)` | Placebo test (treatment permutation) for spurious signal. |
| `calibration_check(X, ...)` | Predicted CATE vs observed outcome by quantile groups. |
| `fairness_report(X, protected_attrs)` | Disparity analysis across protected groups. |

### Model Management

| Method | Description |
|--------|-------------|
| `save(path)` | Serialise fitted model to disk (.joblib). |
| `load(path)` | Static: load fitted model from disk. |
| `get_audit_trail()` | Return version, timestamp, params, missing data, sample size. |

## References

- Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Meta-learners for estimating heterogeneous treatment effects. *Annals of Applied Statistics*, 13(2), 893-934.
- Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299-319.
- Kennedy, E. H. (2020). Towards optimal doubly robust estimation of heterogeneous treatment effects. *Electronic Journal of Statistics*, 14(1), 3008-3048.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: introducing the E-value. *Annals of Internal Medicine*, 167(4), 268-274.
- Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference. *Journal of Computational and Graphical Statistics*, 20(1), 217-240. (IHDP benchmark)

## License

MIT
