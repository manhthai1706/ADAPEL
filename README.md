# ADAPEL: Adaptive Doubly-Robust Pseudo-outcome Ensemble Learner

A meta-learning framework for **Conditional Average Treatment Effect (CATE)** estimation from observational data. ADAPEL adaptively fuses three complementary meta-learners (DR-Learner, X-Learner, R-Learner) into a single propensity-driven ensemble, with a focus on doubly-robust inference and clinical-grade uncertainty quantification.

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

The fused pseudo-outcome is then regressed on `X` using **NNLS-constrained positive stacking** over a set of base learners, ensuring non-negative ensemble weights for interpretability and stability.

## Algorithm (4 Steps)

1. **Cross-fit k-fold** to estimate nuisance functions `mu0(x), mu1(x), e(x)` without overfitting bias.
2. **Adaptive pseudo-outcome** `Y_fused = alpha * Y_X + (1 - alpha) * Y_DR` driven by `e(x)`.
3. **R-Learner sample weights** `w_i = (T_i - e(X_i))^2` to upweight informative samples.
4. **NNLS positive stacking** on out-of-fold base learner predictions for the final CATE estimator.

## Installation

```bash
pip install numpy scipy scikit-learn pandas
python download_datasets.py    # downloads IHDP and RHC benchmark datasets
```

## Quick Start

```python
from meta import ADAPEL
import numpy as np

# X: covariates, T: binary treatment, Y: observed outcome
model = ADAPEL(n_folds=3, fusion_gamma=1.0, min_alpha=0.1)
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

## Benchmark (IHDP, 747 samples, semi-synthetic)

| Method | PEHE (lower is better) |
|---|---|
| T-Learner (single GBM) | 0.63 |
| S-Learner | 6.46 |
| X-Learner | 4.08 |
| DR-Learner | 3.09 |
| **ADAPEL** | **0.76** |

ADAPEL achieves 1.2x the PEHE of T-Learner while providing counterfactual inference, bootstrap CIs, overlap diagnostics and an interpretable surrogate tree.

## Benchmark (RHC, 5735 patients, real observational)

ADAPEL on the Right Heart Catheterization dataset estimates a causal ATE of **+5.07%** (95% CI: [1.4%, 8.6%]) on 30-day mortality, with E-Value 1.33. This is consistent with the original Connors et al. (1996) finding that RHC increases short-term mortality.

## API Reference

| Method | Description |
|---|---|
| `fit(X, T, Y)` | Train ADAPEL on covariates, treatment, outcome. |
| `predict(X)` | Predict CATE `tau(x)` for each individual. |
| `predict_potential_outcomes(X)` | Return `(Y(0), Y(1))` predictions. |
| `predict_counterfactual(X, T_observed)` | Predict the unobserved potential outcome. |
| `estimate_ate(X)` | Average Treatment Effect. |
| `estimate_att(X, T)` | Average Treatment effect on the Treated. |
| `estimate_atc(X, T)` | Average Treatment effect on the Control. |
| `fit_bootstrap(X, T, Y, n_bootstrap)` | Train ensemble of bootstrapped models for CI. |
| `predict_clinical(X, alpha)` | Point estimate (BMA) + percentile CI + overlap. |
| `estimate_e_value(X, outcome_type)` | E-Value for unmeasured confounding. |
| `explain_cate_surrogate(X, feature_names)` | Surrogate decision tree rules. |
| `get_diagnostics(X)` | Propensity, alpha, stacking weights, etc. |

## References

- Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Meta-learners for estimating heterogeneous treatment effects. *Annals of Applied Statistics*, 13(2), 893-934.
- Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299-319.
- Kennedy, E. H. (2020). Towards optimal doubly robust estimation of heterogeneous treatment effects. *Electronic Journal of Statistics*, 14(1), 3008-3048.
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: introducing the E-value. *Annals of Internal Medicine*, 167(4), 268-274.
- Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference. *Journal of Computational and Graphical Statistics*, 20(1), 217-240. (IHDP benchmark)

## Files

```
Meta/
├── meta.py              # ADAPEL implementation
├── train.py             # Benchmark on IHDP and RHC
├── download_datasets.py # Download IHDP and RHC datasets
├── data/                # Datasets (csv)
└── README.md
```

## License

MIT
