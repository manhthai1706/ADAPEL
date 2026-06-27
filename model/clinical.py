"""Clinical utility functions for ADAPEL.

Includes subgroup analysis, variable importance, balance checking,
negative control testing, calibration checks, and fairness metrics.
"""

from __future__ import annotations
from typing import Optional, Literal
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor
from .nuisance import clip_e, check_sample_size


def subgroup_analysis(
    model_self,
    X: ArrayLike,
    T: ArrayLike,
    Y: ArrayLike,
    subgroups: Optional[dict[str, np.ndarray]] = None,
    feature_names: Optional[list[str]] = None,
    n_bins: int = 4,
) -> dict:
    """Analyse CATE heterogeneity across subgroups.

    Parameters
    ----------
    subgroups : dict of str -> boolean mask, optional
        Pre-defined subgroup masks. If None, creates subgroups
        based on each feature quartile.
    feature_names : list of str, optional
        Names of features for reporting.
    n_bins : int
        Number of bins for automatic subgroup creation (per feature).

    Returns
    -------
    dict with keys: subgroups (list of dict with name, n, size_pct,
    cate_mean, cate_std, ate_diff_from_overall, p_quartile).
    """
    from scipy.stats import norm as _norm

    cate = model_self.predict(X)
    overall_ate = float(cate.mean())

    results = []
    if subgroups is not None:
        for name, mask in subgroups.items():
            sub_cate = cate[mask]
            if len(sub_cate) < 5:
                continue
            se = float(sub_cate.std(ddof=1)) / max(np.sqrt(len(sub_cate)), 1e-10)
            z = (sub_cate.mean() - overall_ate) / max(se, 1e-10)
            p_val = 2.0 * (1.0 - _norm.cdf(abs(z)))
            results.append(_subgroup_entry(name, mask, cate, overall_ate, sub_cate, p_val))
    else:
        names = feature_names or [f"F{i}" for i in range(X.shape[1])]
        for j in range(X.shape[1]):
            col = X[:, j]
            unique_vals = np.unique(col)
            if len(unique_vals) <= n_bins:
                for v in unique_vals:
                    mask = col == v
                    if mask.sum() < 5:
                        continue
                    sub_cate = cate[mask]
                    se = float(sub_cate.std(ddof=1)) / max(np.sqrt(len(sub_cate)), 1e-10)
                    z = (sub_cate.mean() - overall_ate) / max(se, 1e-10)
                    p_val = 2.0 * (1.0 - _norm.cdf(abs(z)))
                    name = f"{names[j]}={v:.2g}" if isinstance(v, (int, float)) else f"{names[j]}={v}"
                    results.append(_subgroup_entry(name, mask, cate, overall_ate, sub_cate, p_val))
            else:
                bins = np.percentile(col, np.linspace(0, 100, n_bins + 1))
                for k in range(n_bins):
                    lo, hi = bins[k], bins[k + 1]
                    if k == n_bins - 1:
                        mask = (col >= lo) & (col <= hi)
                    else:
                        mask = (col >= lo) & (col < hi)
                    if mask.sum() < 5:
                        continue
                    sub_cate = cate[mask]
                    se = float(sub_cate.std(ddof=1)) / max(np.sqrt(len(sub_cate)), 1e-10)
                    z = (sub_cate.mean() - overall_ate) / max(se, 1e-10)
                    p_val = 2.0 * (1.0 - _norm.cdf(abs(z)))
                    name = f"{names[j]}[{lo:.2g},{hi:.2g})"
                    if k == n_bins - 1:
                        name = f"{names[j]}[{lo:.2g},{hi:.2g}]"
                    results.append(_subgroup_entry(name, mask, cate, overall_ate, sub_cate, p_val))

    return {"overall_ate": overall_ate, "subgroups": results}


def _subgroup_entry(name, mask, cate_all, overall_ate, sub_cate, p_val):
    return {
        "name": name,
        "n": int(mask.sum()),
        "size_pct": float(mask.mean() * 100),
        "cate_mean": float(sub_cate.mean()),
        "cate_std": float(sub_cate.std(ddof=1)),
        "ate_diff": float(sub_cate.mean() - overall_ate),
        "p_value": float(p_val),
    }


def variable_importance(
    model_self,
    X: ArrayLike,
    feature_names: Optional[list[str]] = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict:
    """Permutation-based variable importance for CATE predictions.

    Returns
    -------
    dict with keys: importances_mean, importances_std, importances (per repeat).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    baseline = model_self.predict(X)

    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    importances = np.zeros((n_repeats, n_features))

    for r in range(n_repeats):
        for f in range(n_features):
            X_perm = X.copy()
            X_perm[:, f] = rng.permutation(X_perm[:, f])
            perm_pred = model_self.predict(X_perm)
            importances[r, f] = np.mean((perm_pred - baseline) ** 2)

    names = feature_names or [f"F{i}" for i in range(n_features)]
    return {
        "importances_mean": importances.mean(axis=0),
        "importances_std": importances.std(axis=0, ddof=1),
        "importances": importances,
        "feature_names": names,
    }


def balance_check(
    X: ArrayLike,
    T: ArrayLike,
    feature_names: Optional[list[str]] = None,
    weights: Optional[np.ndarray] = None,
) -> dict:
    """Check covariate balance via standardized mean difference (SMD).

    Compares treated vs control before and after weighting.

    Parameters
    ----------
    weights : np.ndarray, optional
        Sample weights (e.g., inverse propensity weights).
        If None, only unweighted SMD is reported.

    Returns
    -------
    dict with keys: smd_unweighted, smd_weighted, threshold_exceeded
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    T = np.asarray(T, dtype=float).ravel()
    n_features = X.shape[1]
    names = feature_names or [f"F{i}" for i in range(n_features)]

    i1 = T == 1
    i0 = T == 0

    def _smd(x1, x0, w1=None, w0=None):
        if w1 is None:
            m1, m0 = x1.mean(axis=0), x0.mean(axis=0)
            v1, v0 = x1.var(ddof=1, axis=0), x0.var(ddof=1, axis=0)
        else:
            w1 = w1 / w1.sum()
            w0 = w0 / w0.sum()
            m1 = (x1 * w1[:, None]).sum(axis=0)
            m0 = (x0 * w0[:, None]).sum(axis=0)
            v1 = ((x1 - m1) ** 2 * w1[:, None]).sum(axis=0) / (1.0 - (w1 ** 2).sum())
            v0 = ((x0 - m0) ** 2 * w0[:, None]).sum(axis=0) / (1.0 - (w0 ** 2).sum())
        pooled_std = np.sqrt((v1 + v0) / 2.0)
        pooled_std = np.maximum(pooled_std, 1e-10)
        return (m1 - m0) / pooled_std

    smd_unweighted = _smd(X[i1], X[i0])
    result = {
        "smd_unweighted": dict(zip(names, smd_unweighted)),
        "smd_unweighted_max": float(np.max(np.abs(smd_unweighted))),
        "smd_unweighted_mean": float(np.mean(np.abs(smd_unweighted))),
    }

    if weights is not None:
        w = np.asarray(weights).ravel()
        if w.shape[0] == X.shape[0]:
            # For ATE weighting: w_treated = 1, w_control = e/(1-e)
            # Use IPW-style: treated weight = 1/e, control weight = 1/(1-e)
            w1 = w[i1] / w[i1].sum()
            w0 = w[i0] / w[i0].sum()
            smd_weighted = _smd(X[i1], X[i0], w1, w0)
            result["smd_weighted"] = dict(zip(names, smd_weighted))
            result["smd_weighted_max"] = float(np.max(np.abs(smd_weighted)))
            result["smd_weighted_mean"] = float(np.mean(np.abs(smd_weighted)))
            result["smd_improvement"] = float(
                np.mean(np.abs(smd_unweighted)) - np.mean(np.abs(smd_weighted))
            )

    threshold_exceeded = {
        "unweighted": [k for k, v in result.get("smd_unweighted", {}).items() if abs(v) > 0.1],
    }
    if "smd_weighted" in result:
        threshold_exceeded["weighted"] = [
            k for k, v in result["smd_weighted"].items() if abs(v) > 0.1
        ]
    result["threshold_exceeded"] = threshold_exceeded
    return result


def negative_control_test(
    model_self,
    X_outcome: ArrayLike,
    X_treatment: Optional[ArrayLike] = None,
    n_permute: int = 100,
    random_state: int = 42,
) -> dict:
    """Negative control outcome / placebo test.

    Checks for spurious CATE signal by permuting the treatment-outcome
    relationship or testing with a known null (negative control) outcome.

    Parameters
    ----------
    X_outcome : array-like
        Covariates for the negative control outcome.
    X_treatment : array-like, optional
        Different covariates for treatment assignment (for placebo test).
        If None, uses X_outcome.
    n_permute : int
        Number of permutations for the placebo test.

    Returns
    -------
    dict with keys: observed_ate, permuted_ates, p_value_placebo.
    """
    from .model import ADAPEL

    X = np.atleast_2d(np.asarray(X_outcome, dtype=float))
    e_pred = clip_e(
        model_self._prop_full.predict_proba(X)[:, 1],
        model_self.clip_propensity,
    )
    # Placebo test: permute treatment labels
    rng = np.random.default_rng(random_state)
    perm_ates = []
    for _ in range(n_permute):
        T_perm = rng.permutation(e_pred)
        sw_perm = (T_perm - e_pred) ** 2
        sw_perm = sw_perm / max(float(sw_perm.mean()), 1e-10)
        mu0 = model_self._m0_full.predict(X)
        mu1 = model_self._m1_full.predict(X)
        pseudo = np.where(
            T_perm == 1,
            model_self.predict(X),
            model_self.predict(X),
        )
        perm_ates.append(float(np.average(pseudo, weights=sw_perm)))

    observed_ate = model_self.estimate_ate(X)
    perm_ates = np.array(perm_ates)
    p_val = float(np.mean(np.abs(perm_ates) >= abs(observed_ate)))

    return {
        "observed_ate": observed_ate,
        "permuted_ates_mean": float(perm_ates.mean()),
        "permuted_ates_std": float(perm_ates.std(ddof=1)),
        "p_value_placebo": p_val,
        "n_permute": n_permute,
    }


def calibration_check(
    model_self,
    X: ArrayLike,
    n_groups: int = 10,
) -> dict:
    """Calibration check: compare mean predicted CATE vs observed
    outcome difference within groups defined by CATE quantiles.

    Returns
    -------
    dict with keys: groups (list of dict with n, cate_pred_mean,
    outcome_diff_mean, calib_error).
    """
    cate = model_self.predict(X)
    y0, y1 = model_self.predict_potential_outcomes(X)
    # Approx observed diff using pseudo-outcome
    e_pred = clip_e(
        model_self._prop_full.predict_proba(X)[:, 1],
        model_self.clip_propensity,
    )
    # DR pseudo-outcome as observed CATE proxy
    pseudo_dr = (y1 - y0) + (model_self._t_res_std * np.random.randn(X.shape[0]) * 0)

    # Group by predicted CATE quantile
    quantiles = np.percentile(cate, np.linspace(0, 100, n_groups + 1))
    groups = []
    for k in range(n_groups):
        lo, hi = quantiles[k], quantiles[k + 1]
        if k == n_groups - 1:
            mask = cate >= lo
        else:
            mask = (cate >= lo) & (cate < hi)
        if mask.sum() < 5:
            continue
        pred_mean = float(cate[mask].mean())
        # Use difference in potential outcomes as observed proxy
        y0_m, y1_m = y0[mask].mean(), y1[mask].mean()
        obs_diff = float(y1_m - y0_m)
        groups.append({
            "n": int(mask.sum()),
            "cate_pred_mean": pred_mean,
            "outcome_diff_mean": obs_diff,
            "calib_error": float(pred_mean - obs_diff),
        })

    calib_error_overall = float(np.mean([g["calib_error"] for g in groups]))
    return {
        "groups": groups,
        "calib_error_overall": calib_error_overall,
        "n_groups": len(groups),
    }


def fairness_report(
    model_self,
    X: ArrayLike,
    protected_attributes: dict[str, np.ndarray],
    feature_names: Optional[list[str]] = None,
) -> dict:
    """Fairness assessment: compare CATE across protected groups.

    Parameters
    ----------
    protected_attributes : dict of str -> array-like
        Each entry is a binary/categorical vector defining a group.

    Returns
    -------
    dict with keys: groups (list of dict with name, n, cate_mean,
    cate_std, disparity_vs_complement, p_value).
    """
    from scipy.stats import norm as _norm

    cate = model_self.predict(X)
    results = []

    for attr_name, attr_vals in protected_attributes.items():
        attr = np.asarray(attr_vals).ravel()
        unique_vals = np.unique(attr)
        for v in unique_vals:
            mask = attr == v
            if mask.sum() < 10:
                continue
            comp = ~mask
            if comp.sum() < 10:
                continue
            sub_cate = cate[mask]
            comp_cate = cate[comp]
            diff = float(sub_cate.mean() - comp_cate.mean())
            se = float(np.sqrt(
                sub_cate.var(ddof=1) / sub_cate.size
                + comp_cate.var(ddof=1) / comp_cate.size
            ))
            z = diff / max(se, 1e-10)
            p_val = 2.0 * (1.0 - _norm.cdf(abs(z)))
            results.append({
                "attribute": attr_name,
                "group": str(v),
                "n": int(mask.sum()),
                "cate_mean": float(sub_cate.mean()),
                "cate_std": float(sub_cate.std(ddof=1)),
                "disparity_vs_complement": diff,
                "p_value": float(p_val),
            })

    return {"groups": results}


def sample_size_report(model_self, X: ArrayLike, T: ArrayLike) -> dict:
    """Generate sample size adequacy report."""
    X_arr = np.atleast_2d(np.asarray(X, dtype=float))
    T_arr = np.asarray(T, dtype=float).ravel()
    n = X_arr.shape[0]
    n1, n0 = int(T_arr.sum()), int((1 - T_arr).sum())
    p = X_arr.shape[1]
    ratio = max(n1, n0) / max(min(n1, n0), 1)
    n_per_feature = n / max(p, 1)

    warnings = check_sample_size(X_arr, T_arr)

    return {
        "n": n,
        "n_treated": n1,
        "n_control": n0,
        "n_features": p,
        "treatment_ratio": ratio,
        "samples_per_feature": n_per_feature,
        "warnings": warnings,
        "adequate": len(warnings) == 0,
    }
