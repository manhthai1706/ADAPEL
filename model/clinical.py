"""Clinical utility functions for ADAPEL."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm as _norm

from .nuisance import check_sample_size, clip_e


# ── Subgroup analysis ──


def _zscore_p(diff: float, se: float) -> float:
    if se <= 1e-10:
        return 1.0
    return 2.0 * (1.0 - _norm.cdf(abs(diff / se)))


def _subgroup_entry(name, mask, cate, overall_ate, sub_cate, p_val):
    return {
        "name": name,
        "n": int(mask.sum()),
        "size_pct": float(mask.mean() * 100),
        "cate_mean": float(sub_cate.mean()),
        "cate_std": float(sub_cate.std(ddof=1)),
        "ate_diff": float(sub_cate.mean() - overall_ate),
        "p_value": float(p_val),
    }


def _auto_subgroups(model_self, X, feature_names, n_bins):
    """Generate per-feature subgroup masks and their CATE summaries."""
    cate = model_self.predict(X)
    overall_ate = float(cate.mean())
    names = feature_names or [f"F{i}" for i in range(X.shape[1])]
    results = []

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
                p_val = _zscore_p(sub_cate.mean() - overall_ate, se)
                label = f"{names[j]}={v}"
                results.append(_subgroup_entry(label, mask, cate, overall_ate, sub_cate, p_val))
        else:
            bins = np.percentile(col, np.linspace(0, 100, n_bins + 1))
            for k in range(n_bins):
                lo, hi = bins[k], bins[k + 1]
                mask = (col >= lo) & (col <= hi if k == n_bins - 1 else col < hi)
                if mask.sum() < 5:
                    continue
                sub_cate = cate[mask]
                se = float(sub_cate.std(ddof=1)) / max(np.sqrt(len(sub_cate)), 1e-10)
                p_val = _zscore_p(sub_cate.mean() - overall_ate, se)
                if k == n_bins - 1:
                    label = f"{names[j]}[{lo:.2g},{hi:.2g}]"
                else:
                    label = f"{names[j]}[{lo:.2g},{hi:.2g})"
                results.append(_subgroup_entry(label, mask, cate, overall_ate, sub_cate, p_val))
    return overall_ate, results


def subgroup_analysis(
    model_self,
    X: ArrayLike,
    T: ArrayLike,
    Y: ArrayLike,
    subgroups: Optional[dict[str, np.ndarray]] = None,
    feature_names: Optional[list[str]] = None,
    n_bins: int = 4,
) -> dict:
    """CATE heterogeneity across user-defined or auto-binned subgroups."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    T = np.asarray(T, dtype=float).ravel()
    cate = model_self.predict(X)
    overall_ate = float(cate.mean())

    if subgroups is not None:
        results = []
        for name, mask in subgroups.items():
            sub_cate = cate[mask]
            if len(sub_cate) < 5:
                continue
            se = float(sub_cate.std(ddof=1)) / max(np.sqrt(len(sub_cate)), 1e-10)
            p_val = _zscore_p(sub_cate.mean() - overall_ate, se)
            results.append(_subgroup_entry(name, mask, cate, overall_ate, sub_cate, p_val))
    else:
        overall_ate, results = _auto_subgroups(model_self, X, feature_names, n_bins)

    return {"overall_ate": overall_ate, "subgroups": results}


# ── Variable importance ──


def variable_importance(
    model_self,
    X: ArrayLike,
    feature_names: Optional[list[str]] = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict:
    """Permutation-based variable importance for CATE predictions."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    baseline = model_self.predict(X)
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]

    importances = np.zeros((n_repeats, n_features))
    for f in range(n_features):
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, f] = rng.permutation(X_perm[:, f])
            importances[r, f] = np.mean((model_self.predict(X_perm) - baseline) ** 2)

    names = feature_names or [f"F{i}" for i in range(n_features)]
    return {
        "importances_mean": importances.mean(axis=0),
        "importances_std": importances.std(axis=0, ddof=1),
        "importances": importances,
        "feature_names": names,
    }


# ── Balance check ──


def _weighted_mean_var(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = w / w.sum()
    mean = (x * w[:, None]).sum(axis=0)
    var = ((x - mean) ** 2 * w[:, None]).sum(axis=0) / (1.0 - (w ** 2).sum())
    return mean, var


def _smd(x1: np.ndarray, x0: np.ndarray, w1=None, w0=None) -> np.ndarray:
    if w1 is None:
        m1, m0 = x1.mean(axis=0), x0.mean(axis=0)
        v1, v0 = x1.var(ddof=1, axis=0), x0.var(ddof=1, axis=0)
    else:
        m1, v1 = _weighted_mean_var(x1, w1)
        m0, v0 = _weighted_mean_var(x0, w0)
    pooled = np.sqrt((v1 + v0) / 2.0)
    return (m1 - m0) / np.maximum(pooled, 1e-10)


def balance_check(
    X: ArrayLike,
    T: ArrayLike,
    feature_names: Optional[list[str]] = None,
    weights: Optional[np.ndarray] = None,
) -> dict:
    """Standardised mean difference (SMD) before/after propensity weighting."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    T = np.asarray(T, dtype=float).ravel()
    names = feature_names or [f"F{i}" for i in range(X.shape[1])]

    i1, i0 = T == 1, T == 0
    smd_unweighted = _smd(X[i1], X[i0])
    result = {
        "smd_unweighted": dict(zip(names, smd_unweighted)),
        "smd_unweighted_max": float(np.max(np.abs(smd_unweighted))),
        "smd_unweighted_mean": float(np.mean(np.abs(smd_unweighted))),
    }

    if weights is not None:
        w = np.asarray(weights).ravel()
        if w.shape[0] == X.shape[0]:
            smd_weighted = _smd(X[i1], X[i0], w[i1], w[i0])
            result["smd_weighted"] = dict(zip(names, smd_weighted))
            result["smd_weighted_max"] = float(np.max(np.abs(smd_weighted)))
            result["smd_weighted_mean"] = float(np.mean(np.abs(smd_weighted)))
            result["smd_improvement"] = float(
                result["smd_unweighted_mean"] - result["smd_weighted_mean"]
            )

    threshold_exceeded = {
        "unweighted": [k for k, v in result["smd_unweighted"].items() if abs(v) > 0.1],
    }
    if "smd_weighted" in result:
        threshold_exceeded["weighted"] = [
            k for k, v in result["smd_weighted"].items() if abs(v) > 0.1
        ]
    result["threshold_exceeded"] = threshold_exceeded
    return result


# ── Negative control / placebo ──


def negative_control_test(
    model_self,
    X_outcome: ArrayLike,
    X_treatment: Optional[ArrayLike] = None,
    n_permute: int = 100,
    random_state: int = 42,
) -> dict:
    """Placebo test: permute treatment to check for spurious CATE signal."""
    X = np.atleast_2d(np.asarray(X_outcome, dtype=float))
    e = clip_e(
        model_self._prop_full.predict_proba(X)[:, 1],
        model_self.clip_propensity,
    )
    mu0, mu1 = model_self._m0_full.predict(X), model_self._m1_full.predict(X)

    rng = np.random.default_rng(random_state)
    perm_ates = np.empty(n_permute)
    for i in range(n_permute):
        T_perm = rng.permutation(e)
        pseudo = np.where(T_perm > 0.5, mu1, mu0)
        sw = (T_perm - e) ** 2
        sw = sw / max(float(sw.mean()), 1e-10)
        perm_ates[i] = float(np.average(pseudo, weights=sw))

    observed_ate = model_self.estimate_ate(X)
    p_val = float(np.mean(np.abs(perm_ates) >= abs(observed_ate)))
    return {
        "observed_ate": observed_ate,
        "permuted_ates_mean": float(perm_ates.mean()),
        "permuted_ates_std": float(perm_ates.std(ddof=1)),
        "p_value_placebo": p_val,
        "n_permute": n_permute,
    }


# ── Calibration ──


def calibration_check(model_self, X: ArrayLike, n_groups: int = 10) -> dict:
    """Calibration check by CATE quantile: predicted CATE vs observed diff."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    cate = model_self.predict(X)
    y0, y1 = model_self.predict_potential_outcomes(X)

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
        obs_diff = float(y1[mask].mean() - y0[mask].mean())
        groups.append({
            "n": int(mask.sum()),
            "cate_pred_mean": pred_mean,
            "outcome_diff_mean": obs_diff,
            "calib_error": float(pred_mean - obs_diff),
        })

    return {
        "groups": groups,
        "calib_error_overall": float(
            np.mean([g["calib_error"] for g in groups])
        ) if groups else 0.0,
        "n_groups": len(groups),
    }


# ── Fairness ──


def fairness_report(
    model_self,
    X: ArrayLike,
    protected_attributes: dict[str, np.ndarray],
    feature_names: Optional[list[str]] = None,
) -> dict:
    """CATE disparity across protected groups."""
    cate = model_self.predict(X)
    results = []

    for attr_name, attr_vals in protected_attributes.items():
        attr = np.asarray(attr_vals).ravel()
        for v in np.unique(attr):
            mask = attr == v
            if mask.sum() < 10 or (~mask).sum() < 10:
                continue
            sub_cate, comp_cate = cate[mask], cate[~mask]
            diff = float(sub_cate.mean() - comp_cate.mean())
            se = float(np.sqrt(
                sub_cate.var(ddof=1) / sub_cate.size
                + comp_cate.var(ddof=1) / comp_cate.size
            ))
            p_val = _zscore_p(diff, se)
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


# ── Sample size ──


def sample_size_report(model_self, X: ArrayLike, T: ArrayLike) -> dict:
    """Sample size adequacy report."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    T = np.asarray(T, dtype=float).ravel()
    n1, n0 = int(T.sum()), len(T) - int(T.sum())
    n = X.shape[0]
    p = X.shape[1]
    warnings = check_sample_size(X, T)
    return {
        "n": n,
        "n_treated": n1,
        "n_control": n0,
        "n_features": p,
        "treatment_ratio": max(n1, n0) / max(min(n1, n0), 1),
        "samples_per_feature": n / max(p, 1),
        "warnings": warnings,
        "adequate": len(warnings) == 0,
    }
