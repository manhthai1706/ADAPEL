from __future__ import annotations
from typing import Optional, Literal
import numpy as np
from numpy.typing import ArrayLike
from sklearn.tree import DecisionTreeRegressor, export_text
from .nuisance import clip_e, alpha


def compute_diagnostics(model_self, X: ArrayLike) -> dict:
    """Compute per-sample diagnostics for a fitted ADAPEL.

    Returns
    -------
    dict with keys: propensity, alpha, meta_weights,
                    pct_dr_dominant, pct_x_dominant,
                    ensemble_std, t_res_std_train.
    """
    e = clip_e(
        model_self._prop_full.predict_proba(X)[:, 1],
        model_self.clip_propensity,
    )
    a = alpha(e, model_self.fusion_gamma, model_self.min_alpha)
    ensemble_preds = np.column_stack([
        m.predict(X) if m is not None else np.zeros(X.shape[0])
        for m in model_self._fitted_finals
    ])
    return {
        "propensity": e,
        "alpha": a,
        "meta_weights": model_self._meta.coef_,
        "pct_dr_dominant": float((a < 0.5).mean()),
        "pct_x_dominant": float((a >= 0.5).mean()),
        "ensemble_std": ensemble_preds.std(axis=1),
        "t_res_std_train": model_self._t_res_std,
    }


def estimate_e_value(
    model_self, X: ArrayLike, outcome_type: Literal["binary", "continuous"] = "binary"
) -> float:
    """Compute E-Value (VanderWeele & Ding 2017) for unmeasured confounding.

    The E-Value is the minimum strength of association (on the risk-ratio
    scale) that an unmeasured confounder would need to have with both
    treatment and outcome to explain away the observed ATE.
    """
    ate = model_self.estimate_ate(X)
    if outcome_type == "binary":
        p0 = float(np.clip(
            np.mean(model_self._m0_full.predict(X)), 1e-5, 1 - 1e-5
        ))
        p1 = float(np.clip(
            np.mean(model_self._m1_full.predict(X)), 1e-5, 1 - 1e-5
        ))
        rr = max(p1 / p0, p0 / p1)
    else:
        y0, y1 = model_self.predict_potential_outcomes(X)
        std = max(float(np.std(np.concatenate([y0, y1]))), 1e-5)
        rr = np.exp(0.91 * abs(ate / std))
    if rr <= 1.0:
        return 1.0
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def explain_surrogate(
    model_self,
    X: ArrayLike,
    feature_names: Optional[list] = None,
    max_depth: int = 3,
) -> str:
    """Fit a shallow decision tree to explain CATE predictions with rules.

    Parameters
    ----------
    feature_names : list of str, optional
        Names for each column of X.
    max_depth : int
        Maximum depth of the surrogate tree.

    Returns
    -------
    str
        Text-based decision rules.
    """
    surrogate = DecisionTreeRegressor(max_depth=max_depth, random_state=42).fit(
        X, model_self.predict(X)
    )
    names = feature_names or [f"F{i}" for i in range(X.shape[1])]
    out = []
    for line in export_text(surrogate, feature_names=names).split("\n"):
        if line.strip():
            out.append(line)
    return "\n".join(out)
