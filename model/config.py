"""ADAPEL config — dataclass + mode presets (single source of truth)."""
from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """All hyperparameters for ADAPEL.

    Single source of truth: ``MODE_PRESETS`` is ``dict[str, ModelConfig]``;
    callers can either pick a preset (via ``mode=...``) or pass a fully
    customised ``ModelConfig`` instance directly.
    """

    # ── Cross-validation / OOF ──
    n_folds: int = 3
    oof_frac: float = 0.5
    min_samples_per_arm: int = 30

    # ── Pseudo-outcome fusion ──
    fusion_gamma: float = 1.0
    min_alpha: float = 0.1
    clip_propensity: float = 0.05

    # ── Feature selection ──
    feature_select: bool = False
    feature_frac: float = 0.5

    # ── Outcome model (HistGradientBoostingRegressor) ──
    outcome_iter: int = 150
    outcome_depth: int = 6
    outcome_lr: float = 0.05

    # ── Propensity model (HistGradientBoostingClassifier) ──
    prop_iter: int = 150
    prop_depth: int = 6

    # ── Base learner #1: HistGradientBoostingRegressor ──
    gbm_iter: int = 200
    gbm_depth: int = 5
    gbm_lr: float = 0.05

    # ── Base learner #2: ExtraTreesRegressor ──
    et_n: int = 200
    et_leaf: int = 5
    et_max_features: float = 0.7

    # ── Base learner #3: Ridge ──
    ridge_alpha: float = 1.0

    # ── Base learner #4: DecisionTreeRegressor ──
    dt_max_depth: int = 5
    dt_min_samples_leaf: int = 10

    # ── Base learner #5: Lasso ──
    lasso_alpha: float = 0.01
    lasso_max_iter: int = 5000

    # ── Bootstrap ──
    boot_frac: float = 0.7

    # ── Misc ──
    random_state: int = 42


MODE_PRESETS: dict[str, ModelConfig] = {
    "fast": ModelConfig(
        outcome_iter=80, outcome_depth=4,
        prop_iter=80, prop_depth=4,
        gbm_iter=100, gbm_depth=4,
        et_n=100, et_leaf=10,
        oof_frac=0.35, boot_frac=0.5,
    ),
    "balanced": ModelConfig(
        outcome_iter=150, outcome_depth=6,
        prop_iter=150, prop_depth=6,
        gbm_iter=200, gbm_depth=5,
        et_n=200, et_leaf=5,
        oof_frac=0.5, boot_frac=0.7,
    ),
    "accurate": ModelConfig(
        outcome_iter=300, outcome_depth=8,
        prop_iter=300, prop_depth=8,
        gbm_iter=400, gbm_depth=6,
        et_n=500, et_leaf=3,
        oof_frac=0.7, boot_frac=0.85,
    ),
}


def get_config(mode: Optional[str] = None, **overrides) -> ModelConfig:
    """Resolve a :class:`ModelConfig` from a mode name + per-field overrides."""
    base = MODE_PRESETS[mode] if mode else ModelConfig()
    return dataclasses.replace(base, **overrides) if overrides else base