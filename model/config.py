MODE_PRESETS = {
    "fast": {
        "outcome_iter": 80, "outcome_depth": 4,
        "prop_iter": 80, "prop_depth": 4,
        "gbm_iter": 100, "gbm_depth": 4,
        "et_n": 100, "et_leaf": 10,
        "oof_frac": 0.35, "boot_frac": 0.5,
    },
    "balanced": {
        "outcome_iter": 150, "outcome_depth": 6,
        "prop_iter": 150, "prop_depth": 6,
        "gbm_iter": 200, "gbm_depth": 5,
        "et_n": 200, "et_leaf": 5,
        "oof_frac": 0.5, "boot_frac": 0.7,
    },
    "accurate": {
        "outcome_iter": 300, "outcome_depth": 8,
        "prop_iter": 300, "prop_depth": 8,
        "gbm_iter": 400, "gbm_depth": 6,
        "et_n": 500, "et_leaf": 3,
        "oof_frac": 0.7, "boot_frac": 0.85,
    },
}
