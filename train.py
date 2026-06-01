import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.base import clone

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

from adapel import ADAPEL


# ── Benchmarks don le de so sanh ─────────────────────────────────────────────

def benchmark_t_learner(X, T, Y, true_cate, n_splits=5, seed=42):
    pred = np.zeros(len(X))
    for tr, val in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        m = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed)
        m.fit(np.column_stack([X[tr], T[tr]]), Y[tr])
        pred[val] = m.predict(np.column_stack([X[val], np.ones(len(val))])) - m.predict(np.column_stack([X[val], np.zeros(len(val))]))
    return np.sqrt(np.mean((pred - true_cate) ** 2))


def benchmark_s_learner(X, T, Y, true_cate, n_splits=5, seed=42):
    pred = np.zeros(len(X))
    for tr, val in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        m = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed)
        m.fit(X[tr][T[tr] == 0], Y[tr][T[tr] == 0])
        pred[val] = -m.predict(X[val])
    return np.sqrt(np.mean((pred - true_cate) ** 2))


def benchmark_x_learner(X, T, Y, true_cate, n_splits=5, seed=42):
    pred = np.zeros(len(X))
    for tr, val in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        Xtr, Ttr, Ytr = X[tr], T[tr], Y[tr]
        m0 = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=seed).fit(Xtr[Ttr == 0], Ytr[Ttr == 0])
        m1 = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=seed).fit(Xtr[Ttr == 1], Ytr[Ttr == 1])
        d0 = m0.predict(Xtr[Ttr == 1]) - Ytr[Ttr == 1]
        d1 = Ytr[Ttr == 0] - m1.predict(Xtr[Ttr == 0])
        e = np.clip(np.mean(Ttr), 0.1, 0.9)
        tau0 = m0.predict(X[val])
        tau1 = m1.predict(X[val])
        pred[val] = e * (Ytr[Ttr == 1].mean() - m1.predict(Xtr[Ttr == 0]).mean()) + (1 - e) * (m0.predict(Xtr[Ttr == 1]).mean() - Ytr[Ttr == 0].mean())
    return np.sqrt(np.mean((pred - true_cate) ** 2))


def benchmark_dr_learner(X, T, Y, true_cate, n_splits=5, seed=42):
    pred = np.zeros(len(X))
    for tr, val in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        Xtr, Ttr, Ytr = X[tr], T[tr], Y[tr]
        m0 = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=seed).fit(Xtr[Ttr == 0], Ytr[Ttr == 0])
        m1 = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=seed).fit(Xtr[Ttr == 1], Ytr[Ttr == 1])
        e = np.clip(np.mean(Ttr), 0.1, 0.9)
        mu0, mu1 = m0.predict(X[val]), m1.predict(X[val])
        pred[val] = (mu1 - mu0) + (T[val] - e) / (e * (1 - e)) * (Y[val] - np.where(T[val] == 1, mu1, mu0))
    return np.sqrt(np.mean((pred - true_cate) ** 2))


def run_ihdp_benchmark():
    print("\n" + "=" * 70)
    print(" IHDP BENCHMARK: ADAPEL vs individual learners ")
    print("=" * 70)

    data = np.loadtxt("data/ihdp.csv", delimiter=",")
    T, Y, mu0, mu1 = data[:, 0], data[:, 1], data[:, 3], data[:, 4]
    X = data[:, 5:]
    true_cate = mu1 - mu0
    true_ate = np.mean(true_cate)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, T-rate: {T.mean():.1%}, true ATE: {true_ate:.4f}\n")

    pehe_t = benchmark_t_learner(X, T, Y, true_cate)
    pehe_s = benchmark_s_learner(X, T, Y, true_cate)
    pehe_x = benchmark_x_learner(X, T, Y, true_cate)
    pehe_dr = benchmark_dr_learner(X, T, Y, true_cate)

    learner = ADAPEL(n_folds=5, fusion_gamma=2.0, min_alpha=0.0)
    learner.fit_bootstrap(X, T, Y, n_bootstrap=15, random_state=42)
    clin = learner.predict_clinical(X)

    n_outside = sum(
        1 for i in range(len(X))
        if clin['lower_ci'][i] is not None and (clin['cate'][i] < clin['lower_ci'][i] or clin['cate'][i] > clin['upper_ci'][i])
    )
    coverage = 1 - n_outside / len(X)

    pehe_fusion = np.sqrt(np.mean((clin['cate'] - true_cate) ** 2))
    diag = learner.get_diagnostics(X)

    print(f"{'Method':<20} {'PEHE':<10} {'Note'}")
    print("-" * 70)
    print(f"{'T-Learner (GBM)':<20} {pehe_t:<10.4f} simple baseline")
    print(f"{'S-Learner (GBM)':<20} {pehe_s:<10.4f} confounding risk")
    print(f"{'X-Learner':<20} {pehe_x:<10.4f} imbalanced-friendly")
    print(f"{'DR-Learner':<20} {pehe_dr:<10.4f} doubly robust")
    print(f"{'ADAPEL':<20} {pehe_fusion:<10.4f} DR+X+R ensemble")
    print("-" * 70)
    print(f"Stacking weights: {diag['meta_weights']} (sum={diag['meta_weights'].sum():.3f})")
    print(f"DR-dominant: {diag['pct_dr_dominant']:.1%} | X-dominant: {diag['pct_x_dominant']:.1%}")
    print(f"CI coverage: {coverage:.1%} (target: >= 95%)")
    print(f"Overlap-safe: {clin['in_overlap'].mean():.1%} of patients")


def run_rhc_clinical():
    print("\n" + "=" * 70)
    print(" RHC CLINICAL EVALUATION: 30-day mortality ")
    print("=" * 70)

    df = pd.read_csv("data/rhc.csv")
    T = (df['swang1'] == 'RHC').astype(int).values
    Y = (df['death'] == 'Yes').astype(int).values

    cov_cols = [
        'age', 'sex', 'race', 'edu', 'income', 'ninsclas', 'cat1',
        'das2d3pc', 'dnr1', 'ca', 'surv2md1', 'aps1', 'scoma1', 'meanbp1',
        'wblc1', 'hrt1', 'resp1', 'temp1', 'pafi1', 'alb1', 'hema1', 'bili1',
        'crea1', 'sod1', 'pot1', 'paco21', 'ph1', 'cardiohx', 'chfhx',
        'dementhx', 'psychhx', 'chrpulhx', 'renalhx', 'liverhx', 'gibledhx',
        'malighx', 'immunhx', 'transhx', 'amihx'
    ]
    X_df = df[cov_cols].copy()
    from pandas.api.types import is_numeric_dtype
    for col in X_df.columns:
        if is_numeric_dtype(X_df[col]):
            X_df[col] = X_df[col].fillna(X_df[col].median())
        else:
            X_df[col] = X_df[col].fillna(X_df[col].mode()[0])
    X_df = pd.get_dummies(X_df, drop_first=True)
    X = X_df.astype(float).values

    print(f"Dataset: {X.shape[0]} patients, {X.shape[1]} features, RHC rate: {T.mean():.1%}, mortality: {Y.mean():.1%}")
    print(f"Naive RD: {Y[T==1].mean() - Y[T==0].mean():.4f} (CONFOUNDED)\n")

    learner = ADAPEL(n_folds=3, fusion_gamma=1.0, min_alpha=0.1, clip_propensity=0.05)
    learner.fit_bootstrap(X, T, Y, n_bootstrap=15, random_state=42)
    clin = learner.predict_clinical(X)

    n_outside = sum(
        1 for i in range(len(X))
        if clin['lower_ci'][i] is not None and (clin['cate'][i] < clin['lower_ci'][i] or clin['cate'][i] > clin['upper_ci'][i])
    )
    coverage = 1 - n_outside / len(X)

    ate = clin['cate'].mean()
    preds_all = np.column_stack([m.predict(X) for m in learner._bootstrap_learners])
    ate_lower, ate_upper = np.percentile(preds_all.mean(axis=0), [2.5, 97.5])
    e_val = learner.estimate_e_value(X, outcome_type="binary")
    diag = learner.get_diagnostics(X)

    print(f"{'Causal ATE':<25} {ate:.4f} (95% CI: [{ate_lower:.4f}, {ate_upper:.4f}])")
    print(f"{'E-Value':<25} {e_val:.4f}")
    print(f"{'Stacking weights':<25} {diag['meta_weights']}")
    print(f"{'DR-dominant | X-dominant':<25} {diag['pct_dr_dominant']:.1%} | {diag['pct_x_dominant']:.1%}")
    print(f"{'Per-sample CI coverage':<25} {coverage:.1%}")
    print(f"{'Overlap-safe patients':<25} {clin['in_overlap'].mean():.1%}")

    print(f"\nInterpretation: RHC {'INCREASES' if ate > 0 else 'DECREASES'} 30-day mortality by {abs(ate)*100:.2f}%.")
    print(f"E-Value {e_val:.2f}: unmeasured confounder needs RR >= {e_val:.2f} with both RHC and death to explain this away.")

    print("\nSurrogate decision rules:")
    print(learner.explain_cate_surrogate(X, feature_names=list(X_df.columns), max_depth=3))


if __name__ == "__main__":
    run_ihdp_benchmark()
    run_rhc_clinical()
