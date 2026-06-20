"""
train.py — ADAPEL practical test

Chạy được ngay (tự sinh synthetic data + load data từ paper/ nếu có).
Output: PEHE, ATE, stacking weights, runtime, CI coverage.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except: pass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from adapel import ADAPEL


# ── helpers ──

T_LEARNER_GBM = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
)

def run_t_learner(X_tr, T_tr, Y_tr, X_te, true_cate):
    m = clone(T_LEARNER_GBM).fit(np.column_stack([X_tr, T_tr]), Y_tr)
    pred = (m.predict(np.column_stack([X_te, np.ones(len(X_te))]))
            - m.predict(np.column_stack([X_te, np.zeros(len(X_te))])))
    return np.sqrt(np.mean((pred - true_cate) ** 2))

def fmt_time(s):
    if s < 60: return f"{s:.2f}s"
    return f"{s//60}m {s%60:.0f}s"


# ── 1. Synthetic data ──

def test_synthetic():
    print("=" * 65)
    print("  1. SYNTHETIC DATA — realistic confounding + non-linear CATE")
    print("=" * 65)

    rng = np.random.default_rng(42)
    n, p = 2000, 20

    X = np.zeros((n, p))
    X[:, :10] = rng.normal(0, 1, (n, 10))
    X[:, 10:15] = rng.binomial(1, 0.4, (n, 5))
    X[:, 15:] = rng.normal(0, 1, (n, 5))

    # Confounded propensity
    logit = -1.5 + 0.3*X[:,0] - 0.2*X[:,1] + 0.5*X[:,10] - 0.3*X[:,11]
    T = rng.binomial(1, 1/(1+np.exp(-logit)))

    # Non-linear CATE
    true_cate = 0.8*X[:,0] + 0.5*np.sin(X[:,2]) - 0.3*X[:,1]*X[:,3] + 0.4*X[:,10]
    mu0 = (0.5*X[:,0] - 0.3*X[:,1] + 0.2*X[:,2]**2 + 0.1*X[:,10]
           + 0.05*X[:,11]*X[:,3] - 0.1*np.abs(X[:,4]))
    Y = np.where(T == 1, mu0 + true_cate, mu0) + rng.normal(0, 0.3*mu0.std(), n)

    idx = rng.permutation(n)
    tr, te = idx[:1500], idx[1500:]

    print(f"  Train: {len(tr)} samples, T-rate: {T[tr].mean():.1%}")
    print(f"  Test:  {len(te)} samples, T-rate: {T[te].mean():.1%}")
    print(f"  True ATE: {true_cate[te].mean():.4f} | Naive RD: {Y[T==1].mean()-Y[T==0].mean():.4f}")

    # T-Learner
    t0 = time.time()
    pehe_t = run_t_learner(X[tr], T[tr], Y[tr], X[te], true_cate[te])
    t_t = time.time() - t0

    # ADAPEL
    t0 = time.time()
    model = ADAPEL(n_folds=5).fit(X[tr], T[tr], Y[tr])
    cate = model.predict(X[te])
    pehe_a = np.sqrt(np.mean((cate - true_cate[te])**2))
    t_a = time.time() - t0
    d = model.get_diagnostics(X[te])

    # ADAPEL + feature selection
    t0 = time.time()
    model_fs = ADAPEL(n_folds=5, feature_select=True).fit(X[tr], T[tr], Y[tr])
    pehe_fs = np.sqrt(np.mean((model_fs.predict(X[te]) - true_cate[te])**2))
    t_fs = time.time() - t0

    # Bootstrap CI
    model.fit_bootstrap(X[tr], T[tr], Y[tr], n_bootstrap=30)
    clin = model.predict_clinical(X[te])
    cov = np.mean((clin["lower_ci"] <= true_cate[te]) & (true_cate[te] <= clin["upper_ci"]))

    ate = model.estimate_ate(X[te])

    print(f"\n  {'Method':<25} {'PEHE':<10} {'Time':<10}")
    print(f"  {'-'*45}")
    print(f"  {'T-Learner (GBM)':<25} {pehe_t:<10.4f} {fmt_time(t_t):<10}")
    print(f"  {'ADAPEL':<25} {pehe_a:<10.4f} {fmt_time(t_a):<10}")
    print(f"  {'ADAPEL + feat select':<25} {pehe_fs:<10.4f} {fmt_time(t_fs):<10}")
    print(f"  {'ADAPEL ATE':<25} {ate:.4f} (true: {true_cate[te].mean():.4f})")
    print(f"  {'Stacking weights':<25} {np.round(d['meta_weights'], 3)} (active: {(d['meta_weights']>1e-8).sum()}/5)")
    print(f"  {'DR-dominant':<25} {d['pct_dr_dominant']:.1%}")
    print(f"  {'Boot CI coverage':<25} {cov:.1%}")
    return model


# ── 2. IHDP (real covariates, semi-synthetic) ──

def test_ihdp(path="data/ihdp/ihdp.csv", alt="paper/ihdp/ihdp.csv"):
    f = path if __import__("os").path.exists(path) else alt
    print("\n" + "=" * 65)
    print(f"  2. IHDP — semi-synthetic (747 samples, 25 features)")
    print("=" * 65)

    data = np.loadtxt(f, delimiter=",")
    T, Y, mu0, mu1 = data[:, 0], data[:, 1], data[:, 3], data[:, 4]
    X = data[:, 5:]
    true_cate = mu1 - mu0

    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, T-rate: {T.mean():.1%}")
    print(f"  True ATE: {true_cate.mean():.4f}")

    pehe_t = run_t_learner(X, T, Y, X, true_cate)

    t0 = time.time()
    model = ADAPEL(n_folds=5, fusion_gamma=2.0, min_alpha=0.0).fit(X, T, Y)
    cate = model.predict(X)
    pehe_a = np.sqrt(np.mean((cate - true_cate)**2))
    t_a = time.time() - t0
    d = model.get_diagnostics(X)

    print(f"\n  {'Method':<25} {'PEHE':<10} {'Time':<10}")
    print(f"  {'-'*45}")
    print(f"  {'T-Learner (GBM)':<25} {pehe_t:<10.4f} {'--':<10}")
    print(f"  {'ADAPEL':<25} {pehe_a:<10.4f} {fmt_time(t_a):<10}")
    print(f"  {'Weights':<25} {np.round(d['meta_weights'], 3)} (active: {(d['meta_weights']>1e-8).sum()}/5)")
    print(f"  {'DR-dominant':<25} {d['pct_dr_dominant']:.1%}")
    return model


# ── 3. RHC (real observational) ──

def test_rhc(path="data/rhc/rhc.csv", alt="paper/rhc/rhc.csv"):
    f = path if __import__("os").path.exists(path) else alt
    print("\n" + "=" * 65)
    print("  3. RHC — real observational (5735 patients, 39 features)")
    print("=" * 65)

    df = pd.read_csv(f)
    T = (df["swang1"] == "RHC").astype(int).values
    Y = (df["death"] == "Yes").astype(int).values
    cov_cols = ["age","sex","race","edu","income","ninsclas","cat1","das2d3pc",
        "dnr1","ca","surv2md1","aps1","scoma1","meanbp1","wblc1","hrt1","resp1",
        "temp1","pafi1","alb1","hema1","bili1","crea1","sod1","pot1","paco21",
        "ph1","cardiohx","chfhx","dementhx","psychhx","chrpulhx","renalhx",
        "liverhx","gibledhx","malighx","immunhx","transhx","amihx"]
    X_df = df[cov_cols].copy()
    for col in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = X_df[col].fillna(X_df[col].median())
        else:
            X_df[col] = X_df[col].fillna(X_df[col].mode()[0])
    X = pd.get_dummies(X_df, drop_first=True).astype(float).values

    print(f"  Patients: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  RHC rate: {T.mean():.1%}, Mortality: {Y.mean():.1%}")
    print(f"  Naive RD: {Y[T==1].mean()-Y[T==0].mean():.4f} (confounded)")

    t0 = time.time()
    model = ADAPEL(n_folds=3).fit(X, T, Y)
    ate = model.estimate_ate(X)
    t_a = time.time() - t0
    e_val = model.estimate_e_value(X, "binary")
    d = model.get_diagnostics(X)

    model.fit_bootstrap(X, T, Y, n_bootstrap=15)
    clin = model.predict_clinical(X)
    ate_boot = clin["cate"].mean()

    print(f"\n  {'ADAPEL ATE (point)':<25} {ate:.4f}")
    print(f"  {'ADAPEL ATE (BMA)':<25} {ate_boot:.4f}")
    print(f"  {'E-Value':<25} {e_val:.4f}")
    print(f"  {'Weights':<25} {np.round(d['meta_weights'], 3)}")
    print(f"  {'Fit time':<25} {fmt_time(t_a)}")
    print(f"  {'Interpretation: RHC':<25} {'INCREASES' if ate>0 else 'DECREASES'} mortality by {abs(ate)*100:.2f}%")


# ── 4. Hillstrom (RCT benchmark) ──

def test_hillstrom(path="data/hillstrom/hillstrom.csv", alt="paper/hillstrom/hillstrom.csv"):
    f = path if __import__("os").path.exists(path) else alt
    print("\n" + "=" * 65)
    print("  4. HILLSTROM — RCT benchmark (64000 customers)")
    print("=" * 65)

    df = pd.read_csv(f)
    T = np.where(df["segment"] != "No E-Mail", 1, 0)
    X_df = df[["recency","history","mens","womens","newbie"]].copy()
    for name in ("zip_code", "channel"):
        X_df = pd.concat([X_df, pd.get_dummies(df[name], prefix=name, drop_first=True)], axis=1)
    X = X_df.astype(float).values
    Y = df["spend"].values

    ate_rct = Y[T==1].mean() - Y[T==0].mean()

    t0 = time.time()
    model = ADAPEL(n_folds=3).fit(X, T, Y)
    ate = model.estimate_ate(X)
    t_a = time.time() - t0
    d = model.get_diagnostics(X)

    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, Email rate: {T.mean():.1%}")
    print(f"  RCT ATE: {ate_rct:.4f}")
    print(f"  ADAPEL ATE: {ate:.4f} (diff: {abs(ate-ate_rct):.4f})")
    print(f"  Weights: {np.round(d['meta_weights'], 3)}")
    print(f"  Time: {fmt_time(t_a)}")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    t_start = time.time()

    test_synthetic()
    test_ihdp()
    test_rhc()
    test_hillstrom()

    print("\n" + "=" * 65)
    print(f"  DONE — total: {fmt_time(time.time() - t_start)}")
    print("=" * 65)
