import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype

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

    data = np.loadtxt("data/ihdp/ihdp.csv", delimiter=",")
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

    df = pd.read_csv("data/rhc/rhc.csv")
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


# ── Lalonde (Labor Economics) ─────────────────────────────────────────────────

def load_lalonde():
    df = pd.read_csv("data/lalonde/lalonde.csv")
    T = df["treat"].values
    Y = df["re78"].values
    cov_cols = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]
    X = df[cov_cols].values
    return X, T, Y


def run_lalonde_benchmark():
    print("\n" + "=" * 70)
    print(" LALONDE BENCHMARK: Labor training (NSW experimental)")
    print("=" * 70)

    X, T, Y = load_lalonde()
    # Normalize earnings to $1000s for numerical stability
    Y = Y / 1000.0

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, T-rate: {T.mean():.1%}")
    print(f"Mean outcome: treated={Y[T==1].mean():.3f}k, control={Y[T==0].mean():.3f}k")
    print(f"Naive ATE (RCT benchmark): {(Y[T==1].mean() - Y[T==0].mean()):.3f}k\n")

    learner = ADAPEL(n_folds=3, fusion_gamma=1.0, min_alpha=0.1, clip_propensity=0.05)
    learner.fit(X, T, Y)
    cate = learner.predict(X)
    ate = cate.mean()
    diag = learner.get_diagnostics(X)

    print(f"{'ADAPEL ATE':<25} {ate:.4f}k")
    print(f"{'Naive diff-in-means':<25} {(Y[T==1].mean() - Y[T==0].mean()):.4f}k")
    print(f"{'Stacking weights':<25} {diag['meta_weights']}")
    print(f"{'DR-dominant | X-dominant':<25} {diag['pct_dr_dominant']:.1%} | {diag['pct_x_dominant']:.1%}")


# ── Hillstrom (Marketing / RCT) ──────────────────────────────────────────────

def load_hillstrom():
    df = pd.read_csv("data/hillstrom/hillstrom.csv")
    # Binary treatment: 1 = received email (Mens or Womens), 0 = No E-Mail
    T = np.where(df["segment"] != "No E-Mail", 1, 0)
    # Outcome: spend ($)
    Y = df["spend"].values
    # Covariates
    cov_cols = ["recency", "history", "mens", "womens", "newbie"]
    X_df = df[cov_cols].copy()
    cat_cols = {
        "zip_code": df["zip_code"],
        "channel": df["channel"],
    }
    for name, col in cat_cols.items():
        dummies = pd.get_dummies(col, prefix=name, drop_first=True)
        X_df = pd.concat([X_df, dummies], axis=1)
    X = X_df.astype(float).values
    return X, T, Y


def run_hillstrom_benchmark():
    print("\n" + "=" * 70)
    print(" HILLSTROM BENCHMARK: Email marketing (RCT)")
    print("=" * 70)

    X, T, Y = load_hillstrom()

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, email rate: {T.mean():.1%}")
    print(f"Mean spend: email={Y[T==1].mean():.4f}, no-email={Y[T==0].mean():.4f}")
    print(f"ATE (diff-in-means, unbiased due to RCT): {(Y[T==1].mean() - Y[T==0].mean()):.4f}\n")

    learner = ADAPEL(n_folds=3, fusion_gamma=1.0, min_alpha=0.1, clip_propensity=0.05)
    learner.fit(X, T, Y)
    cate = learner.predict(X)
    ate = cate.mean()
    ate_rct = Y[T==1].mean() - Y[T==0].mean()
    diag = learner.get_diagnostics(X)

    print(f"{'ADAPEL ATE':<25} {ate:.4f}")
    print(f"{'RCT ATE (unbiased)':<25} {ate_rct:.4f}")
    print(f"{'Difference':<25} {abs(ate - ate_rct):.4f}")
    print(f"{'Stacking weights':<25} {diag['meta_weights']}")
    print(f"{'DR-dominant | X-dominant':<25} {diag['pct_dr_dominant']:.1%} | {diag['pct_x_dominant']:.1%}")

    # CATE heterogeneity: compare top/bottom quartile
    high_impact = cate > np.percentile(cate, 75)
    low_impact = cate < np.percentile(cate, 25)
    print(f"\nCATE heterogeneity:")
    print(f"  Top 25% spend lift: {Y[T==1][high_impact[T==1]].mean() - Y[T==0][high_impact[T==0]].mean():.4f}")
    print(f"  Bottom 25% spend lift: {Y[T==1][low_impact[T==1]].mean() - Y[T==0][low_impact[T==0]].mean():.4f}")


# ── TWINS (Health, proxy ground truth CATE) ──────────────────────────────────

def load_twins():
    x = pd.read_csv("data/twins/X.csv", index_col=0)
    t = pd.read_csv("data/twins/T.csv", index_col=0)
    y = pd.read_csv("data/twins/Y.csv", index_col=0)

    # Standard filter: both twins <= 2000g (per literature)
    mask = (t["dbirwt_0"] <= 2000) & (t["dbirwt_1"] <= 2000)
    x, t, y = x[mask].reset_index(drop=True), t[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    records = []
    for i in range(len(x)):
        lighter = x.iloc[i].to_dict()
        lighter["treatment"] = 0
        lighter["outcome"] = y.iloc[i]["mort_0"]
        lighter["cate_truth"] = y.iloc[i]["mort_1"] - y.iloc[i]["mort_0"]
        records.append(lighter)

        heavier = x.iloc[i].to_dict()
        heavier["treatment"] = 1
        heavier["outcome"] = y.iloc[i]["mort_1"]
        heavier["cate_truth"] = y.iloc[i]["mort_1"] - y.iloc[i]["mort_0"]
        records.append(heavier)

    df = pd.DataFrame(records)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    drop_cols = [c for c in df.columns if df[c].isna().mean() > 0.5 or c.startswith("Unnamed")]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=["treatment", "outcome", "cate_truth"]).values
    T = df["treatment"].values
    Y = df["outcome"].values
    true_cate = df["cate_truth"].values
    return X, T, Y, true_cate


def run_twins_benchmark():
    print("\n" + "=" * 70)
    print(" TWINS BENCHMARK: Birth weight & mortality (proxy ground truth)")
    print("=" * 70)

    X, T, Y, true_cate = load_twins()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"Dataset: {X.shape[0]} children ({X.shape[0]//2} twin pairs), {X.shape[1]} features")
    print(f"Heavier twin rate: {T.mean():.1%}, mortality: {Y.mean():.1%}")
    print(f"True ATE: {true_cate.mean():.4f}\n")

    # Subsample for speed (if large)
    n_lim = min(5000, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=n_lim, replace=False)
    X, T, Y, true_cate = X[idx], T[idx], Y[idx], true_cate[idx]

    pehe_t = benchmark_t_learner(X, T, Y, true_cate)
    pehe_s = benchmark_s_learner(X, T, Y, true_cate)
    pehe_x = benchmark_x_learner(X, T, Y, true_cate)
    pehe_dr = benchmark_dr_learner(X, T, Y, true_cate)

    learner = ADAPEL(n_folds=3, fusion_gamma=1.0, min_alpha=0.1)
    learner.fit(X, T, Y)
    cate_adapel = learner.predict(X)
    pehe_fusion = np.sqrt(np.mean((cate_adapel - true_cate) ** 2))
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


# ── IHDP-100 Benchmark (chuẩn 100 realizations) ──────────────────────────────

def run_ihdp100_benchmark():
    print("\n" + "=" * 70)
    print(" IHDP-100 BENCHMARK: 100 realizations (train/test split)")
    print("=" * 70)

    train = np.load("data/ihdp/train.npz")
    test = np.load("data/ihdp/test.npz")
    n_reps = train["x"].shape[2]

    scores = {"T-Learner": [], "S-Learner": [], "ADAPEL": []}

    gbm_base = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)

    for rep in range(n_reps):
        X_tr = train["x"][:, :, rep]
        T_tr = train["t"][:, rep]
        Y_tr = train["yf"][:, rep]
        X_te = test["x"][:, :, rep]
        true_te = test["mu1"][:, rep] - test["mu0"][:, rep]

        # T-Learner
        m_t = clone(gbm_base).fit(np.column_stack([X_tr, T_tr]), Y_tr)
        pred_t = m_t.predict(np.column_stack([X_te, np.ones(len(X_te))])) - m_t.predict(np.column_stack([X_te, np.zeros(len(X_te))]))
        scores["T-Learner"].append(np.sqrt(np.mean((pred_t - true_te) ** 2)))

        # S-Learner
        m0_s = clone(gbm_base).fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
        pred_s = -m0_s.predict(X_te)
        scores["S-Learner"].append(np.sqrt(np.mean((pred_s - true_te) ** 2)))

        # ADAPEL
        learner = ADAPEL(n_folds=3, fusion_gamma=2.0, min_alpha=0.0)
        learner.fit(X_tr, T_tr, Y_tr)
        scores["ADAPEL"].append(np.sqrt(np.mean((learner.predict(X_te) - true_te) ** 2)))

        if (rep + 1) % 20 == 0:
            print(f"  Replication {rep+1}/{n_reps} done...")

    print(f"\n{'Method':<20} {'Mean PEHE':<12} {'Std PEHE':<12} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    for name, vals in scores.items():
        arr = np.array(vals)
        print(f"{name:<20} {arr.mean():<12.4f} {arr.std():<12.4f} {arr.min():<10.4f} {arr.max():<10.4f}")

    best = min((np.array(v).mean(), k) for k, v in scores.items())[1]
    print(f"\nBest mean PEHE: {best}")


# ── ACIC 2016 Benchmark ─────────────────────────────────────────────────────

def run_acic2016_benchmark():
    print("\n" + "=" * 70)
    print(" ACIC 2016 BENCHMARK: 10 settings (mean +/- std)")
    print("=" * 70)

    X_df = pd.read_csv("data/acic2016/x.csv")
    X_df = pd.get_dummies(X_df, drop_first=True)
    X_all = X_df.values.astype(float)
    feat_names = list(X_df.columns)

    scores = {"T-Learner": [], "S-Learner": [], "ADAPEL": []}
    gbm_base = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)

    for setting in range(1, 11):
        zymu = pd.read_csv(f"data/acic2016/zymu_{setting}.csv")
        T = zymu["z"].values
        Y = np.where(T == 1, zymu["y1"].values, zymu["y0"].values)
        true_cate = zymu["mu1"].values - zymu["mu0"].values

        rng = np.random.default_rng(setting)
        n = len(X_all)
        idx = rng.permutation(n)
        split = int(n * 0.8)
        tr, te = idx[:split], idx[split:]

        X_tr, T_tr, Y_tr = X_all[tr], T[tr], Y[tr]
        X_te, true_te = X_all[te], true_cate[te]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # T-Learner
        m_t = clone(gbm_base).fit(np.column_stack([X_tr_s, T_tr]), Y_tr)
        pred_t = m_t.predict(np.column_stack([X_te_s, np.ones(len(X_te_s))])) - m_t.predict(np.column_stack([X_te_s, np.zeros(len(X_te_s))]))
        scores["T-Learner"].append(np.sqrt(np.mean((pred_t - true_te) ** 2)))

        # S-Learner
        m0_s = clone(gbm_base).fit(X_tr_s[T_tr == 0], Y_tr[T_tr == 0])
        pred_s = -m0_s.predict(X_te_s)
        scores["S-Learner"].append(np.sqrt(np.mean((pred_s - true_te) ** 2)))

        # ADAPEL
        learner = ADAPEL(n_folds=3, fusion_gamma=1.0, min_alpha=0.1, clip_propensity=0.05)
        learner.fit(X_tr_s, T_tr, Y_tr)
        scores["ADAPEL"].append(np.sqrt(np.mean((learner.predict(X_te_s) - true_te) ** 2)))

        print(f"  Setting {setting:2d}/10 - T={scores['T-Learner'][-1]:.3f} S={scores['S-Learner'][-1]:.3f} AD={scores['ADAPEL'][-1]:.3f}")

    print(f"\n{'Method':<20} {'Mean PEHE':<12} {'Std PEHE':<12} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    for name, vals in scores.items():
        arr = np.array(vals)
        print(f"{name:<20} {arr.mean():<12.4f} {arr.std():<12.4f} {arr.min():<10.4f} {arr.max():<10.4f}")

    best = min((np.array(v).mean(), k) for k, v in scores.items())[1]
    print(f"\nBest mean PEHE: {best}")


if __name__ == "__main__":
    run_ihdp_benchmark()
    run_rhc_clinical()
    run_lalonde_benchmark()
    run_hillstrom_benchmark()
    run_twins_benchmark()
    run_ihdp100_benchmark()
    run_acic2016_benchmark()
