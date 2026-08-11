"""train.py — ADAPEL benchmark runner + visual report generator.

Chạy synthetic benchmark (luôn có sẵn) và các benchmark thật (IHDP / RHC /
Hillstrom) nếu file dữ liệu tồn tại. Xuất các biểu đồ báo cáo ra thư mục
``plots/`` bằng matplotlib.

Usage:
    python train.py -m fast              # benchmark + plots
    python train.py -m fast --no-plots   # chỉ chạy, không xuất hình
    python train.py -m fast -o report    # plots vào thư mục report/
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor

from model import ADAPEL
from plots import HAS_MATPLOTLIB, plot_report


T_LEARNER_GBM = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
)

# ── CLI ──


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADAPEL benchmarks + plots")
    parser.add_argument("-m", "--mode", choices=["fast", "balanced", "accurate"],
                        default="fast", help="Model complexity mode (def: fast)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--outdir", default="plots",
                        help="Output directory for figures (def: plots)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip figure generation")
    return parser.parse_args()


# ── Helpers ──


def _fmt_time(s: float) -> str:
    return f"{s:.2f}s" if s < 60 else f"{int(s // 60)}m {s % 60:.0f}s"


def _resolve_data(path: str, alt: str) -> str | None:
    for candidate in (path, alt):
        if os.path.exists(candidate):
            return candidate
    return None


def _run_t_learner(X_tr, T_tr, Y_tr, X_te, true_cate):
    model = clone(T_LEARNER_GBM).fit(np.column_stack([X_tr, T_tr]), Y_tr)
    pred = (model.predict(np.column_stack([X_te, np.ones(len(X_te))]))
            - model.predict(np.column_stack([X_te, np.zeros(len(X_te))])))
    return np.sqrt(np.mean((pred - true_cate) ** 2))


def _pehe(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def _print_results(rows: list[tuple[str, str, str]]) -> None:
    print(f"\n  {'Method':<25} {'PEHE':<10} {'Time':<10}")
    print(f"  {'-' * 45}")
    for name, metric, t in rows:
        print(f"  {name:<25} {metric:<10} {t:<10}")


# ── 1. Synthetic data ──


def test_synthetic(args) -> ADAPEL:
    print("=" * 65)
    print("  1. SYNTHETIC DATA — realistic confounding + non-linear CATE")
    print("=" * 65)

    rng = np.random.default_rng(42)
    n, p = 2000, 20
    X = np.zeros((n, p))
    X[:, :10] = rng.standard_normal((n, 10))
    X[:, 10:15] = rng.binomial(1, 0.4, (n, 5))
    X[:, 15:] = rng.standard_normal((n, 5))

    logit = -1.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.5 * X[:, 10] - 0.3 * X[:, 11]
    T = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    true_cate = 0.8 * X[:, 0] + 0.5 * np.sin(X[:, 2]) - 0.3 * X[:, 1] * X[:, 3] + 0.4 * X[:, 10]
    mu0 = (0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2] ** 2 + 0.1 * X[:, 10]
           + 0.05 * X[:, 11] * X[:, 3] - 0.1 * np.abs(X[:, 4]))
    Y = np.where(T == 1, mu0 + true_cate, mu0) + rng.normal(0, 0.3 * mu0.std(), n)

    idx = rng.permutation(n)
    tr, te = idx[:1500], idx[1500:]

    print(f"  Train: {len(tr)} samples, T-rate: {T[tr].mean():.1%}")
    print(f"  Test:  {len(te)} samples, T-rate: {T[te].mean():.1%}")
    print(f"  True ATE: {true_cate[te].mean():.4f} | Naive RD: {Y[T == 1].mean() - Y[T == 0].mean():.4f}")

    t0 = time.time()
    pehe_t = _run_t_learner(X[tr], T[tr], Y[tr], X[te], true_cate[te])
    t_t = _fmt_time(time.time() - t0)

    t0 = time.time()
    model = ADAPEL(n_folds=5, mode=args.mode, verbose=args.verbose).fit(X[tr], T[tr], Y[tr])
    pehe_a = _pehe(model.predict(X[te]), true_cate[te])
    t_a = _fmt_time(time.time() - t0)
    d = model.get_diagnostics(X[te])

    t0 = time.time()
    model_fs = ADAPEL(
        n_folds=5, feature_select=True, mode=args.mode, verbose=args.verbose,
    ).fit(X[tr], T[tr], Y[tr])
    pehe_fs = _pehe(model_fs.predict(X[te]), true_cate[te])
    t_fs = _fmt_time(time.time() - t0)

    model.fit_bootstrap(X[tr], T[tr], Y[tr], n_bootstrap=30)
    clin = model.predict_clinical(X[te])
    cov = float(np.mean((clin["lower_ci"] <= true_cate[te]) & (true_cate[te] <= clin["upper_ci"])))
    ate = model.estimate_ate(X[te])

    _print_results([
        ("T-Learner (GBM)", f"{pehe_t:.4f}", t_t),
        ("ADAPEL", f"{pehe_a:.4f}", t_a),
        ("ADAPEL + feat select", f"{pehe_fs:.4f}", t_fs),
    ])
    print(f"  {'ADAPEL ATE':<25} {ate:.4f} (true: {true_cate[te].mean():.4f})")
    print(f"  {'Stacking weights':<25} {np.round(d['meta_weights'], 3)} "
          f"(active: {(d['meta_weights'] > 1e-8).sum()}/5)")
    print(f"  {'DR-dominant':<25} {d['pct_dr_dominant']:.1%}")
    print(f"  {'Boot CI coverage':<25} {cov:.1%}")

    if HAS_MATPLOTLIB and not args.no_plots:
        print(f"\n  Generating plots -> {args.outdir}/")
        plot_report(model, X[te], T[te], true_cate[te],
                    clin=clin, d=d, out_dir=args.outdir,
                    report_name="synthetic")

    return model


# ── 2. IHDP ──


def test_ihdp(args) -> None:
    path = _resolve_data("data/ihdp/ihdp.csv", "paper/ihdp/ihdp.csv")
    if path is None:
        print("\n  [skip] IHDP — chưa có file data (data/ihdp/ihdp.csv).")
        return
    print("\n" + "=" * 65)
    print("  2. IHDP — semi-synthetic (747 samples, 25 features)")
    print("=" * 65)

    data = np.loadtxt(path, delimiter=",")
    T, Y, mu0, mu1 = data[:, 0], data[:, 1], data[:, 3], data[:, 4]
    X = data[:, 5:]
    true_cate = mu1 - mu0

    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, T-rate: {T.mean():.1%}")
    print(f"  True ATE: {true_cate.mean():.4f}")

    pehe_t = _run_t_learner(X, T, Y, X, true_cate)

    t0 = time.time()
    model = ADAPEL(
        n_folds=5, fusion_gamma=2.0, min_alpha=0.0,
        mode=args.mode, verbose=args.verbose,
    ).fit(X, T, Y)
    pehe_a = _pehe(model.predict(X), true_cate)
    t_a = _fmt_time(time.time() - t0)
    d = model.get_diagnostics(X)

    _print_results([("T-Learner (GBM)", f"{pehe_t:.4f}", "--"), ("ADAPEL", f"{pehe_a:.4f}", t_a)])
    print(f"  {'Weights':<25} {np.round(d['meta_weights'], 3)} "
          f"(active: {(d['meta_weights'] > 1e-8).sum()}/5)")
    print(f"  {'DR-dominant':<25} {d['pct_dr_dominant']:.1%}")


# ── 3. RHC ──


def test_rhc(args) -> None:
    path = _resolve_data("data/rhc/rhc.csv", "paper/rhc/rhc.csv")
    if path is None:
        print("\n  [skip] RHC — chưa có file data (data/rhc/rhc.csv).")
        return
    print("\n" + "=" * 65)
    print("  3. RHC — real observational (5735 patients, 39 features)")
    print("=" * 65)

    df = pd.read_csv(path)
    T = (df["swang1"] == "RHC").astype(int).values
    Y = (df["death"] == "Yes").astype(int).values
    cov_cols = ["age", "sex", "race", "edu", "income", "ninsclas", "cat1", "das2d3pc",
                "dnr1", "ca", "surv2md1", "aps1", "scoma1", "meanbp1", "wblc1", "hrt1",
                "resp1", "temp1", "pafi1", "alb1", "hema1", "bili1", "crea1", "sod1",
                "pot1", "paco21", "ph1", "cardiohx", "chfhx", "dementhx", "psychhx",
                "chrpulhx", "renalhx", "liverhx", "gibledhx", "malighx", "immunhx",
                "transhx", "amihx"]
    X_df = df[cov_cols].copy()
    for col in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = X_df[col].fillna(X_df[col].median())
        else:
            X_df[col] = X_df[col].fillna(X_df[col].mode()[0])
    X = pd.get_dummies(X_df, drop_first=True).astype(float).values

    print(f"  Patients: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  RHC rate: {T.mean():.1%}, Mortality: {Y.mean():.1%}")
    print(f"  Naive RD: {Y[T == 1].mean() - Y[T == 0].mean():.4f} (confounded)")

    t0 = time.time()
    model = ADAPEL(n_folds=3, mode=args.mode, verbose=args.verbose).fit(X, T, Y)
    ate = model.estimate_ate(X)
    t_a = _fmt_time(time.time() - t0)
    e_val = model.estimate_e_value(X, "binary")
    d = model.get_diagnostics(X)

    model.fit_bootstrap(X, T, Y, n_bootstrap=15)
    clin = model.predict_clinical(X)
    ate_boot = float(clin["cate"].mean())

    print(f"\n  {'ADAPEL ATE (point)':<25} {ate:.4f}")
    print(f"  {'ADAPEL ATE (BMA)':<25} {ate_boot:.4f}")
    print(f"  {'E-Value':<25} {e_val:.4f}")
    print(f"  {'Weights':<25} {np.round(d['meta_weights'], 3)}")
    print(f"  {'Fit time':<25} {t_a}")
    print(f"  {'Interpretation: RHC':<25} "
          f"{'INCREASES' if ate > 0 else 'DECREASES'} mortality by {abs(ate) * 100:.2f}%")

    if HAS_MATPLOTLIB and not args.no_plots:
        print(f"\n  Generating plots -> {args.outdir}/")
        plot_report(model, X, T, clin=clin, d=d, out_dir=args.outdir,
                    report_name="rhc", n_bins=4)


# ── 4. Hillstrom ──


def test_hillstrom(args) -> None:
    path = _resolve_data("data/hillstrom/hillstrom.csv", "paper/hillstrom/hillstrom.csv")
    if path is None:
        print("\n  [skip] Hillstrom — chưa có file data (data/hillstrom/hillstrom.csv).")
        return
    print("\n" + "=" * 65)
    print("  4. HILLSTROM — RCT benchmark (64000 customers)")
    print("=" * 65)

    df = pd.read_csv(path)
    T = np.where(df["segment"] != "No E-Mail", 1, 0)
    X_df = df[["recency", "history", "mens", "womens", "newbie"]].copy()
    for name in ("zip_code", "channel"):
        X_df = pd.concat([X_df, pd.get_dummies(df[name], prefix=name, drop_first=True)], axis=1)
    X = X_df.astype(float).values
    Y = df["spend"].values

    ate_rct = Y[T == 1].mean() - Y[T == 0].mean()

    t0 = time.time()
    model = ADAPEL(n_folds=3, mode=args.mode, verbose=args.verbose).fit(X, T, Y)
    ate = model.estimate_ate(X)
    t_a = _fmt_time(time.time() - t0)
    d = model.get_diagnostics(X)

    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, Email rate: {T.mean():.1%}")
    print(f"  RCT ATE: {ate_rct:.4f}")
    print(f"  ADAPEL ATE: {ate:.4f} (diff: {abs(ate - ate_rct):.4f})")
    print(f"  Weights: {np.round(d['meta_weights'], 3)}")
    print(f"  Time: {t_a}")

    if HAS_MATPLOTLIB and not args.no_plots:
        print(f"\n  Generating plots -> {args.outdir}/")
        plot_report(model, X, T, clin=clin, d=d, out_dir=args.outdir,
                    report_name="hillstrom", n_bins=4)


if __name__ == "__main__":
    args = _parse_args()
    np.set_printoptions(precision=3, suppress=True)

    if HAS_MATPLOTLIB and not args.no_plots:
        os.makedirs(args.outdir, exist_ok=True)
    elif not HAS_MATPLOTLIB and not args.no_plots:
        print("[warn] matplotlib chưa cài — bỏ qua xuất hình. Cài bằng: pip install matplotlib")

    t_start = time.time()
    print(f"ADAPEL mode: {args.mode}")

    test_synthetic(args)
    test_ihdp(args)
    test_rhc(args)
    test_hillstrom(args)

    print("\n" + "=" * 65)
    print(f"  DONE — total: {_fmt_time(time.time() - t_start)}")
    print("=" * 65)
