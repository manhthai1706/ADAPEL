"""ihdp.py — Train ADAPEL on the IHDP benchmark (download from GitHub).

IHDP (Infant Health and Development Program): 747 samples, 25 covariates,
semi-synthetic outcomes with known ground-truth CATE (mu1 - mu0). Data is
downloaded on first run from the AMLab-Amsterdam CEVAE GitHub mirror and
cached under ``data/ihdp/``.

Usage:
    python ihdp.py -m fast              # train + report + plots
    python ihdp.py -m fast --no-plots   # train only
    python ihdp.py -m accurate -o report
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor

from model import ADAPEL
from plots import HAS_MATPLOTLIB, plot_report

IHDP_URL = (
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/"
    "master/datasets/IHDP/csv/ihdp_npci_1.csv"
)
CACHE = "data/ihdp/ihdp.csv"

T_LEARNER_GBM = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADAPEL on IHDP benchmark")
    parser.add_argument("-m", "--mode", choices=["fast", "balanced", "accurate"],
                        default="fast")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--outdir", default="plots")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _download() -> str:
    if os.path.exists(CACHE):
        print(f"  Data cached: {CACHE}")
        return CACHE
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    print(f"  Downloading IHDP -> {CACHE} ...")
    urllib.request.urlretrieve(IHDP_URL, CACHE)
    print("  Done.")
    return CACHE


def load_ihdp(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, T, Y, true_cate) from the CEVAE ihdp_npci_1.csv format."""
    data = np.loadtxt(path, delimiter=",")
    T, Y, mu0, mu1 = data[:, 0], data[:, 1], data[:, 3], data[:, 4]
    X = data[:, 5:]
    return X, T, Y, mu1 - mu0


def _run_t_learner(X, T, Y, true_cate) -> float:
    model = clone(T_LEARNER_GBM).fit(np.column_stack([X, T]), Y)
    pred = (model.predict(np.column_stack([X, np.ones(len(X))]))
            - model.predict(np.column_stack([X, np.zeros(len(X))])))
    return float(np.sqrt(np.mean((pred - true_cate) ** 2)))


def _pehe(pred, true) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def main() -> None:
    args = _parse_args()
    np.set_printoptions(precision=3, suppress=True)
    t_start = time.time()

    print("=" * 65)
    print("  IHDP — semi-synthetic benchmark (747 samples, 25 features)")
    print("=" * 65)

    path = _download()
    X, T, Y, true_cate = load_ihdp(path)
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, T-rate: {T.mean():.1%}")
    print(f"  True ATE: {true_cate.mean():.4f}")

    pehe_t = _run_t_learner(X, T, Y, true_cate)

    t0 = time.time()
    model = ADAPEL(
        n_folds=5, fusion_gamma=2.0, min_alpha=0.0,
        mode=args.mode, verbose=args.verbose,
    ).fit(X, T, Y)
    pehe_a = _pehe(model.predict(X), true_cate)
    t_a = time.time() - t0
    d = model.get_diagnostics(X)
    ate = model.estimate_ate(X)

    print(f"\n  {'Method':<25} {'PEHE':<10} {'Time':<10}")
    print(f"  {'-' * 45}")
    print(f"  {'T-Learner (GBM)':<25} {pehe_t:.4f} {'--':<10}")
    print(f"  {'ADAPEL':<25} {pehe_a:.4f} {t_a:.2f}s")
    print(f"  {'ADAPEL ATE':<25} {ate:.4f} (true: {true_cate.mean():.4f})")
    print(f"  {'Stacking weights':<25} {np.round(d['meta_weights'], 3)} "
          f"(active: {(d['meta_weights'] > 1e-8).sum()}/5)")
    print(f"  {'DR-dominant':<25} {d['pct_dr_dominant']:.1%}")

    model.fit_bootstrap(X, T, Y, n_bootstrap=20)
    clin = model.predict_clinical(X)
    cov = float(np.mean((clin["lower_ci"] <= true_cate) & (true_cate <= clin["upper_ci"])))
    print(f"  {'Boot CI coverage':<25} {cov:.1%}")

    if HAS_MATPLOTLIB and not args.no_plots:
        print(f"\n  Generating plots -> {args.outdir}/")
        plot_report(model, X, T, true_cate, clin=clin, d=d,
                    out_dir=args.outdir, report_name="ihdp")

    print("\n" + "=" * 65)
    print(f"  DONE — total: {time.time() - t_start:.2f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()