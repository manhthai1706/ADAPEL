"""hillstrom.py — Train ADAPEL on the Hillstrom email marketing dataset.

Hillstrom MineThatData challenge: 64,000 customers, RCT with three arms
(no email / mens email / womens email). Outcome = spend (dollars). Data is
downloaded on first run from a public GitHub mirror and cached under
``data/hillstrom/``. ATE is validated against the RCT estimate.

Usage:
    python hillstrom.py -m fast              # train + report + plots
    python hillstrom.py -m fast --no-plots   # train only
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
import pandas as pd

from model import ADAPEL
from plots import HAS_MATPLOTLIB, plot_report

HILLSTROM_URL = (
    "https://raw.githubusercontent.com/estimand/"
    "intro-to-python-for-data-science/master/hillstrom.csv"
)
CACHE = "data/hillstrom/hillstrom.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADAPEL on Hillstrom dataset")
    parser.add_argument("-m", "--mode", choices=["fast", "balanced", "accurate"],
                        default="fast")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--outdir", default="plots")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--arm", choices=["mens", "womens", "any"],
                        default="mens",
                        help="Treatment arm vs no-email control (def: mens)")
    return parser.parse_args()


def _download() -> str:
    if os.path.exists(CACHE):
        print(f"  Data cached: {CACHE}")
        return CACHE
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    print(f"  Downloading Hillstrom -> {CACHE} ...")
    urllib.request.urlretrieve(HILLSTROM_URL, CACHE)
    print("  Done.")
    return CACHE


def load_hillstrom(path: str, arm: str = "mens"):
    """Return (X, T, Y, feature_names) for the requested arm vs control."""
    df = pd.read_csv(path)
    if arm == "any":
        T = np.where(df["segment"] != "No E-Mail", 1, 0).astype(float)
    else:
        df = df[df["segment"].isin(["No E-Mail", f"{arm.capitalize()} E-Mail"])]
        T = np.where(df["segment"] == f"{arm.capitalize()} E-Mail", 1, 0).astype(float)

    X_df = pd.DataFrame({
        "recency": pd.to_numeric(df["recency"], errors="coerce"),
        "history": pd.to_numeric(df["history"], errors="coerce"),
        "mens": pd.to_numeric(df["mens"], errors="coerce"),
        "womens": pd.to_numeric(df["womens"], errors="coerce"),
        "newbie": pd.to_numeric(df["newbie"], errors="coerce"),
    })
    for name in ("zip_code", "channel"):
        X_df = pd.concat(
            [X_df, pd.get_dummies(df[name], prefix=name, drop_first=True)], axis=1
        )
    X = X_df.astype(float).values
    Y = pd.to_numeric(df["spend"], errors="coerce").values
    feature_names = list(X_df.columns)
    return X, T, Y, feature_names


def main() -> None:
    args = _parse_args()
    np.set_printoptions(precision=3, suppress=True)
    t_start = time.time()

    print("=" * 65)
    print("  HILLSTROM — RCT benchmark (email marketing, spend outcome)")
    print("=" * 65)

    path = _download()
    X, T, Y, feature_names = load_hillstrom(path, args.arm)

    ate_rct = Y[T == 1].mean() - Y[T == 0].mean()
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, "
          f"Email rate: {T.mean():.1%}")
    print(f"  Arm: {args.arm} vs No E-Mail")
    print(f"  RCT ATE: {ate_rct:.4f}")

    t0 = time.time()
    model = ADAPEL(n_folds=3, mode=args.mode, verbose=args.verbose).fit(X, T, Y)
    ate = model.estimate_ate(X)
    t_a = time.time() - t0
    d = model.get_diagnostics(X)

    print(f"\n  {'ADAPEL ATE':<25} {ate:.4f} (RCT: {ate_rct:.4f}, "
          f"diff: {abs(ate - ate_rct):.4f})")
    print(f"  {'Stacking weights':<25} {np.round(d['meta_weights'], 3)} "
          f"(active: {(d['meta_weights'] > 1e-8).sum()}/5)")
    print(f"  {'DR-dominant':<25} {d['pct_dr_dominant']:.1%}")
    print(f"  {'Fit time':<25} {t_a:.2f}s")

    if HAS_MATPLOTLIB and not args.no_plots:
        print(f"\n  Generating plots -> {args.outdir}/")
        plot_report(model, X, T, clin=None, d=d, out_dir=args.outdir,
                    report_name=f"hillstrom_{args.arm}", feature_names=feature_names,
                    n_bins=4)

    print("\n" + "=" * 65)
    print(f"  DONE — total: {time.time() - t_start:.2f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()