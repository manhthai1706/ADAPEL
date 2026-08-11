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

try:
    import matplotlib

    matplotlib.use("Agg")  # headless backend, không cần display
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False


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


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _print_results(rows: list[tuple[str, str, str]]) -> None:
    print(f"\n  {'Method':<25} {'PEHE':<10} {'Time':<10}")
    print(f"  {'-' * 45}")
    for name, metric, t in rows:
        print(f"  {name:<25} {metric:<10} {t:<10}")


# ── Plotting ──


def _save_fig(out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [plot] {name}")


def _combine_plots(out_dir: str, names: list[str], output: str) -> None:
    """Gộp các PNG thành 1 ảnh lưới (cols × rows), xóa các PNG gốc."""
    from PIL import Image, ImageDraw, ImageFont

    paths = [os.path.join(out_dir, n) for n in names]
    imgs = [Image.open(p).convert("RGB") for p in paths if os.path.exists(p)]
    if not imgs:
        print("    [combine] no images to merge")
        return

    cell_w, cell_h = 880, 580
    pad, title_h = 24, 56
    cols = 2

    def _fit(im: Image.Image) -> Image.Image:
        iw, ih = im.size
        scale = min(cell_w / iw, cell_h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        im2 = im.resize((nw, nh), Image.LANCZOS)
        canvas = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
        canvas.paste(im2, ((cell_w - nw) // 2, (cell_h - nh) // 2))
        return canvas

    fitted = [_fit(im) for im in imgs]
    rows = (len(fitted) + cols - 1) // cols
    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas_h = title_h + rows * cell_h + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    try:
        font_t = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font_t = ImageFont.load_default()
    title = "ADAPEL — Synthetic Benchmark Report"
    bbox = draw.textbbox((0, 0), title, font=font_t)
    tw = bbox[2] - bbox[0]
    draw.text(((canvas_w - tw) // 2, 14), title, fill="black", font=font_t)

    for i, im in enumerate(fitted):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = title_h + pad + r * (cell_h + pad)
        canvas.paste(im, (x, y))

    out_path = os.path.join(out_dir, output)
    canvas.save(out_path, "PNG", dpi=(150, 150))
    print(f"    [combine] {output} ({canvas_w}x{canvas_h})")

    for n in names:
        p = os.path.join(out_dir, n)
        if os.path.exists(p):
            os.remove(p)


def _plot_cate_calibration(pred, true, out_dir: str) -> None:
    """Predicted vs true CATE (identity line = perfect calibration)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(true, pred, s=10, alpha=0.4, color="#1f77b4")
    lo, hi = min(pred.min(), true.min()), max(pred.max(), true.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="Ideal")
    ax.set_xlabel("True CATE")
    ax.set_ylabel("Predicted CATE")
    ax.set_title(f"CATE calibration  (R$^2$ = {_r2(pred, true):.3f})")
    ax.legend()
    _save_fig(out_dir, "01_cate_calibration.png")


def _plot_cate_distribution(pred, true, out_dir: str) -> None:
    """Histogram so sánh phân phối CATE dự đoán vs thực tế."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.histogram_bin_edges(np.concatenate([pred, true]), bins=40)
    ax.hist(true, bins=bins, alpha=0.5, label="True", color="#2ca02c")
    ax.hist(pred, bins=bins, alpha=0.5, label="Predicted", color="#1f77b4")
    ax.set_xlabel("CATE")
    ax.set_ylabel("Count")
    ax.set_title("CATE distribution")
    ax.legend()
    _save_fig(out_dir, "02_cate_distribution.png")


def _plot_propensity(e, T, out_dir: str) -> None:
    """Phân phối propensity theo nhóm treatment (kiểm tra overlap)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(e[T == 0], bins=bins, alpha=0.5, label="Control", color="#ff7f0e")
    ax.hist(e[T == 1], bins=bins, alpha=0.5, label="Treated", color="#1f77b4")
    ax.axvspan(0, 0.05, color="gray", alpha=0.3)
    ax.axvspan(0.95, 1, color="gray", alpha=0.3)
    ax.set_xlabel("Propensity e(x)")
    ax.set_ylabel("Count")
    ax.set_title("Propensity overlap by treatment arm")
    ax.legend()
    _save_fig(out_dir, "03_propensity_overlap.png")


def _plot_alpha_curve(X, model, out_dir: str) -> None:
    """Đường cong alpha(e) lý thuyết + giá trị thực tế."""
    from model.core import alpha

    grid = np.linspace(0, 1, 200)
    curve = alpha(grid, model.fusion_gamma, model.min_alpha)

    d = model.get_diagnostics(X)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, curve, "k-", lw=2, label="alpha(e) curve")
    ax.scatter(d["propensity"], d["alpha"], s=8, alpha=0.3,
               color="#1f77b4", label="Samples")
    ax.axhline(model.min_alpha, color="r", ls="--", lw=1, label="min_alpha")
    ax.set_xlabel("Propensity e(x)")
    ax.set_ylabel("Fusion weight alpha")
    ax.set_title("X vs DR fusion weight (alpha)")
    ax.legend()
    _save_fig(out_dir, "04_alpha_curve.png")


def _plot_stacking_weights(meta_weights, out_dir: str) -> None:
    """Biểu đồ cột cho stacking weights của từng base learner."""
    names = ["HistGBM", "ExtraTrees", "Ridge", "DecisionTree", "Lasso"]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ca02c" if w > 1e-8 else "#d62728" for w in meta_weights]
    ax.bar(names, meta_weights, color=colors)
    ax.set_ylabel("NNLS weight")
    ax.set_title("Stacking weights (green=active, red=pruned)")
    ax.tick_params(axis="x", rotation=20)
    for i, w in enumerate(meta_weights):
        ax.text(i, w + 0.01, f"{w:.2f}", ha="center", fontsize=9)
    _save_fig(out_dir, "05_stacking_weights.png")


def _plot_bootstrap_ci(cate, lower, upper, true, in_overlap, out_dir: str) -> None:
    """CATE từng mẫu kèm khoảng tin cậy bootstrap (mẫu con để dễ đọc)."""
    n = min(len(cate), 300)
    idx = np.argsort(cate)[:n]
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(x, cate[idx], yerr=np.vstack([
        cate[idx] - lower[idx], upper[idx] - cate[idx],
    ]), fmt="o", ms=3, elinewidth=0.7, color="#1f77b4", capsize=0)
    ax.scatter(x, true[idx], s=6, color="#2ca02c", label="True CATE", zorder=5)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Sample (sorted by predicted CATE)")
    ax.set_ylabel("CATE")
    ax.set_title("Bootstrap CI (green=coverage)")
    ax.legend()
    _save_fig(out_dir, "06_bootstrap_ci.png")


def _plot_feature_importance(imp, out_dir: str) -> None:
    """Permutation importance bar chart."""
    names = imp["feature_names"]
    mean, std = imp["importances_mean"], imp["importances_std"]
    order = np.argsort(mean)

    fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(names))))
    ax.barh([names[i] for i in order], mean[order],
            xerr=std[order], color="#1f77b4", alpha=0.85)
    ax.set_xlabel("Permutation importance (MSE)")
    ax.set_title("CATE variable importance")
    _save_fig(out_dir, "07_feature_importance.png")


def _plot_subgroup(subgroups, out_dir: str) -> None:
    """Bar chart CATE trung bình theo subgroup."""
    top = subgroups[:10]
    labels = [s["name"] for s in top]
    means = [s["cate_mean"] for s in top]
    overall = float(np.mean([s["cate_mean"] for s in subgroups]))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(top))))
    colors = ["#2ca02c" if m >= 0 else "#d62728" for m in means]
    ax.barh(labels, means, color=colors, alpha=0.85)
    ax.axvline(overall, color="k", ls="--", lw=1, label=f"Overall {overall:.3f}")
    ax.set_xlabel("CATE mean")
    ax.set_title("Subgroup analysis (top-10)")
    ax.legend()
    _save_fig(out_dir, "08_subgroup_analysis.png")


def save_synthetic_figures(model, X_te, T_te, true_cate, clin, d, out_dir: str) -> None:
    """Tạo toàn bộ biểu đồ cho benchmark synthetic."""
    pred = model.predict(X_te)
    e = d["propensity"]
    imp = model.variable_importance(X_te, n_repeats=5, random_state=42)
    sub = model.subgroup_analysis(X_te, T_te, true_cate, n_bins=4)["subgroups"]

    _plot_cate_calibration(pred, true_cate, out_dir)
    _plot_cate_distribution(pred, true_cate, out_dir)
    _plot_propensity(e, T_te, out_dir)
    _plot_alpha_curve(X_te, model, out_dir)
    _plot_stacking_weights(d["meta_weights"], out_dir)
    _plot_bootstrap_ci(clin["cate"], clin["lower_ci"], clin["upper_ci"],
                       true_cate, clin["in_overlap"], out_dir)
    _plot_feature_importance(imp, out_dir)
    _plot_subgroup(sub, out_dir)

    _combine_plots(out_dir, [
        "01_cate_calibration.png",
        "02_cate_distribution.png",
        "03_propensity_overlap.png",
        "04_alpha_curve.png",
        "05_stacking_weights.png",
        "06_bootstrap_ci.png",
        "07_feature_importance.png",
        "08_subgroup_analysis.png",
    ], output="synthetic.png")


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
        save_synthetic_figures(model, X[te], T[te], true_cate[te], clin, d, args.outdir)

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
        _plot_propensity(d["propensity"], T, args.outdir)
        _plot_stacking_weights(d["meta_weights"], args.outdir)
        _plot_bootstrap_ci(clin["cate"], clin["lower_ci"], clin["upper_ci"],
                           np.zeros_like(clin["cate"]), clin["in_overlap"], args.outdir)


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
        _plot_propensity(d["propensity"], T, args.outdir)
        _plot_stacking_weights(d["meta_weights"], args.outdir)


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
