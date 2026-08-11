"""plots.py — Reusable figure generation for ADAPEL runs.

Use these helpers from any training script (synthetic, IHDP, RHC, or your own
data) to produce a consistent visual report. Each ``plot_*`` function writes a
PNG into ``out_dir``; ``plot_report`` renders all eight figures and merges
them into a single report image via ``combine_plots``.

Example:
    from plots import plot_report

    model.fit(X, T, Y)
    plot_report(model, X, T, true_cate, out_dir="plots", report_name="my_data")
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from model.core import alpha

try:
    import matplotlib

    matplotlib.use("Agg")  # headless backend, no display needed
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

LEARNER_NAMES = ["HistGBM", "ExtraTrees", "Ridge", "DecisionTree", "Lasso"]
REPORT_PLOTS = [
    "01_cate_calibration.png",
    "02_cate_distribution.png",
    "03_propensity_overlap.png",
    "04_alpha_curve.png",
    "05_stacking_weights.png",
    "06_bootstrap_ci.png",
    "07_feature_importance.png",
    "08_subgroup_analysis.png",
]


def r2(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _save_fig(out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [plot] {name}")


def _as_1d(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    return np.asarray(x).ravel()


# ── Individual figures ──


def plot_cate_calibration(pred, true, out_dir: str) -> None:
    """Predicted vs true CATE (identity line = perfect calibration)."""
    pred, true = np.asarray(pred).ravel(), np.asarray(true).ravel()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(true, pred, s=10, alpha=0.4, color="#1f77b4")
    lo, hi = min(pred.min(), true.min()), max(pred.max(), true.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="Ideal")
    ax.set_xlabel("True CATE")
    ax.set_ylabel("Predicted CATE")
    ax.set_title(f"CATE calibration  (R$^2$ = {r2(pred, true):.3f})")
    ax.legend()
    _save_fig(out_dir, "01_cate_calibration.png")


def plot_cate_distribution(pred, true=None, out_dir: str = "plots") -> None:
    """Histogram of predicted (and optional true) CATE."""
    pred = np.asarray(pred).ravel()
    true = _as_1d(true)
    all_vals = np.concatenate([pred] if true is None else [pred, true])
    bins = np.histogram_bin_edges(all_vals, bins=40)
    fig, ax = plt.subplots(figsize=(6, 4))
    if true is not None:
        ax.hist(true, bins=bins, alpha=0.5, label="True", color="#2ca02c")
    ax.hist(pred, bins=bins, alpha=0.5, label="Predicted", color="#1f77b4")
    ax.set_xlabel("CATE")
    ax.set_ylabel("Count")
    ax.set_title("CATE distribution")
    ax.legend()
    _save_fig(out_dir, "02_cate_distribution.png")


def plot_propensity(e, T, out_dir: str) -> None:
    """Propensity distribution by treatment arm (overlap check)."""
    e, T = np.asarray(e).ravel(), np.asarray(T).ravel()
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


def plot_alpha_curve(model, X, out_dir: str) -> None:
    """Theoretical alpha(e) curve + per-sample values from a fitted model."""
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


def plot_stacking_weights(meta_weights, out_dir: str,
                          names: Optional[list[str]] = None) -> None:
    """Bar chart of stacking weights per base learner."""
    meta_weights = np.asarray(meta_weights).ravel()
    names = names or LEARNER_NAMES[: len(meta_weights)]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ca02c" if w > 1e-8 else "#d62728" for w in meta_weights]
    ax.bar(names, meta_weights, color=colors)
    ax.set_ylabel("NNLS weight")
    ax.set_title("Stacking weights (green=active, red=pruned)")
    ax.tick_params(axis="x", rotation=20)
    for i, w in enumerate(meta_weights):
        ax.text(i, w + 0.01, f"{w:.2f}", ha="center", fontsize=9)
    _save_fig(out_dir, "05_stacking_weights.png")


def plot_bootstrap_ci(cate, lower, upper, true=None, in_overlap=None,
                      out_dir: str = "plots") -> None:
    """CATE per sample with bootstrap CI (subset for readability)."""
    cate, lower, upper = (np.asarray(x).ravel() for x in (cate, lower, upper))
    true = _as_1d(true)
    n = min(len(cate), 300)
    idx = np.argsort(cate)[:n]
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(x, cate[idx], yerr=np.vstack([
        cate[idx] - lower[idx], upper[idx] - cate[idx],
    ]), fmt="o", ms=3, elinewidth=0.7, color="#1f77b4", capsize=0)
    if true is not None:
        ax.scatter(x, true[idx], s=6, color="#2ca02c", label="True CATE", zorder=5)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Sample (sorted by predicted CATE)")
    ax.set_ylabel("CATE")
    ax.set_title("Bootstrap CI (green=coverage)")
    if true is not None:
        ax.legend()
    _save_fig(out_dir, "06_bootstrap_ci.png")


def plot_feature_importance(imp: dict, out_dir: str) -> None:
    """Permutation importance bar chart from ``variable_importance`` output."""
    names = imp["feature_names"]
    mean, std = imp["importances_mean"], imp["importances_std"]
    order = np.argsort(mean)

    fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(names))))
    ax.barh([names[i] for i in order], mean[order],
            xerr=std[order], color="#1f77b4", alpha=0.85)
    ax.set_xlabel("Permutation importance (MSE)")
    ax.set_title("CATE variable importance")
    _save_fig(out_dir, "07_feature_importance.png")


def plot_subgroup(subgroups: list[dict], out_dir: str) -> None:
    """Bar chart of mean CATE per subgroup (top-10)."""
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


# ── Combine / high-level report ──


def combine_plots(out_dir: str, names: list[str], output: str,
                  title: str = "ADAPEL Report") -> None:
    """Merge several PNGs into one grid image; delete the originals."""
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


def plot_report(model, X, T=None, true_cate=None, clin=None, d=None,
                out_dir: str = "plots", report_name: str = "report",
                feature_names: Optional[list[str]] = None, n_bins: int = 4,
                n_repeats: int = 5) -> None:
    """High-level: render the full 8-figure report + combined PNG.

    Parameters
    ----------
    model : fitted ADAPEL
    X : array-like, shape (n, p)
    T : array-like, optional — binary treatment, needed for propensity plot.
    true_cate : array-like, optional — ground-truth CATE, needed for
        calibration / coverage overlays. Omit for real data without ground truth.
    clin, d : dict, optional — pre-computed ``predict_clinical(X)`` /
        ``get_diagnostics(X)``. Computed internally if omitted.
    out_dir : output directory for PNGs.
    report_name : name of the combined report image (``{report_name}.png``).
    feature_names : optional feature labels for importance plot.
    n_bins, n_repeats : subgroup bin count / permutation importance repeats.
    """
    if not HAS_MATPLOTLIB:
        print("[plots] matplotlib not installed — skipping figures")
        return

    X = np.atleast_2d(np.asarray(X, dtype=float))
    T = _as_1d(T)
    true_cate = _as_1d(true_cate)
    d = d or model.get_diagnostics(X)
    clin = clin or model.predict_clinical(X)
    pred = model.predict(X)

    imp = model.variable_importance(
        X, feature_names, n_repeats=n_repeats, random_state=42,
    )
    sub = None
    if T is not None:
        sub = model.subgroup_analysis(X, T, pred, n_bins=n_bins)["subgroups"]

    if true_cate is not None:
        plot_cate_calibration(pred, true_cate, out_dir)
    plot_cate_distribution(pred, true_cate, out_dir)
    if T is not None:
        plot_propensity(d["propensity"], T, out_dir)
    plot_alpha_curve(model, X, out_dir)
    plot_stacking_weights(d["meta_weights"], out_dir)
    if clin.get("lower_ci") is not None:
        plot_bootstrap_ci(clin["cate"], clin["lower_ci"], clin["upper_ci"],
                          true_cate, clin.get("in_overlap"), out_dir)
    plot_feature_importance(imp, out_dir)
    if sub:
        plot_subgroup(sub, out_dir)

    combine_plots(out_dir, REPORT_PLOTS, output=f"{report_name}.png",
                  title=f"ADAPEL — {report_name}")