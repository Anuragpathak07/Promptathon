"""
===============================================================
Step 5 — Evaluation: AUROC & F1-score on the MVTec test set
---------------------------------------------------------------
Produces per-category and aggregate metrics, plus:
  • ROC curves
  • Precision-Recall curves
  • Score distribution plots
  • Saved to outputs/results/<category>/
===============================================================
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    confusion_matrix,
)
import pandas as pd
import yaml
from tqdm import tqdm

from model import build_patchcore, MVTecDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

RESULTS_DIR = Path(CFG["evaluation"]["output_dir"])


# ================================================================== #
#  Per-category evaluation                                             #
# ================================================================== #
def evaluate_category(category: str) -> Dict:
    """
    Run PatchCore inference on the test split of *category*,
    compute AUROC and best-threshold F1, save plots.

    Returns dict with metric values.
    """
    log.info(f"━━━ Evaluating: {category} ━━━")

    # Load model
    pc = build_patchcore(category)
    pc.load()

    # Test dataset
    test_ds = MVTecDataset(
        category, split="test",
        image_size=CFG["dataset"]["image_size"],
        data_dir=CFG["dataset"]["data_dir"],
    )
    loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=0,   # 0 = required on Windows (avoids spawn deadlocks)
        shuffle=False,
    )

    scores: List[float] = []
    labels: List[int]   = []

    for img, label, _ in tqdm(loader, desc=f"  Scoring {category}"):
        score, _ = pc.predict_image(img)
        scores.append(score)
        labels.append(int(label))

    scores = np.array(scores)
    labels = np.array(labels)

    # ── AUROC ──────────────────────────────────────────────────────
    auroc = roc_auc_score(labels, scores)

    # ── Best F1 (sweep thresholds) ─────────────────────────────────
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = (
        2 * precisions * recalls / (precisions + recalls + 1e-8)
    )
    best_idx  = np.argmax(f1_scores)
    best_f1   = float(f1_scores[best_idx])
    best_thr  = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(pc.threshold)

    avg_prec  = average_precision_score(labels, scores)

    # ── Predictions at best threshold ─────────────────────────────
    preds = (scores >= best_thr).astype(int)
    cm    = confusion_matrix(labels, preds)

    metrics = {
        "category":  category,
        "auroc":     round(auroc,    4),
        "f1":        round(best_f1,  4),
        "avg_prec":  round(avg_prec, 4),
        "threshold": round(best_thr, 4),
        "n_normal":  int((labels == 0).sum()),
        "n_anomaly": int((labels == 1).sum()),
    }
    log.info(
        f"  AUROC={auroc:.4f}  F1={best_f1:.4f}  AvgPrec={avg_prec:.4f}"
    )

    # ── Plots ──────────────────────────────────────────────────────
    out_dir = RESULTS_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_roc(labels, scores, auroc, category, out_dir)
    _plot_pr(precisions, recalls, avg_prec, category, out_dir)
    _plot_score_distribution(scores, labels, best_thr, category, out_dir)

    return metrics


# ================================================================== #
#  Plot helpers                                                        #
# ================================================================== #
STYLE = {
    "figure.facecolor":  "#0f0f1a",
    "axes.facecolor":    "#16172b",
    "axes.edgecolor":    "#3a3b5c",
    "axes.labelcolor":   "#c8c9e0",
    "xtick.color":       "#c8c9e0",
    "ytick.color":       "#c8c9e0",
    "text.color":        "#e8e9f0",
    "grid.color":        "#2a2b4c",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
}


def _apply_style(ax: plt.Axes) -> None:
    with plt.rc_context(STYLE):
        pass
    ax.set_facecolor("#16172b")
    ax.tick_params(colors="#c8c9e0")
    ax.yaxis.label.set_color("#c8c9e0")
    ax.xaxis.label.set_color("#c8c9e0")
    ax.title.set_color("#e8e9f0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3b5c")
    ax.grid(True, color="#2a2b4c", linestyle="--", alpha=0.5)


def _plot_roc(labels, scores, auroc, category, out_dir):
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f0f1a")
    _apply_style(ax)
    ax.plot(fpr, tpr, color="#7c6af7", lw=2, label=f"AUC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], ":", color="#555580", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color="#7c6af7")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {category}")
    ax.legend(loc="lower right", facecolor="#16172b", edgecolor="#3a3b5c",
              labelcolor="#e8e9f0")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pr(precisions, recalls, avg_prec, category, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f0f1a")
    _apply_style(ax)
    ax.plot(recalls, precisions, color="#f7a96a", lw=2,
            label=f"AP = {avg_prec:.4f}")
    ax.fill_between(recalls, precisions, alpha=0.15, color="#f7a96a")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {category}")
    ax.legend(loc="upper right", facecolor="#16172b", edgecolor="#3a3b5c",
              labelcolor="#e8e9f0")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_score_distribution(scores, labels, threshold, category, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0f0f1a")
    _apply_style(ax)

    normal_scores  = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    bins = np.linspace(scores.min(), scores.max(), 40)
    ax.hist(normal_scores,  bins=bins, alpha=0.7, color="#4ecdc4",
            label="Normal", edgecolor="#0f0f1a", lw=0.4)
    ax.hist(anomaly_scores, bins=bins, alpha=0.7, color="#ff6b6b",
            label="Anomaly", edgecolor="#0f0f1a", lw=0.4)
    ax.axvline(threshold, color="#ffd93d", lw=2, ls="--",
               label=f"Threshold = {threshold:.3f}")

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution — {category}")
    ax.legend(facecolor="#16172b", edgecolor="#3a3b5c", labelcolor="#e8e9f0")
    fig.tight_layout()
    fig.savefig(out_dir / "score_distribution.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ================================================================== #
#  Main                                                                #
# ================================================================== #
def main():
    parser = argparse.ArgumentParser(description="Evaluate PatchCore models")
    parser.add_argument(
        "--categories", nargs="+",
        default=CFG["dataset"]["categories"],
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip evaluation and directly generate markdown report from metrics_summary.csv"
    )
    args = parser.parse_args()

    if args.report_only:
        csv_path = RESULTS_DIR / "metrics_summary.csv"
        if not csv_path.exists():
            log.error(f"Cannot generate report: {csv_path} does not exist. Run a full evaluation first.")
            return
        log.info(f"Loading existing metrics from {csv_path}...")
        df = pd.read_csv(csv_path).set_index("category")
        _generate_markdown_report(df)
        return

    all_metrics: List[Dict] = []
    for cat in args.categories:
        try:
            m = evaluate_category(cat)
            all_metrics.append(m)
        except Exception as e:
            log.error(f"Failed to evaluate {cat}: {e}")

    if not all_metrics:
        log.error("No metrics collected.")
        return

    # ── Summary table ──────────────────────────────────────────────
    df = pd.DataFrame(all_metrics).set_index("category")
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info("\n" + df.to_string())

    mean_auroc = df["auroc"].mean()
    mean_f1    = df["f1"].mean()
    log.info(f"\nMean AUROC : {mean_auroc:.4f}")
    log.info(f"Mean F1    : {mean_f1:.4f}")

    # Save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "metrics_summary.csv"
    df.to_csv(csv_path)
    log.info(f"\nResults saved to {csv_path}")

    # ── Aggregate bar chart ────────────────────────────────────────
    _plot_aggregate(df)

    # ── Markdown report ───────────────────────────────────────────
    _generate_markdown_report(df)


def _plot_aggregate(df: pd.DataFrame) -> None:
    cats   = df.index.tolist()
    aurocs = df["auroc"].tolist()
    f1s    = df["f1"].tolist()

    x  = np.arange(len(cats))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(cats) * 2), 5))
    fig.patch.set_facecolor("#0f0f1a")
    _apply_style(ax)

    b1 = ax.bar(x - w / 2, aurocs, w, label="AUROC", color="#7c6af7",
                alpha=0.85, edgecolor="#0f0f1a")
    b2 = ax.bar(x + w / 2, f1s,    w, label="F1",    color="#f7a96a",
                alpha=0.85, edgecolor="#0f0f1a")

    ax.bar_label(b1, fmt="%.3f", color="#e8e9f0", fontsize=8, padding=3)
    ax.bar_label(b2, fmt="%.3f", color="#e8e9f0", fontsize=8, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("PatchCore — AUROC & F1 by Category")
    ax.legend(facecolor="#16172b", edgecolor="#3a3b5c", labelcolor="#e8e9f0")

    fig.tight_layout()
    out = RESULTS_DIR / "aggregate_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Aggregate plot saved to {out}")


def _generate_markdown_report(df: pd.DataFrame) -> None:
    report_path = RESULTS_DIR / "evaluation_report.md"

    # Compute overall statistics
    mean_auroc = df["auroc"].mean()
    mean_f1 = df["f1"].mean()
    mean_ap = df["avg_prec"].mean() if "avg_prec" in df.columns else df["auroc"].mean()
    total_normal = df["n_normal"].sum()
    total_anomaly = df["n_anomaly"].sum()
    total_images = total_normal + total_anomaly

    md_content = []
    md_content.append("# 🔬 Industrial Anomaly Detection — Project Evaluation Report\n")
    md_content.append(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append("This report summarizes the performance of the PatchCore anomaly detection model (built on a ResNet-50 backbone) evaluated across all active categories in the project.\n")

    md_content.append("## 📊 Executive Summary\n")

    md_content.append("| Metric | Value | Description |")
    md_content.append("| :--- | :--- | :--- |")
    md_content.append(f"| **Mean AUROC** | **{mean_auroc:.4f}** | Area Under the ROC Curve (Image-level performance) |")
    md_content.append(f"| **Mean F1-Score** | **{mean_f1:.4f}** | Harmonic mean of precision and recall at the optimal threshold |")
    md_content.append(f"| **Mean Average Precision (AP)** | **{mean_ap:.4f}** | Precision-Recall curve area (robustness across thresholds) |")
    md_content.append(f"| **Total Evaluated Images** | **{total_images}** | Total images tested across all categories |")
    md_content.append(f"| **Normal / Anomaly Split** | **{total_normal} Normal / {total_anomaly} Defective** | Balance of test dataset |")
    md_content.append("\n")

    md_content.append("### 📈 Performance Overview Plot\n")
    md_content.append("Below is the summary bar chart showing performance of the PatchCore model by component category:\n\n")
    md_content.append("![Aggregate Metrics](./aggregate_metrics.png)\n\n")

    md_content.append("## 🏆 Detailed Metrics Table\n")
    md_content.append("The table below details evaluation metrics for each MVTec dataset category:\n")

    md_content.append("| Category | AUROC | F1-Score | Avg Precision | Optimal Threshold | Normal Samples | Anomaly Samples |")
    md_content.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for category, row in df.iterrows():
        ap_val = row.get("avg_prec", 0.0)
        thr_val = row.get("threshold", 0.0)
        n_norm = int(row.get("n_normal", 0))
        n_anom = int(row.get("n_anomaly", 0))
        md_content.append(f"| **{category}** | {row['auroc']:.4f} | {row['f1']:.4f} | {ap_val:.4f} | {thr_val:.4f} | {n_norm} | {n_anom} |")
    md_content.append("\n")

    best_cat_auroc = df["auroc"].idxmax()
    best_auroc = df["auroc"].max()
    worst_cat_auroc = df["auroc"].idxmin()
    worst_auroc = df["auroc"].min()

    md_content.append("## 💡 Key Insights & Diagnostics\n")
    md_content.append("> [!TIP]\n")
    md_content.append(f"> **Top Performing Category:** `{best_cat_auroc}` achieved an outstanding **{best_auroc:.4f} AUROC**. PatchCore features are extremely effective at representing normal structures for this component class.\n")
    md_content.append("\n")

    if worst_auroc < 0.90:
        md_content.append("> [!WARNING]\n")
        md_content.append(f"> **Lowest Performing Category:** `{worst_cat_auroc}` scored **{worst_auroc:.4f} AUROC**. For this category, you may want to:\n")
        md_content.append("> 1. Increase the coreset sampling ratio (`coreset_sampling_ratio` in `config.yaml`) to keep more reference patches in the memory bank.\n")
        md_content.append("> 2. Tune the patch-core neighborhood size (`patch_size` in `config.yaml`) or layers hook definition.\n")
        md_content.append("> 3. Adjust threshold percentile calibration (`threshold_percentile` in `config.yaml`).\n")
    else:
        md_content.append("> [!NOTE]\n")
        md_content.append("> **Excellent Generalization:** All evaluated categories achieved an AUROC greater than **90%**! This demonstrates the incredible robustness of the PatchCore coreset memory bank approach on the pre-trained ResNet-50 representation.\n")
    md_content.append("\n")

    md_content.append("---\n")
    md_content.append("## 🔍 Detailed Visualizations by Category\n")
    md_content.append("Below are the individual performance plots for each evaluated class:\n\n")

    for category in df.index:
        md_content.append(f"### 📦 Category: `{category}`\n")
        md_content.append("| ROC Curve | Precision-Recall | Anomaly Score Distribution |")
        md_content.append("| :---: | :---: | :---: |")
        md_content.append(f"| ![{category} ROC](./{category}/roc_curve.png) | ![{category} PR](./{category}/pr_curve.png) | ![{category} Scores](./{category}/score_distribution.png) |")
        md_content.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    log.info(f"Markdown report generated successfully at: {report_path}")


if __name__ == "__main__":
    main()
