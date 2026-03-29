"""
src/visualization.py
====================
Plotting utilities for AnomalyVision results.
All functions return matplotlib Figure objects ready for st.pyplot().
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Regularity timeline ────────────────────────────────────────────────────────

def plot_regularity_score(regularity: np.ndarray,
                          threshold: float = 0.5) -> plt.Figure:
    """
    Line chart of per-frame regularity scores with anomaly regions shaded.
    """
    fig, ax = plt.subplots(figsize=(13, 4))
    frames  = np.arange(len(regularity))

    ax.fill_between(frames, 0, 1,
                    where=(regularity < threshold),
                    color="#ef4444", alpha=0.15, label="Anomaly Region")
    ax.plot(frames, regularity, color="#3b82f6", linewidth=1.8,
            label="Regularity Score")
    ax.axhline(y=threshold, color="#ef4444", linestyle="--",
               linewidth=1.2, alpha=0.8, label=f"Threshold ({threshold:.2f})")

    ax.set_xlim(0, max(len(regularity) - 1, 1))
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Frame Index", fontsize=12)
    ax.set_ylabel("Regularity Score", fontsize=12)
    ax.set_title("Anomaly Detection — Regularity Score Over Time",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


# ── Error distribution histogram ───────────────────────────────────────────────

def plot_error_distribution(errors: np.ndarray,
                            anomaly_flags: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))

    normal_e = errors[~anomaly_flags]
    anomal_e = errors[ anomaly_flags]

    if len(normal_e):
        ax.hist(normal_e, bins=30, color="#22c55e", alpha=0.6, label="Normal")
    if len(anomal_e):
        ax.hist(anomal_e, bins=30, color="#ef4444", alpha=0.6, label="Anomaly")

    ax.set_xlabel("Reconstruction Error", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Reconstruction Error Distribution",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ── Heatmap overlay ────────────────────────────────────────────────────────────

def create_heatmap_overlay(frame: np.ndarray, heatmap: np.ndarray,
                           alpha: float = 0.45) -> np.ndarray:
    """
    Blend a reconstruction-error heatmap over a grayscale frame.

    Parameters
    ----------
    frame   : (H, W) float32 in [0, 1]
    heatmap : (H, W) float32 in [0, 1]
    alpha   : heatmap opacity

    Returns
    -------
    (H, W, 3) uint8 RGB image
    """
    frame_bgr  = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    heat_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay    = cv2.addWeighted(frame_bgr, 1 - alpha, heat_color, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ── Frame comparison ───────────────────────────────────────────────────────────

def plot_frame_comparison(sequences: np.ndarray,
                          reconstructed: np.ndarray,
                          heatmaps: np.ndarray,
                          regularity: np.ndarray,
                          anomaly_flags: np.ndarray,
                          idx: int) -> plt.Figure:
    """
    Three-panel figure: original | reconstructed | heatmap overlay.
    """
    orig    = sequences[idx, -1, :, :, 0]
    recon   = reconstructed[idx, -1, :, :, 0]
    hmap    = heatmaps[idx]
    overlay = create_heatmap_overlay(orig, hmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig, cmap="gray")
    axes[0].set_title("Original Frame", fontsize=13, fontweight="bold")

    axes[1].imshow(recon, cmap="gray")
    axes[1].set_title("Reconstructed Frame", fontsize=13, fontweight="bold")

    axes[2].imshow(overlay)
    axes[2].set_title("Anomaly Heatmap", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.axis("off")

    score  = float(regularity[idx])
    status = "ANOMALY DETECTED" if anomaly_flags[idx] else "NORMAL"
    color  = "#ef4444" if anomaly_flags[idx] else "#22c55e"
    fig.suptitle(f"Frame {idx}  |  Regularity: {score:.3f}  |  {status}",
                 fontsize=13, color=color, fontweight="bold", y=1.01)

    plt.tight_layout()
    return fig


# ── ROC curve ─────────────────────────────────────────────────────────────────

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray,
                   auc: float, label: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3b82f6", linewidth=2,
            label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Frame-level Anomaly Detection",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig
