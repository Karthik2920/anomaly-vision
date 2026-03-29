"""
scripts/evaluate.py
===================
Frame-level evaluation of the anomaly detector against UCSD ground truth.

Computes
--------
- AUC-ROC  (Area Under the Receiver Operating Characteristic Curve)
- EER      (Equal Error Rate)
- AP       (Average Precision / area under PR curve)
- Best-F1  (F1 at the threshold that maximises it)

Ground truth
------------
The UCSD Ped1 test set ships with per-frame binary annotations.
We derive them here from the published frame-range tables so no extra
files are needed (source: UCSD Anomaly Detection Dataset README, 2010).

Usage
-----
python scripts/evaluate.py \\
    --data_path UCSD_Anomaly_Dataset.v1p2 \\
    --dataset   UCSDped1 \\
    --model     model_anomaly_detection.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FRAME_SIZE, SEQ_LEN


# ── Published ground-truth frame ranges (UCSD Ped1 Test, 1-indexed) ──────────
# Each entry: list of (start, end) inclusive ranges that contain anomalies.
UCSD_PED1_GT: dict[str, list[tuple[int, int]]] = {
    "Test001": [(60,  152)],
    "Test002": [(50,  175)],
    "Test003": [(91,  200)],
    "Test004": [(31,  168)],
    "Test005": [(5,    90), (140, 200)],
    "Test006": [(1,   100), (110, 200)],
    "Test007": [(1,   175)],
    "Test008": [(1,    94)],
    "Test009": [(1,    48)],
    "Test010": [(1,   140)],
    "Test011": [(70,  165)],
    "Test012": [(130, 200)],
    "Test013": [(1,   156)],
    "Test014": [(1,   200)],
    "Test015": [(138, 200)],
    "Test016": [(123, 200)],
    "Test017": [(1,    47)],
    "Test018": [(54,  120)],
    "Test019": [(64,  138)],
    "Test020": [(45,  175)],
    "Test021": [(31,  200)],
    "Test022": [(16,  107)],
    "Test023": [(8,   165)],
    "Test024": [(50,  171)],
    "Test025": [(1,   165)],
    "Test026": [(86,  200)],
    "Test027": [(15,  139)],
    "Test028": [(15,  200)],
    "Test029": [(40,  200)],
    "Test030": [(77,  200)],
    "Test031": [(10,  122)],
    "Test032": [(105, 200)],
    "Test033": [(1,   200)],
    "Test034": [(5,   165)],
    "Test035": [(1,    45)],
    "Test036": [(175, 200)],
}


# ── Frame loading ──────────────────────────────────────────────────────────────

def load_frames(folder: Path) -> np.ndarray:
    from PIL import Image
    frames = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() == ".tif":
            img = Image.open(f).convert("L").resize(FRAME_SIZE)
            frames.append(np.array(img, dtype=np.float32) / 255.0)
    return np.array(frames)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_regularity(model, frames: np.ndarray) -> np.ndarray:
    """Return per-frame regularity scores (length = len(frames) - SEQ_LEN)."""
    sz        = len(frames) - SEQ_LEN
    sequences = np.stack([frames[i : i + SEQ_LEN] for i in range(sz)])
    sequences = sequences[..., np.newaxis]

    recon  = model.predict(sequences, batch_size=4, verbose=0)
    errors = np.array([np.linalg.norm(sequences[i] - recon[i]) for i in range(sz)])

    norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
    return 1.0 - norm          # regularity: high = normal, low = anomaly


# ── Ground-truth helpers ───────────────────────────────────────────────────────

def build_gt_vector(seq_name: str, n_frames: int,
                    gt_table: dict) -> np.ndarray | None:
    """
    Build a binary ground-truth vector (1 = anomaly) for `seq_name`.
    Returns None if the sequence has no entry in `gt_table`.
    """
    if seq_name not in gt_table:
        return None
    gt = np.zeros(n_frames, dtype=int)
    for start, end in gt_table[seq_name]:
        gt[start - 1 : end] = 1        # convert 1-indexed to 0-indexed
    return gt


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """
    Compute AUC-ROC, EER, Average Precision, and best-F1.

    Parameters
    ----------
    y_true  : binary ground truth (1 = anomaly)
    scores  : anomaly score — here we use (1 - regularity) so higher = more anomalous
    """
    from sklearn.metrics import (roc_auc_score, roc_curve,
                                  average_precision_score,
                                  precision_recall_curve, f1_score)
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc         = roc_auc_score(y_true, scores)

    # EER: point where FPR == FNR  (FNR = 1 - TPR)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    ap = average_precision_score(y_true, scores)

    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1s            = 2 * prec * rec / (prec + rec + 1e-8)
    best_f1        = float(f1s.max())
    best_thr       = float(thr[f1s[:-1].argmax()]) if len(thr) else 0.5

    return {
        "auc"     : float(auc),
        "eer"     : float(eer),
        "ap"      : float(ap),
        "best_f1" : best_f1,
        "best_thr": best_thr,
        "fpr"     : fpr,
        "tpr"     : tpr,
    }


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    import tensorflow as tf

    print(f"\n{'='*60}")
    print(f"  AnomalyVision — Benchmark Evaluation")
    print(f"{'='*60}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Model   : {args.model}")
    print(f"{'='*60}\n")

    model    = tf.keras.models.load_model(args.model)
    test_dir = Path(args.data_path) / args.dataset / "Test"
    gt_table = UCSD_PED1_GT if args.dataset == "UCSDped1" else {}

    all_gt, all_scores = [], []
    seq_results        = {}

    for seq_folder in sorted(test_dir.iterdir()):
        if not seq_folder.is_dir():
            continue

        name   = seq_folder.name
        frames = load_frames(seq_folder)
        if len(frames) < SEQ_LEN + 1:
            continue

        regularity = predict_regularity(model, frames)
        anomaly_score = 1.0 - regularity          # higher = more anomalous

        n_scored = len(anomaly_score)
        gt = build_gt_vector(name, n_scored, gt_table)

        if gt is not None and len(gt) == n_scored:
            all_gt.append(gt)
            all_scores.append(anomaly_score)
            seq_results[name] = {"regularity": regularity, "gt": gt}
            print(f"  {name}: {n_scored} frames  |  "
                  f"anomaly ratio = {gt.mean()*100:.1f}%")
        else:
            print(f"  {name}: {n_scored} frames  |  no ground truth")

    if not all_gt:
        print("\nNo ground-truth sequences found. Cannot compute metrics.")
        print("Ground truth is only available for UCSDped1.")
        return

    y_true    = np.concatenate(all_gt)
    scores    = np.concatenate(all_scores)
    metrics   = compute_metrics(y_true, scores)

    print(f"\n{'─'*40}")
    print(f"  AUC-ROC  : {metrics['auc']:.4f}")
    print(f"  EER      : {metrics['eer']:.4f}  ({metrics['eer']*100:.2f}%)")
    print(f"  Avg Prec : {metrics['ap']:.4f}")
    print(f"  Best F1  : {metrics['best_f1']:.4f}  (thr={metrics['best_thr']:.3f})")
    print(f"{'─'*40}\n")

    # Save ROC plot
    from src.visualization import plot_roc_curve
    fig = plot_roc_curve(metrics["fpr"], metrics["tpr"],
                         metrics["auc"], label=args.dataset)
    out_png = Path(args.model).parent / f"roc_{args.dataset}.png"
    fig.savefig(str(out_png), dpi=150)
    print(f"ROC curve saved → {out_png}")
    plt.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate anomaly detector on UCSD benchmark.")
    p.add_argument("--data_path", default="UCSD_Anomaly_Dataset.v1p2")
    p.add_argument("--dataset",   default="UCSDped1",
                   choices=["UCSDped1", "UCSDped2"])
    p.add_argument("--model",     default="model_anomaly_detection.h5")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
