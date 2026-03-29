"""
scripts/train.py
================
Training pipeline for the ConvLSTM Autoencoder.

Usage
-----
python scripts/train.py \\
    --data_path UCSD_Anomaly_Dataset.v1p2 \\
    --dataset   UCSDped2 \\
    --epochs    50 \\
    --batch_size 4 \\
    --output    model_anomaly_detection.h5

The model is trained on *normal* sequences only (Train split).
Reconstruction error on unseen sequences is used as the anomaly score.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_model, compile_model
from src.config import FRAME_SIZE, SEQ_LEN, BATCH_SIZE, TRAIN_EPOCHS


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ucsd_train(data_root: str, subset: str = "UCSDped2") -> np.ndarray:
    """
    Load all training sequences from a UCSD subset.

    Returns
    -------
    np.ndarray  shape (N_sequences, SEQ_LEN, H, W, 1)
    """
    from PIL import Image

    train_dir = Path(data_root) / subset / "Train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    all_sequences = []

    for seq_folder in sorted(train_dir.iterdir()):
        if not seq_folder.is_dir():
            continue

        frames = []
        for fname in sorted(seq_folder.iterdir()):
            if fname.suffix.lower() == ".tif":
                img = Image.open(fname).convert("L").resize(FRAME_SIZE)
                frames.append(np.array(img, dtype=np.float32) / 255.0)

        if len(frames) < SEQ_LEN:
            continue

        # Sliding window — two passes with different strides for data augmentation
        for stride in (1, 2):
            for start in range(0, len(frames) - SEQ_LEN, stride):
                clip = np.stack(frames[start : start + SEQ_LEN])  # (10, H, W)
                all_sequences.append(clip)

    sequences = np.array(all_sequences)             # (N, 10, H, W)
    sequences = sequences[..., np.newaxis]           # (N, 10, H, W, 1)
    print(f"Loaded {len(sequences)} training sequences from {subset}/Train")
    return sequences


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(output_path: str, patience: int = 7):
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_path,
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="logs/", histogram_freq=1,
        ),
    ]


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    import tensorflow as tf

    print(f"\n{'='*60}")
    print(f"  AnomalyVision — ConvLSTM Autoencoder Training")
    print(f"{'='*60}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch_size}")
    print(f"  Output   : {args.output}")
    print(f"{'='*60}\n")

    # Load data
    sequences = load_ucsd_train(args.data_path, args.dataset)

    # Train / validation split (90 / 10)
    split     = int(len(sequences) * 0.9)
    idx       = np.random.permutation(len(sequences))
    train_seq = sequences[idx[:split]]
    val_seq   = sequences[idx[split:]]
    print(f"Train: {len(train_seq)}  |  Val: {len(val_seq)}")

    # Build & compile model
    model = build_model(SEQ_LEN, *FRAME_SIZE)
    model = compile_model(model, learning_rate=args.lr)
    model.summary()

    # Train
    history = model.fit(
        train_seq, train_seq,               # autoencoder: input == target
        validation_data=(val_seq, val_seq),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=get_callbacks(args.output, patience=args.patience),
        shuffle=True,
    )

    # Save final model
    model.save(args.output)
    print(f"\nModel saved → {args.output}")

    # Plot training curves
    _plot_history(history, save_path=str(Path(args.output).parent / "training_curves.png"))


def _plot_history(history, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("MSE Loss", fontsize=13)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Learning rate (if ReduceLROnPlateau fired)
    if "lr" in history.history:
        axes[1].semilogy(history.history["lr"], color="orange", label="LR")
        axes[1].set_title("Learning Rate Schedule", fontsize=13)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("LR (log scale)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved → {save_path}")
    plt.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the ConvLSTM Autoencoder on UCSD Pedestrian data."
    )
    p.add_argument("--data_path",  default="UCSD_Anomaly_Dataset.v1p2",
                   help="Root directory of the UCSD dataset")
    p.add_argument("--dataset",    default="UCSDped2",
                   choices=["UCSDped1", "UCSDped2"],
                   help="Which UCSD subset to train on")
    p.add_argument("--epochs",     type=int,   default=TRAIN_EPOCHS)
    p.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--patience",   type=int,   default=7)
    p.add_argument("--output",     default="model_anomaly_detection.h5",
                   help="Path to save the trained model")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
