"""
src/config.py
=============
Centralised configuration for AnomalyVision.
Override any value with an environment variable of the same name.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_PATH  = Path(os.getenv("MODEL_PATH",  str(BASE_DIR / "model_anomaly_detection.h5")))
DATASET_DIR = Path(os.getenv("DATASET_DIR", str(BASE_DIR / "UCSD_Anomaly_Dataset.v1p2")))
DB_PATH     = Path(os.getenv("DB_PATH",     str(BASE_DIR / "data.json")))

# ── Model / inference ─────────────────────────────────────────────────────────
FRAME_SIZE         : tuple[int, int] = (256, 256)
SEQ_LEN            : int   = 10
BATCH_SIZE         : int   = 4
DEFAULT_THRESHOLD  : float = 0.50   # regularity score below this → anomaly

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_EPOCHS       : int   = 50
TRAIN_LR           : float = 1e-4
TRAIN_DECAY        : float = 1e-5
EARLY_STOP_PATIENCE: int   = 7
