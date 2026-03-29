"""
AnomalyVision — Real-Time Surveillance Video Anomaly Detection
==============================================================
A deep learning system for detecting anomalies in surveillance
footage using a ConvLSTM Autoencoder architecture.

Author: Built with TensorFlow & Streamlit
Dataset: UCSD Pedestrian Anomaly Detection Dataset
"""

import os
import json
import hashlib
import secrets
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AnomalyVision",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1f2937;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .info-card {
        background: linear-gradient(135deg, #f8faff 0%, #e8eeff 100%);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
FRAME_SIZE   = (256, 256)
SEQ_LEN      = 10
DB_PATH      = Path(__file__).parent / "data.json"
BASE_DIR     = Path(__file__).parent
MODEL_PATHS  = [
    BASE_DIR / "model_anomaly_detection.h5",
    BASE_DIR / "model_anomaly.keras",
]
# GitHub Releases URL — update tag if you create a new release
MODEL_RELEASE_URL = (
    "https://github.com/Karthik2920/anomaly-vision/releases/download"
    "/v1.0/model_anomaly_detection.h5"
)

# ── Model auto-download ────────────────────────────────────────────────────────
def _ensure_model() -> Optional[Path]:
    """
    Return path to the model file.
    If not present locally, download it from GitHub Releases.
    """
    for p in MODEL_PATHS:
        if p.exists():
            return p
    # Try to download
    dest = BASE_DIR / "model_anomaly_detection.h5"
    try:
        import urllib.request
        with st.spinner("Downloading model from GitHub Releases (~23 MB)…"):
            urllib.request.urlretrieve(MODEL_RELEASE_URL, str(dest))
        if dest.exists():
            return dest
    except Exception as e:
        st.warning(f"Could not download model: {e}")
    return None

# ── Model loading (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading anomaly detection model…")
def load_model():
    import os
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    import tensorflow as tf
    path = _ensure_model()
    if path is None:
        return None
    try:
        # Try tf_keras first (best compatibility with models saved in TF2/Keras2)
        import tf_keras
        return tf_keras.models.load_model(str(path))
    except Exception:
        return tf.keras.models.load_model(str(path))


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames_from_video(video_bytes: bytes):
    """Extract normalised grayscale frames from uploaded video bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_bytes)
        tmp_path = f.name

    cap  = cv2.VideoCapture(tmp_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, FRAME_SIZE)
        frames.append(resized.astype(np.float32) / 255.0)

    cap.release()
    os.unlink(tmp_path)
    return np.array(frames), fps


def load_tif_sequence(folder_path: str) -> np.ndarray:
    """Load a .tif image sequence from a UCSD dataset folder."""
    frames = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(".tif"):
            img = Image.open(os.path.join(folder_path, fname)).convert("L")
            img = img.resize(FRAME_SIZE)
            frames.append(np.array(img, dtype=np.float32) / 255.0)
    return np.array(frames)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(model, frames: np.ndarray, threshold: float = 0.5) -> Optional[dict]:
    """
    Slide a window over the frame sequence, reconstruct with the autoencoder,
    and compute per-frame regularity scores.

    Returns a results dict or None if there aren't enough frames.
    """
    n = len(frames)
    if n < SEQ_LEN:
        return None

    sz        = n - SEQ_LEN
    sequences = np.stack([frames[i : i + SEQ_LEN] for i in range(sz)])
    sequences = sequences[..., np.newaxis]           # (sz, 10, 256, 256, 1)

    reconstructed = model.predict(sequences, batch_size=4, verbose=0)

    # Per-sequence L2 reconstruction error
    errors = np.array([
        np.linalg.norm(sequences[i] - reconstructed[i])
        for i in range(sz)
    ])

    norm_errors  = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
    regularity   = 1.0 - norm_errors
    anomaly_mask = regularity < threshold

    # Per-frame spatial error heatmaps (last frame of each sequence)
    heatmaps = []
    for i in range(sz):
        orig  = sequences[i, -1, :, :, 0]
        recon = reconstructed[i, -1, :, :, 0]
        err   = np.abs(orig - recon)
        err   = (err - err.min()) / (err.max() - err.min() + 1e-8)
        heatmaps.append(err)

    return {
        "regularity"    : regularity,
        "errors"        : errors,
        "anomaly_flags" : anomaly_mask,
        "heatmaps"      : np.array(heatmaps),
        "sequences"     : sequences,
        "reconstructed" : reconstructed,
        "n_frames"      : sz,
        "anomaly_ratio" : anomaly_mask.mean(),
        "anomaly_frames": np.where(anomaly_mask)[0].tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_regularity_score(regularity: np.ndarray, threshold: float) -> plt.Figure:
    """Regularity-score timeline with anomaly regions shaded."""
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


def create_heatmap_overlay(frame: np.ndarray, heatmap: np.ndarray,
                           alpha: float = 0.45) -> np.ndarray:
    """Blend reconstruction-error heatmap over original grayscale frame (RGB out)."""
    frame_bgr   = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    heat_color  = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay     = cv2.addWeighted(frame_bgr, 1 - alpha, heat_color, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def plot_frame_comparison(results: dict, idx: int) -> plt.Figure:
    """Show original | reconstructed | heatmap-overlay for a given frame index."""
    orig    = results["sequences"][idx, -1, :, :, 0]
    recon   = results["reconstructed"][idx, -1, :, :, 0]
    hmap    = results["heatmaps"][idx]
    overlay = create_heatmap_overlay(orig, hmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig, cmap="gray");       axes[0].set_title("Original Frame",      fontsize=13, fontweight="bold")
    axes[1].imshow(recon, cmap="gray");      axes[1].set_title("Reconstructed Frame", fontsize=13, fontweight="bold")
    axes[2].imshow(overlay);                 axes[2].set_title("Anomaly Heatmap",      fontsize=13, fontweight="bold")

    for ax in axes:
        ax.axis("off")

    score  = results["regularity"][idx]
    status = "ANOMALY DETECTED" if results["anomaly_flags"][idx] else "NORMAL"
    color  = "#ef4444" if results["anomaly_flags"][idx] else "#22c55e"
    fig.suptitle(f"Frame {idx}  |  Regularity: {score:.3f}  |  {status}",
                 fontsize=13, color=color, fontweight="bold", y=1.01)

    plt.tight_layout()
    return fig


def plot_error_distribution(results: dict) -> plt.Figure:
    """Histogram of reconstruction errors to visualise separation."""
    fig, ax = plt.subplots(figsize=(8, 4))

    normal_e  = results["errors"][~results["anomaly_flags"]]
    anomal_e  = results["errors"][ results["anomaly_flags"]]

    if len(normal_e):
        ax.hist(normal_e, bins=30, color="#22c55e", alpha=0.6, label="Normal")
    if len(anomal_e):
        ax.hist(anomal_e, bins=30, color="#ef4444", alpha=0.6, label="Anomaly")

    ax.set_xlabel("Reconstruction Error", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Reconstruction Error Distribution", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════

def _hash_password(password: str) -> str:
    salt   = secrets.token_hex(16)
    digest = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{digest}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split(":", 1)
        return hashlib.sha256((salt + password).encode()).hexdigest() == digest
    except ValueError:
        return password == stored          # legacy plaintext fall-through


def _load_db() -> dict:
    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        return {"users": []}
    with open(DB_PATH) as f:
        return json.load(f)


def _save_db(data: dict) -> None:
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=2)


def create_user(name: str, email: str, age: int, sex: str, password: str) -> str:
    """Returns 'ok', 'duplicate', or 'error'."""
    try:
        db = _load_db()
        if any(u["email"] == email for u in db["users"]):
            return "duplicate"
        db["users"].append({
            "name": name, "email": email, "age": int(age),
            "sex": sex, "password": _hash_password(password),
        })
        _save_db(db)
        return "ok"
    except Exception:
        return "error"


def authenticate(email: str, password: str) -> Optional[dict]:
    db = _load_db()
    for user in db["users"]:
        if user["email"] == email and _verify_password(password, user["password"]):
            return user
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_csv_report(results: dict) -> str:
    df = pd.DataFrame({
        "frame"               : range(results["n_frames"]),
        "regularity_score"    : results["regularity"].round(4),
        "reconstruction_error": results["errors"].round(4),
        "is_anomaly"          : results["anomaly_flags"].astype(int),
    })
    return df.to_csv(index=False)


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def display_results(results: dict, threshold: float) -> None:
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    # ── Summary KPIs ──
    n_anom  = int(results["anomaly_flags"].sum())
    n_total = results["n_frames"]
    anom_pc = results["anomaly_ratio"] * 100
    avg_sc  = float(results["regularity"].mean())
    verdict = "⚠️ Anomaly Detected" if n_anom > 0 else "✅ No Anomaly"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames Analysed",   n_total)
    c2.metric("Anomalous Frames",  n_anom,
              delta=f"{anom_pc:.1f}%", delta_color="inverse")
    c3.metric("Avg Regularity",    f"{avg_sc:.3f}")
    c4.metric("Verdict",           verdict)

    st.markdown("---")

    # ── Regularity plot ──
    st.markdown("### Regularity Score Timeline")
    st.caption("Frames below the threshold (red dashed line) are flagged as anomalies.")
    fig = plot_regularity_score(results["regularity"], threshold)
    st.pyplot(fig); plt.close()

    # ── Error distribution ──
    with st.expander("📈 Reconstruction Error Distribution"):
        fig2 = plot_error_distribution(results)
        st.pyplot(fig2); plt.close()

    # ── Frame inspector ──
    if results["anomaly_frames"]:
        st.markdown("### 🔍 Anomalous Frame Inspector")
        st.caption("Select a flagged frame to inspect the original, reconstruction, and spatial error heatmap.")

        sel = st.select_slider(
            "Anomalous frame index",
            options=results["anomaly_frames"],
        )
        fig3 = plot_frame_comparison(results, sel)
        st.pyplot(fig3); plt.close()
        st.caption("**Left:** Original frame  |  **Center:** Model reconstruction  |  **Right:** Pixel-wise error heatmap (red = high anomaly)")

        # Show anomaly timeline as table
        with st.expander("📋 All Anomalous Frames"):
            df_anom = pd.DataFrame({
                "Frame"            : results["anomaly_frames"],
                "Regularity Score" : [round(float(results["regularity"][i]), 4)
                                      for i in results["anomaly_frames"]],
            })
            st.dataframe(df_anom, use_container_width=True)
    else:
        st.success("No anomalies detected in this sequence.")

    # ── Export ──
    st.markdown("### 📥 Export Report")
    csv = generate_csv_report(results)
    st.download_button(
        label="⬇️ Download Anomaly Report (CSV)",
        data=csv,
        file_name="anomaly_report.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════════════════

def page_home() -> None:
    st.markdown('<p class="main-header">AnomalyVision</p>', unsafe_allow_html=True)
    st.markdown("#### Real-Time Surveillance Video Anomaly Detection using Deep Learning")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🧠 ConvLSTM Autoencoder**\n\nLearns normal motion patterns; flags deviations as anomalies")
    with col2:
        st.info("**🎥 Video Upload**\n\nAnalyse any `.mp4` / `.avi` surveillance footage in seconds")
    with col3:
        st.info("**🔥 Spatial Heatmaps**\n\nVisualise exactly *where* in the frame an anomaly occurs")

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""
### How It Works

The model is a **sequence-to-sequence ConvLSTM Autoencoder** trained exclusively on
*normal* pedestrian activity from the UCSD Pedestrian Dataset.

At inference time the model attempts to reconstruct every 10-frame clip.
High reconstruction error indicates the model has seen something unexpected —
i.e. an **anomaly**.

#### Key Features
- Upload `.mp4` / `.avi` video for instant analysis
- Interactive **threshold slider** to tune sensitivity
- Per-frame **spatial heatmaps** showing anomaly location
- Side-by-side original vs. reconstructed frame comparison
- UCSD **Dataset Browser** for benchmark evaluation
- One-click **CSV export** of all frame-level scores
        """)

    with col_r:
        st.markdown("""
### Model Architecture
```
Input  (batch, 10, 256×256, 1)
       │
  ┌────▼─────┐
  │ ENCODER  │ TimeDistributed Conv2D ×2
  └────┬─────┘ (128→64 filters)
       │
  ┌────▼──────────┐
  │ TEMPORAL CORE │ ConvLSTM2D: 64→32→64
  └────┬──────────┘
       │
  ┌────▼─────┐
  │ DECODER  │ ConvTranspose2D ×2
  └────┬─────┘ (64→128 filters)
       │
Output (batch, 10, 256×256, 1)
```
**Parameters:** ~1.96 M
**Loss:** Mean Squared Error
**Optimizer:** Adam (lr = 1e-4)
**Dataset:** UCSD Pedestrian (Ped1 + Ped2)
        """)

    st.markdown("---")
    st.markdown("""
### Dataset — UCSD Pedestrian Anomaly Detection

The UCSD dataset captures pedestrian walkways from a stationary camera.
Anomalies include non-pedestrian objects (skateboards, bicycles, carts) and
abnormal motion patterns.

| Subset   | Train seqs | Test seqs | Avg frames |
|----------|-----------|----------|-----------|
| UCSDped1 | 34        | 36       | 200       |
| UCSDped2 | 16        | 12       | 200       |
    """)


def page_analyze() -> None:
    st.markdown("## 🎯 Analyze Video")

    if not st.session_state.get("logged_in"):
        st.warning("Please **Login** (sidebar → Login/Signup) to use the analyser.")
        return

    model = load_model()
    if model is None:
        st.error("Model file not found. Ensure `model_anomaly_detection.h5` or "
                 "`model_anomaly.keras` is in the project root.")
        return

    # Threshold control (sidebar)
    threshold = st.sidebar.slider(
        "Anomaly Threshold",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Regularity scores below this value are flagged as anomalies. "
             "Lower value = more sensitive.",
    )

    tab_upload, tab_ucsd = st.tabs(["📤 Upload Video", "🗂️ UCSD Dataset Browser"])

    # ─── Tab 1: Upload ────────────────────────────────────────────────────────
    with tab_upload:
        st.markdown("Upload any surveillance video file for anomaly analysis.")
        uploaded = st.file_uploader(
            "Choose a video file", type=["mp4", "avi", "mov"],
            help="Max file size depends on your Streamlit server config.",
        )

        if uploaded:
            st.video(uploaded)          # preview
            if st.button("▶ Run Anomaly Detection", type="primary"):
                with st.spinner("Extracting frames from video…"):
                    frames, fps = extract_frames_from_video(uploaded.read())
                st.info(f"Extracted **{len(frames)}** frames at **{fps:.1f} FPS**")

                with st.spinner("Running model inference…"):
                    results = run_inference(model, frames, threshold)

                if results:
                    st.session_state["results"] = results
                    display_results(results, threshold)
                else:
                    st.error("Too few frames for analysis — need at least 10.")

    # ─── Tab 2: UCSD Browser ──────────────────────────────────────────────────
    with tab_ucsd:
        dataset_root = BASE_DIR / "UCSD_Anomaly_Dataset.v1p2"
        if not dataset_root.exists():
            st.warning("UCSD dataset not found in project directory. "
                       "Download from: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                ped = st.selectbox("Subset", ["UCSDped1", "UCSDped2"])
            with c2:
                split = st.selectbox("Split", ["Test", "Train"])
            with c3:
                seq_dir = dataset_root / ped / split
                if seq_dir.exists():
                    seqs = sorted(
                        d for d in os.listdir(seq_dir)
                        if os.path.isdir(seq_dir / d)
                    )
                    sel_seq = st.selectbox("Sequence", seqs)
                else:
                    sel_seq = None
                    st.error("Path not found.")

            if sel_seq and st.button("▶ Analyse Sequence", type="primary"):
                seq_path = str(seq_dir / sel_seq)
                with st.spinner("Loading .tif frames…"):
                    frames = load_tif_sequence(seq_path)
                st.info(f"Loaded **{len(frames)}** frames from "
                        f"`{ped}/{split}/{sel_seq}`")

                with st.spinner("Running model inference…"):
                    results = run_inference(model, frames, threshold)

                if results:
                    st.session_state["results"] = results
                    display_results(results, threshold)
                else:
                    st.error("Too few frames for analysis — need at least 10.")


def page_dashboard() -> None:
    st.markdown("## 👤 Dashboard")

    if not st.session_state.get("logged_in"):
        st.warning("Please login to view the dashboard.")
        return

    user = st.session_state["user_info"]

    c1, c2, c3 = st.columns(3)
    c1.info(f"**Name:** {user['name']}")
    c2.info(f"**Email:** {user['email']}")
    c3.info(f"**Age:** {user.get('age', '—')}  |  **Sex:** {user.get('sex', '—')}")

    st.markdown("---")

    if st.session_state.get("results"):
        st.markdown("### Last Analysis Summary")
        results = st.session_state["results"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Frames Analysed", results["n_frames"])
        col2.metric("Anomaly Rate",    f"{results['anomaly_ratio']*100:.1f}%")
        col3.metric("Avg Regularity",  f"{results['regularity'].mean():.3f}")

        if st.button("Re-view Results"):
            display_results(results, st.session_state.get("threshold", 0.5))
    else:
        st.info("No analysis run yet. Go to **Analyze** to get started.")


def page_auth() -> None:
    st.markdown("## 🔐 Login / Sign Up")
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            email    = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", type="primary"):
                user = authenticate(email, password)
                if user:
                    st.session_state["logged_in"]  = True
                    st.session_state["user_info"]  = user
                    st.success(f"Welcome back, **{user['name']}**!")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    with tab_signup:
        with st.form("signup_form"):
            name     = st.text_input("Full Name")
            email    = st.text_input("Email")
            c1, c2   = st.columns(2)
            with c1:
                age  = st.number_input("Age", min_value=1, max_value=120, value=22)
            with c2:
                sex  = st.radio("Sex", ["Male", "Female", "Other"])
            password = st.text_input("Password", type="password")
            confirm  = st.text_input("Confirm Password", type="password")

            if st.form_submit_button("Create Account", type="primary"):
                if not all([name, email, password]):
                    st.error("Please fill in all fields.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    result = create_user(name, email, age, sex, password)
                    if result == "ok":
                        st.success("Account created! Please login.")
                    elif result == "duplicate":
                        st.error("An account with this email already exists.")
                    else:
                        st.error("Unexpected error. Please try again.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Initialise session state defaults
    for key, val in [("logged_in", False), ("results", None), ("threshold", 0.5)]:
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎯 AnomalyVision")
        st.caption("Surveillance Video Anomaly Detection")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Home", "🎯 Analyze", "👤 Dashboard", "🔐 Login / Sign Up"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        if st.session_state["logged_in"]:
            user = st.session_state["user_info"]
            st.success(f"Logged in as\n**{user['name']}**")
            if st.button("Logout", use_container_width=True):
                st.session_state["logged_in"] = False
                st.session_state.pop("user_info", None)
                st.rerun()
        else:
            st.info("Not logged in")

        st.markdown("---")
        st.markdown(
            "<small>Built with TensorFlow & Streamlit<br>"
            "Dataset: UCSD Pedestrian<br>"
            "Model: ConvLSTM Autoencoder</small>",
            unsafe_allow_html=True,
        )

    # ── Route ─────────────────────────────────────────────────────────────────
    routing = {
        "🏠 Home"            : page_home,
        "🎯 Analyze"         : page_analyze,
        "👤 Dashboard"       : page_dashboard,
        "🔐 Login / Sign Up" : page_auth,
    }
    routing[page]()


if __name__ == "__main__":
    main()
