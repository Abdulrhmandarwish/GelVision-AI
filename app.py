"""
GelVision AI — Gel Electrophoresis Analysis Platform
=====================================================

A production-grade Streamlit application for automated gel electrophoresis
analysis powered by YOLOv8 lane detection and U-Net band segmentation.

Usage:
    streamlit run app.py

Author: Abdulrhman
"""

import io
import time
import tempfile
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.preprocess import preprocess_gel
from src.pipeline import GelAnalysisPipeline, load_yolo_model, load_unet_model
from src.report import generate_csv, generate_excel, generate_pdf

# ---------------------------------------------------------------------------
# Page Configuration — must be the FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GelVision AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark Laboratory Theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Keyframe Animations ─────────────────────────────────────────── */
@keyframes pulse-glow {
    0%   { box-shadow: 0 0 5px  rgba(0, 212, 184, 0.2); }
    50%  { box-shadow: 0 0 20px rgba(0, 212, 184, 0.4); }
    100% { box-shadow: 0 0 5px  rgba(0, 212, 184, 0.2); }
}

@keyframes gradient-shift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Global Layout ───────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #0A0F1E;
}

[data-testid="stHeader"] {
    background: rgba(10, 15, 30, 0.8);
    backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid rgba(0, 212, 184, 0.15);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: #111827;
}

/* ── Typography ──────────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #E2E8F0 !important;
}

body, p, span, div, label, .stMarkdown {
    font-family: 'Inter', sans-serif !important;
    color: #E2E8F0;
}

/* ── Buttons ─────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #00D4B8 0%, #00A896 100%);
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.6rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #00E8CA 0%, #00D4B8 100%);
    box-shadow: 0 0 24px rgba(0, 212, 184, 0.45),
                0 4px 12px rgba(0, 0, 0, 0.3);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0);
}

/* ── Download Buttons ────────────────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    color: #00D4B8 !important;
    border: 1px solid rgba(0, 212, 184, 0.3);
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stDownloadButton > button:hover {
    border-color: #00D4B8;
    box-shadow: 0 0 16px rgba(0, 212, 184, 0.3);
    background: linear-gradient(135deg, #1E3A5F 0%, #111827 100%);
}

/* ── Metrics ─────────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    color: #00D4B8 !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── DataFrames & Tables ─────────────────────────────────────────── */
.stDataFrame, [data-testid="stDataFrame"] {
    background: #111827;
    border-radius: 10px;
    overflow: hidden;
}

.stDataFrame table {
    background: #111827 !important;
}

.stDataFrame th {
    background: #1E293B !important;
    color: #00D4B8 !important;
    font-weight: 600 !important;
}

.stDataFrame td {
    color: #E2E8F0 !important;
    border-bottom: 1px solid rgba(0, 212, 184, 0.08) !important;
}

/* ── Expanders ───────────────────────────────────────────────────── */
.stExpander {
    background: #111827;
    border: 1px solid rgba(0, 212, 184, 0.15);
    border-radius: 10px;
}

[data-testid="stExpander"] details {
    background: #111827 !important;
    border: 1px solid rgba(0, 212, 184, 0.15) !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary {
    color: #E2E8F0 !important;
    font-weight: 500;
}

/* ── Tabs ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 2px solid rgba(0, 212, 184, 0.2);
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    color: #94A3B8 !important;
    background: transparent;
    border: none;
    padding: 0.75rem 1.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    font-size: 1rem;
    transition: color 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #00D4B8 !important;
}

.stTabs [aria-selected="true"] {
    color: #00D4B8 !important;
    border-bottom: 2px solid #00D4B8 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #00D4B8 !important;
}

/* ── File Uploader ───────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #111827;
    border: 2px dashed rgba(0, 212, 184, 0.3);
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 212, 184, 0.6);
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
}

[data-testid="stFileUploader"] label {
    color: #94A3B8 !important;
}

/* ── Sliders ─────────────────────────────────────────────────────── */
[data-testid="stSlider"] label {
    color: #94A3B8 !important;
}

[data-baseweb="slider"] div[role="slider"] {
    background: #00D4B8 !important;
}

/* ── Select Boxes ────────────────────────────────────────────────── */
[data-baseweb="select"] {
    background: #1E293B !important;
}

[data-baseweb="select"] div {
    color: #E2E8F0 !important;
}

/* ── Progress Bar ────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #00D4B8, #00E8CA) !important;
}

/* ── Custom Components ───────────────────────────────────────────── */
.metric-card {
    background: #111827;
    border-left: 4px solid #00D4B8;
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25),
                0 0 1px rgba(0, 212, 184, 0.1);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35),
                0 0 12px rgba(0, 212, 184, 0.15);
}

.metric-card .metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #00D4B8;
    line-height: 1.1;
}

.metric-card .metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #94A3B8;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-card .metric-delta {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 0.15rem;
}

.quality-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.02em;
}

.badge-excellent {
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-warning {
    background: rgba(255, 107, 43, 0.15);
    color: #FF6B2B;
    border: 1px solid rgba(255, 107, 43, 0.3);
}

.badge-poor {
    background: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* ── Pipeline Flow Diagram ───────────────────────────────────────── */
.pipeline-step {
    background: #111827;
    border: 1px solid rgba(0, 212, 184, 0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.pipeline-step:hover {
    border-color: rgba(0, 212, 184, 0.5);
    box-shadow: 0 0 16px rgba(0, 212, 184, 0.15);
}

.pipeline-step .step-icon {
    font-size: 1.6rem;
    margin-bottom: 0.4rem;
}

.pipeline-step .step-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: #00D4B8;
    font-size: 0.9rem;
}

.pipeline-step .step-desc {
    font-family: 'Inter', sans-serif;
    color: #94A3B8;
    font-size: 0.75rem;
    margin-top: 0.2rem;
}

.pipeline-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #00D4B8;
    font-size: 1.5rem;
}

/* ── Tech Badge ──────────────────────────────────────────────────── */
.tech-badge {
    display: inline-block;
    background: #1E293B;
    color: #00D4B8;
    border: 1px solid rgba(0, 212, 184, 0.25);
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    font-size: 0.8rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    margin: 0.15rem;
}

/* ── Info Card (About page) ──────────────────────────────────────── */
.info-card {
    background: #111827;
    border: 1px solid rgba(0, 212, 184, 0.12);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.info-card h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #00D4B8 !important;
    margin-bottom: 0.6rem;
    font-size: 1.05rem;
}

.info-card p, .info-card li {
    color: #94A3B8 !important;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Scrollbar ───────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0A0F1E;
}

::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 184, 0.3);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 212, 184, 0.5);
}

/* ── Status Container ────────────────────────────────────────────── */
[data-testid="stStatusWidget"] {
    background: #111827 !important;
    border: 1px solid rgba(0, 212, 184, 0.15) !important;
    border-radius: 10px !important;
}

/* ── Divider ─────────────────────────────────────────────────────── */
hr {
    border-color: rgba(0, 212, 184, 0.12) !important;
}

/* ── Sidebar Section Headers ─────────────────────────────────────── */
.sidebar-section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
}

/* ── Logo Container ──────────────────────────────────────────────── */
.logo-container {
    text-align: center;
    padding: 1.2rem 0.5rem;
}

.logo-container .logo-icon {
    font-size: 2.8rem;
    margin-bottom: 0.3rem;
    animation: pulse-glow 3s ease-in-out infinite;
    display: inline-block;
}

.logo-container .logo-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #E2E8F0;
    letter-spacing: -0.02em;
}

.logo-container .logo-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 0.15rem;
}

/* ── Hero Header Gradient Underline ──────────────────────────────── */
.hero-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #E2E8F0;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid transparent;
    border-image: linear-gradient(90deg, #00D4B8, #0A0F1E) 1;
    margin-bottom: 1.2rem;
}

/* ── Model Info Card ─────────────────────────────────────────────── */
.model-info-card {
    background: #0F172A;
    border: 1px solid rgba(0, 212, 184, 0.1);
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.5rem;
}

.model-info-card .model-name {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: #E2E8F0;
    font-size: 0.85rem;
}

.model-info-card .model-metric {
    font-family: 'Inter', sans-serif;
    color: #00D4B8;
    font-size: 0.78rem;
    font-weight: 500;
}

/* ── Image Container ─────────────────────────────────────────────── */
.image-container {
    background: #111827;
    border: 1px solid rgba(0, 212, 184, 0.12);
    border-radius: 12px;
    padding: 0.8rem;
    overflow: hidden;
}

.image-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #94A3B8;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Footer ──────────────────────────────────────────────────────── */
.sidebar-footer {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: #475569;
    text-align: center;
    padding: 1rem 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "results" not in st.session_state:
    st.session_state.results = None
if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0.0
if "uploaded_image_array" not in st.session_state:
    st.session_state.uploaded_image_array = None
if "preprocessed_images" not in st.session_state:
    st.session_state.preprocessed_images = None


# ---------------------------------------------------------------------------
# Model Loading — cached for the entire session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_models() -> Tuple[Any, Any, torch.device]:
    """Load and cache both ML models for the GelVision AI pipeline.

    Returns:
        Tuple of (yolo_model, unet_model, device).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = load_yolo_model("models/best.pt")
    unet = load_unet_model("models/unet_best.pth", device)
    return yolo, unet, device


def load_pipeline() -> GelAnalysisPipeline:
    """Create a GelAnalysisPipeline with cached models.

    Returns:
        GelAnalysisPipeline: Initialized pipeline with YOLOv8 + U-Net models.
    """
    yolo, unet, device = _load_models()
    return GelAnalysisPipeline(
        yolo_model=yolo,
        unet_model=unet,
        device=device,
    )


def get_device_label() -> str:
    """Return a human-readable label for the current compute device."""
    return "CUDA (GPU)" if torch.cuda.is_available() else "CPU"


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------
LADDER_DISPLAY_NAMES: Dict[str, str] = {
    "1kb Plus DNA Ladder": "1kb_plus",
    "100 bp DNA Ladder": "100bp",
    "NEB 1kb DNA Ladder": "neb_1kb",
}


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to RGB for Streamlit display.

    Args:
        image: BGR numpy array from OpenCV.

    Returns:
        RGB numpy array.
    """
    if image is None:
        return image
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Apply a teal-inferno colormap to a binary segmentation mask.

    Args:
        mask: 2-D uint8 mask (0 or 1).

    Returns:
        RGB numpy array with colormap applied.
    """
    mask_scaled = (mask * 255).astype(np.uint8)
    colored = cv2.applyColorMap(mask_scaled, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _r2_quality_badge(r2: float) -> str:
    """Return an HTML quality badge string based on calibration R² value.

    Args:
        r2: Calibration R² score.

    Returns:
        HTML string for the quality badge.
    """
    if r2 >= 0.95:
        return '<span class="quality-badge badge-excellent">● Excellent</span>'
    elif r2 >= 0.80:
        return '<span class="quality-badge badge-warning">● Acceptable</span>'
    else:
        return '<span class="quality-badge badge-poor">● Poor</span>'


def _build_metric_card(value: str, label: str, delta: str = "") -> str:
    """Build HTML for a custom styled metric card.

    Args:
        value: The primary display value.
        label: Descriptive label below the value.
        delta: Optional secondary text.

    Returns:
        HTML string for the metric card.
    """
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # ── Logo ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="logo-container">
            <div class="logo-icon">🧬</div>
            <div class="logo-title">GelVision AI</div>
            <div class="logo-subtitle">Gel Electrophoresis Analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Analysis Settings ─────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-section-header">⚙️ Analysis Settings</div>',
        unsafe_allow_html=True,
    )

    conf_threshold: float = st.slider(
        "Lane Detection Confidence",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Minimum confidence score for YOLOv8 lane detection. "
             "Lower values detect more lanes but may include false positives.",
    )

    band_threshold: float = st.slider(
        "Band Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Minimum peak height for band detection in the segmentation "
             "profile. Lower values detect fainter bands.",
    )

    min_band_distance: int = st.slider(
        "Min Band Distance (px)",
        min_value=5,
        max_value=50,
        value=10,
        step=1,
        help="Minimum pixel distance between adjacent band peaks. "
             "Increase to merge very close bands.",
    )

    ladder_display = st.selectbox(
        "Ladder Type",
        options=list(LADDER_DISPLAY_NAMES.keys()),
        index=0,
        help="Select the molecular weight ladder used in the experiment.",
    )
    ladder_type: str = LADDER_DISPLAY_NAMES[ladder_display]

    ladder_override: int = st.number_input(
        "Ladder Lane Override",
        min_value=0,
        value=0,
        step=1,
        help="Manually specify the ladder lane number (1-indexed). "
             "Set to 0 for automatic detection.",
    )

    st.divider()

    # ── Model Information ─────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-section-header">🤖 Model Information</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="model-info-card">
            <div class="model-name">YOLOv8n — Lane Detection</div>
            <div class="model-metric">93.1% mAP50</div>
        </div>
        <div class="model-info-card">
            <div class="model-name">U-Net ResNet34 — Segmentation</div>
            <div class="model-metric">Dice 0.165</div>
        </div>
        <div class="model-info-card">
            <div class="model-name">Compute Device</div>
            <div class="model-metric">"""
        + get_device_label()
        + """</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Footer ────────────────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-footer">Built with Streamlit + PyTorch<br>'
        "© 2026 GelVision AI</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Content Area — Three Tabs
# ---------------------------------------------------------------------------
tab_analysis, tab_results, tab_about = st.tabs(
    ["🧬 Analysis", "📊 Results", "ℹ️ About"]
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — ANALYSIS                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab_analysis:
    st.markdown(
        '<div class="hero-header">Upload &amp; Analyze</div>',
        unsafe_allow_html=True,
    )

    import os
    use_sample = st.checkbox("🧬 Use demo sample gel image for quick testing")

    uploaded_file = None
    if not use_sample:
        uploaded_file = st.file_uploader(
            "Drop your gel image here or click to browse",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="Supported formats: JPEG, PNG, TIFF. Maximum size: 50 MB.",
        )

    if use_sample:
        sample_path = "assets/sample_gel.jpg"
        if os.path.exists(sample_path):
            pil_image = Image.open(sample_path).convert("RGB")
            img_array = np.array(pil_image)
            st.session_state.uploaded_image_array = img_array
            original_gray, clahe_image, _ = preprocess_gel(img_array)
            st.session_state.preprocessed_images = (original_gray, clahe_image)
        else:
            st.error("Demo sample gel image not found in assets/ directory.")
            img_array = None
    elif uploaded_file is not None:
        # ── Read & preprocess the upload ──────────────────────────
        file_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_array = np.array(pil_image)

        # Store for later use
        st.session_state.uploaded_image_array = img_array

        # Run preprocessing for display
        original_gray, clahe_image, _ = preprocess_gel(img_array)
        st.session_state.preprocessed_images = (original_gray, clahe_image)

    if (use_sample and os.path.exists("assets/sample_gel.jpg")) or (uploaded_file is not None):

        # ── Side-by-side image preview ────────────────────────────
        col_orig, col_pre = st.columns(2)

        with col_orig:
            st.markdown(
                '<div class="image-label">Original Gel Image</div>',
                unsafe_allow_html=True,
            )
            st.image(
                pil_image,
                use_container_width=True,
                caption=f"{uploaded_file.name if uploaded_file else 'sample_gel.jpg'} — {pil_image.size[0]}×{pil_image.size[1]} px",
            )

        with col_pre:
            st.markdown(
                '<div class="image-label">Preprocessed (CLAHE Enhanced)</div>',
                unsafe_allow_html=True,
            )
            st.image(
                clahe_image,
                use_container_width=True,
                caption="Contrast-enhanced grayscale",
            )

        # ── Analyze Button ────────────────────────────────────────
        st.markdown("")  # spacer
        analyze_clicked = st.button(
            "🔬  Analyze Gel",
            use_container_width=True,
            type="primary",
        )

        if analyze_clicked:
            progress_bar = st.progress(0, text="Initializing pipeline…")
            status_container = st.status("🧬 Running GelVision AI Pipeline…", expanded=True)

            def progress_callback(fraction: float, message: str) -> None:
                """Update the Streamlit progress bar and status text."""
                progress_bar.progress(
                    min(fraction, 1.0),
                    text=message,
                )
                status_container.update(label=message)

            try:
                t_start = time.time()

                with status_container:
                    pipeline = load_pipeline()

                    st.write("Starting analysis…")
                    ladder_lane_num = ladder_override if ladder_override > 0 else None

                    results = pipeline.run(
                        image_input=img_array,
                        conf_threshold=conf_threshold,
                        band_threshold=band_threshold,
                        min_band_distance=min_band_distance,
                        ladder_lane_number=ladder_lane_num,
                        ladder_type=ladder_type,
                        progress_callback=progress_callback,
                    )

                elapsed = time.time() - t_start

                # Persist results in session state
                st.session_state.results = results
                st.session_state.processing_time = elapsed
                st.session_state.analysis_complete = True

                progress_bar.progress(1.0, text="✅ Analysis complete!")
                status_container.update(
                    label="✅ Analysis complete!", state="complete"
                )

            except FileNotFoundError as exc:
                st.error(f"**Model file not found:** {exc}")
                st.info("Ensure `models/best.pt` and `models/unet_best.pth` exist.")
                st.session_state.analysis_complete = False

            except Exception as exc:
                st.error(f"**Pipeline error:** {exc}")
                st.warning(
                    "Try adjusting the confidence threshold or uploading "
                    "a different image."
                )
                st.session_state.analysis_complete = False

        # ── Display Results (persists across reruns) ──────────────
        if st.session_state.analysis_complete and st.session_state.results is not None:
            results: Dict[str, Any] = st.session_state.results
            elapsed: float = st.session_state.processing_time
            lanes = results["lanes"]
            df = results["results_df"]
            r2 = results["calibration_r2"]

            total_bands = sum(len(lane["bands"]) for lane in lanes)
            avg_conf = (
                np.mean([l["confidence"] for l in lanes]) if lanes else 0.0
            )

            # ── Summary Metric Cards ─────────────────────────────
            st.markdown("---")
            mc1, mc2, mc3, mc4 = st.columns(4)

            with mc1:
                st.markdown(
                    _build_metric_card(
                        str(len(lanes)),
                        "Lanes Detected",
                        f"Avg conf: {avg_conf:.1%}",
                    ),
                    unsafe_allow_html=True,
                )
            with mc2:
                st.markdown(
                    _build_metric_card(
                        str(total_bands),
                        "Bands Found",
                        f"Across {len(lanes)} lanes",
                    ),
                    unsafe_allow_html=True,
                )
            with mc3:
                badge = _r2_quality_badge(r2)
                st.markdown(
                    _build_metric_card(
                        f"{r2:.4f}",
                        "Calibration R²",
                        badge,
                    ),
                    unsafe_allow_html=True,
                )
            with mc4:
                st.markdown(
                    _build_metric_card(
                        f"{elapsed:.1f}s",
                        "Processing Time",
                        get_device_label(),
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("")  # spacer

            # ── Annotated Gel Image ──────────────────────────────
            st.markdown(
                '<div class="image-label">Annotated Gel — Detected Lanes &amp; Bands</div>',
                unsafe_allow_html=True,
            )
            annotated_rgb = _bgr_to_rgb(results["img_annotated"])
            st.image(
                annotated_rgb,
                use_container_width=True,
                caption=f"YOLOv8 lane detection — {len(lanes)} lanes identified",
            )

            # ── Quality Warnings ─────────────────────────────────
            if r2 < 0.80:
                st.warning(
                    "⚠️ **Low calibration R²** — Size estimates may be "
                    "inaccurate. Consider adjusting the ladder lane or "
                    "using a different ladder type."
                )
            elif r2 < 0.95:
                st.info(
                    "ℹ️ Calibration R² is acceptable but below 0.95. "
                    "Size estimates should be treated as approximate."
                )

            low_conf_lanes = [l for l in lanes if l["confidence"] < 0.5]
            if low_conf_lanes:
                lane_nums = ", ".join(str(l["number"]) for l in low_conf_lanes)
                st.warning(
                    f"⚠️ Low confidence detection on lane(s): **{lane_nums}**. "
                    "Results for these lanes may be unreliable."
                )

            if not lanes:
                st.error(
                    "❌ **No lanes detected.** Try lowering the confidence "
                    "threshold or uploading a clearer gel image."
                )
            elif total_bands == 0:
                st.warning(
                    "⚠️ **No bands detected** in any lane. Try lowering the "
                    "band detection threshold."
                )

            # ── Per-Lane Detail Expanders ─────────────────────────
            st.markdown("")
            st.markdown("### 🔎 Lane Details")

            for lane in lanes:
                n_bands = len(lane["bands"])
                is_ladder = lane["number"] == results.get("ladder_lane_number")
                lane_label = "Ladder" if is_ladder else lane["label"]
                icon = "📏" if is_ladder else "🧪"

                with st.expander(
                    f"{icon} Lane {lane['number']} — {lane_label}  "
                    f"({n_bands} band{'s' if n_bands != 1 else ''})",
                    expanded=False,
                ):
                    lc_img, lc_mask = st.columns(2)

                    with lc_img:
                        st.markdown(
                            '<div class="image-label">Lane Crop</div>',
                            unsafe_allow_html=True,
                        )
                        st.image(
                            lane["crop"],
                            use_container_width=True,
                            caption=f"512×512 normalized — conf {lane['confidence']:.1%}",
                        )

                    with lc_mask:
                        st.markdown(
                            '<div class="image-label">Segmentation Mask</div>',
                            unsafe_allow_html=True,
                        )
                        colored_mask = _colorize_mask(lane["mask"])
                        st.image(
                            colored_mask,
                            use_container_width=True,
                            caption="U-Net band segmentation (Inferno colormap)",
                        )

                    # Band table for this lane
                    if lane["bands"]:
                        band_data = []
                        for b in lane["bands"]:
                            row = {
                                "Band #": b["band_number"],
                                "Position (y)": f"{b['centroid_y']:.1f}",
                                "Intensity": f"{b['intensity']:.2f}",
                                "Profile Height": f"{b['profile_height']:.3f}",
                            }
                            if "estimated_size_bp" in b:
                                row["Est. Size (bp)"] = f"{int(b['estimated_size_bp']):,}"
                            band_data.append(row)

                        st.dataframe(
                            pd.DataFrame(band_data),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No bands detected in this lane.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — RESULTS                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab_results:
    st.markdown(
        '<div class="hero-header">Results &amp; Export</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.analysis_complete or st.session_state.results is None:
        st.info(
            "🧬 Run an analysis in the **Analysis** tab first to view results here."
        )
    else:
        results = st.session_state.results
        df: pd.DataFrame = results["results_df"]
        r2: float = results["calibration_r2"]
        slope, intercept = results["calibration_params"]

        # ── Results DataFrame ─────────────────────────────────────
        st.markdown("### 📋 Band Quantitation Table")

        if df.empty:
            st.warning(
                "No sample bands to display. Only ladder bands were detected."
            )
        else:
            st.dataframe(
                df.style.format(
                    {
                        "Lane_Confidence": "{:.1%}",
                        "Position": "{:.1f}",
                        "Estimated_Size_bp": "{:,.0f}",
                        "Intensity": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=min(len(df) * 40 + 60, 500),
            )

        # ── Download Section ──────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💾 Export Reports")

        dl_csv, dl_xlsx, dl_pdf = st.columns(3)

        with dl_csv:
            csv_bytes = generate_csv(df) if not df.empty else b""
            st.download_button(
                label="📄 Download CSV",
                data=csv_bytes,
                file_name="gelvision_results.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=df.empty,
            )

        with dl_xlsx:
            xlsx_bytes = generate_excel(df) if not df.empty else b""
            st.download_button(
                label="📊 Download Excel",
                data=xlsx_bytes,
                file_name="gelvision_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                disabled=df.empty,
            )

        with dl_pdf:
            if not df.empty:
                # Save annotated image to a temp file for the PDF generator
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp_img:
                    annotated_rgb = _bgr_to_rgb(results["img_annotated"])
                    Image.fromarray(annotated_rgb).save(tmp_img.name)
                    pdf_bytes = generate_pdf(df, tmp_img.name, r2)
            else:
                pdf_bytes = b""

            st.download_button(
                label="📕 Download PDF",
                data=pdf_bytes,
                file_name="gelvision_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                disabled=df.empty,
            )

        # ── Calibration Details ───────────────────────────────────
        st.markdown("---")
        st.markdown("### 📐 Calibration Details")

        cal_1, cal_2, cal_3 = st.columns(3)

        with cal_1:
            badge = _r2_quality_badge(r2)
            st.markdown(
                _build_metric_card(f"{r2:.4f}", "R² Score", badge),
                unsafe_allow_html=True,
            )
        with cal_2:
            st.markdown(
                _build_metric_card(f"{slope:.6f}", "Slope", "ln(bp) / pixel"),
                unsafe_allow_html=True,
            )
        with cal_3:
            st.markdown(
                _build_metric_card(f"{intercept:.4f}", "Intercept", "ln(bp)"),
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="info-card" style="margin-top: 1rem;">
                <h4>Calibration Model</h4>
                <p>
                    <strong>log-linear regression:</strong>
                    ln(size<sub>bp</sub>) = {slope:.6f} · y + {intercept:.4f}<br>
                    Ladder lane: <strong>
                    {results.get('ladder_lane_number', 'N/A')}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Summary Statistics ────────────────────────────────────
        if not df.empty:
            st.markdown("---")
            st.markdown("### 📊 Summary Statistics")

            ss1, ss2, ss3, ss4 = st.columns(4)

            with ss1:
                st.markdown(
                    _build_metric_card(
                        str(df["Lane"].nunique()),
                        "Sample Lanes",
                    ),
                    unsafe_allow_html=True,
                )
            with ss2:
                st.markdown(
                    _build_metric_card(
                        str(len(df)),
                        "Total Bands",
                    ),
                    unsafe_allow_html=True,
                )
            with ss3:
                if "Estimated_Size_bp" in df.columns and len(df) > 0:
                    size_range = f"{int(df['Estimated_Size_bp'].min()):,}–{int(df['Estimated_Size_bp'].max()):,}"
                else:
                    size_range = "N/A"
                st.markdown(
                    _build_metric_card(size_range, "Size Range (bp)"),
                    unsafe_allow_html=True,
                )
            with ss4:
                avg_intensity = df["Intensity"].mean() if "Intensity" in df.columns else 0
                st.markdown(
                    _build_metric_card(
                        f"{avg_intensity:.1f}",
                        "Avg Intensity",
                    ),
                    unsafe_allow_html=True,
                )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — ABOUT                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab_about:
    st.markdown(
        '<div class="hero-header">About GelVision AI</div>',
        unsafe_allow_html=True,
    )

    # ── Project Description ───────────────────────────────────────
    st.markdown(
        """
        <div class="info-card">
            <h4>🧬 Project Overview</h4>
            <p>
                <strong>GelVision AI</strong> is an end-to-end deep learning
                platform for automated gel electrophoresis image analysis. It
                combines object detection and semantic segmentation to
                identify lanes, detect DNA/protein bands, and estimate
                molecular weights through log-linear calibration against
                standard ladders.
            </p>
            <p>
                Designed for molecular biology researchers, the tool
                eliminates manual, error-prone gel annotation and delivers
                quantitative, reproducible results in seconds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Pipeline Diagram ──────────────────────────────────────────
    st.markdown("### 🔄 Analysis Pipeline")

    pipe_cols = st.columns([1, 0.3, 1, 0.3, 1, 0.3, 1, 0.3, 1])

    steps = [
        ("📤", "Upload", "Gel Image"),
        ("🔧", "Preprocess", "CLAHE + Denoise"),
        ("🎯", "Detect", "YOLOv8 Lanes"),
        ("🧩", "Segment", "U-Net Bands"),
        ("📐", "Calibrate", "Size Estimation"),
    ]

    for i, (icon, title, desc) in enumerate(steps):
        with pipe_cols[i * 2]:
            st.markdown(
                f"""
                <div class="pipeline-step">
                    <div class="step-icon">{icon}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if i < len(steps) - 1:
            with pipe_cols[i * 2 + 1]:
                st.markdown(
                    '<div class="pipeline-arrow">→</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("")

    # ── Model Architecture ────────────────────────────────────────
    about_left, about_right = st.columns(2)

    with about_left:
        st.markdown(
            """
            <div class="info-card">
                <h4>🎯 Lane Detection — YOLOv8n</h4>
                <p>
                    A YOLOv8-nano model fine-tuned for gel lane detection.
                    Trained to distinguish between sample lanes and ladder
                    lanes with high precision.
                </p>
                <ul>
                    <li><strong>Architecture:</strong> YOLOv8n (3.2M params)</li>
                    <li><strong>mAP50:</strong> 93.1%</li>
                    <li><strong>mAP50-95:</strong> 72.4%</li>
                    <li><strong>Classes:</strong> Lane, Ladder</li>
                    <li><strong>Input:</strong> 640×640 px</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with about_right:
        st.markdown(
            """
            <div class="info-card">
                <h4>🧩 Band Segmentation — U-Net</h4>
                <p>
                    A U-Net with a pretrained ResNet34 encoder for pixel-level
                    band segmentation within each detected lane crop.
                </p>
                <ul>
                    <li><strong>Encoder:</strong> ResNet34 (21.8M params)</li>
                    <li><strong>Dice Score:</strong> 0.165</li>
                    <li><strong>Input:</strong> 512×512 px (grayscale)</li>
                    <li><strong>Output:</strong> Binary mask</li>
                    <li><strong>Activation:</strong> Sigmoid</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Dataset Information ───────────────────────────────────────
    st.markdown(
        """
        <div class="info-card">
            <h4>📚 Dataset — GelGenie</h4>
            <p>
                Models were trained on the
                <strong>GelGenie</strong> dataset — a curated collection of
                <strong>575 annotated gel electrophoresis images</strong>
                spanning agarose and polyacrylamide gels, DNA and protein
                applications, and diverse imaging conditions (UV
                transillumination, blue light, white light, fluorescent
                staining).
            </p>
            <ul>
                <li>575 gel images with pixel-level band annotations</li>
                <li>Multiple gel types and staining protocols</li>
                <li>Lane-level bounding boxes for detection training</li>
                <li>80/10/10 train/validation/test split</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Performance Metrics ───────────────────────────────────────
    st.markdown("### ⚡ Performance")

    perf_cols = st.columns(4)
    perf_data = [
        ("93.1%", "mAP50", "Lane Detection"),
        ("0.165", "Dice Score", "Band Segmentation"),
        ("< 5s", "Inference", "CPU (i7-class)"),
        ("575", "Training Images", "GelGenie Dataset"),
    ]

    for col, (val, label, delta) in zip(perf_cols, perf_data):
        with col:
            st.markdown(
                _build_metric_card(val, label, delta),
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Technology Stack ──────────────────────────────────────────
    st.markdown("### 🛠️ Technology Stack")

    tech_items = [
        "Python 3.10+",
        "PyTorch 2.x",
        "Ultralytics YOLOv8",
        "Segmentation Models PyTorch",
        "OpenCV",
        "Streamlit",
        "NumPy",
        "Pandas",
        "SciPy",
        "scikit-image",
        "ReportLab",
        "Pillow",
    ]

    badges_html = " ".join(
        f'<span class="tech-badge">{t}</span>' for t in tech_items
    )
    st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown("")

    # ── Acknowledgments ───────────────────────────────────────────
    st.markdown(
        """
        <div class="info-card" style="margin-top: 1.5rem;">
            <h4>🙏 Acknowledgments</h4>
            <p>
                Built as a capstone project demonstrating the application of
                modern computer vision techniques to molecular biology
                workflows. Special thanks to the GelGenie dataset authors
                for providing high-quality annotated gel images, and to the
                open-source communities behind PyTorch, Ultralytics, and
                Streamlit.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
