"""
Intelligent Anomaly Detection & AES Data Security — Industrial AI Platform
=============================================================================

IEEE research project dashboard for an Industrial IoT anomaly detection
system. The dashboard collects five sensor readings, sends them through
an Autoencoder-based anomaly detector (see ``src/predict.py``), AES-256
encrypts the raw readings for transport/storage, and visualizes the
result alongside prediction history, dashboard statistics, and the
model's offline evaluation graphs.

SCOPE / NON-GOALS
-----------------
This file only concerns itself with presentation, session-state
management, caching, and orchestration. The prediction logic, AES
encryption, threshold calculation, and preprocessing all live in
``src/predict.py`` and are treated as a black box via the single call
to ``predict_anomaly(...)``. Its call signature, argument order, and
return contract are UNCHANGED from the original project. Session-state
keys, model loading, and the encryption pipeline are likewise untouched
— everything in this file is presentation, layered on top of that
existing, working backend.

BUGFIX NOTES (this revision)
-----------------------------
One additional bug fixed on top of the previously-fixed live-trend
accumulation (session_state deques) and risk/health clamp-vs-severity
split (see PredictionResult docstrings below):

3. Deprecation warning box appearing under every chart, in place of a
   clean caption ("The keyword arguments have been deprecated ...
   Use config instead to specify Plotly configuration options."):
   Every ``st.plotly_chart(...)`` call in this file was still passing
   ``use_container_width=True``, while every other widget in this same
   file (``st.dataframe``, ``st.button``, ``st.download_button``,
   ``st.image``) had already been migrated to the newer ``width=`` API.
   The installed Streamlit version deprecated ``use_container_width``
   specifically on ``st.plotly_chart`` and renders a warning box in
   place of the chart's normal caption area instead of silently
   accepting it. Fix: every ``st.plotly_chart(...)`` call below now
   passes ``width="stretch"`` instead of ``use_container_width=True``,
   matching the rest of the file. Nothing about chart content, data,
   colors, sizing, or layout changed — only that one argument name, in
   every call site (gauges, live trend charts, machine-health gauge,
   analytics charts).
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# -------------------------------------------------------------------------
# Path setup — allow "from src.predict import predict_anomaly" to resolve
# regardless of the working directory Streamlit is launched from.
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_anomaly  # noqa: E402  (import after path fix)

# =========================================================================
# Constants
# =========================================================================
RESULTS_DIR = "results"

MODEL_PERFORMANCE_GRAPHS = (
    ("📈 Training Loss", os.path.join(RESULTS_DIR, "training_loss.png"),
     "Autoencoder training loss across epochs."),
    ("📊 Reconstruction Error Histogram",
     os.path.join(RESULTS_DIR, "reconstruction_error_histogram.png"),
     "Distribution of reconstruction error on the evaluation set."),
    ("📉 Reconstruction Error Line Graph",
     os.path.join(RESULTS_DIR, "reconstruction_error_line.png"),
     "Reconstruction error across evaluation samples, in sequence."),
)

FEATURE_DISTRIBUTION_GRAPHS = (
    ("Air Temperature", os.path.join(RESULTS_DIR, "Air_Temp.png"), "🌡️"),
    ("Process Temperature", os.path.join(RESULTS_DIR, "Process_Temp.png"), "🔥"),
    ("Rotational Speed", os.path.join(RESULTS_DIR, "RPM.png"), "⚙️"),
    ("Torque", os.path.join(RESULTS_DIR, "Torque.png"), "🔧"),
    ("Tool Wear", os.path.join(RESULTS_DIR, "Tool_Wear.png"), "⏱️"),
)

HISTORY_COLUMNS = (
    "Timestamp",
    "Status",
    "Reconstruction Error",
    "Threshold",
    "Latency (ms)",
)

GRAPH_CACHE_TTL_SECONDS = 60  # results/ is static during a demo session
LIVE_WINDOW = 100             # points kept in the rolling live-trend charts

SENSOR_SERIES_KEYS = ("air_temp", "process_temp", "rotational_speed", "torque", "tool_wear")
DERIVED_SERIES_KEYS = ("error", "risk", "health")
LIVE_SERIES_KEYS = SENSOR_SERIES_KEYS + DERIVED_SERIES_KEYS

SENSOR_CONFIG = {
    "air_temp": {
        "label": "Air Temperature", "icon": "🌡️", "unit": "K",
        "desc": "Ambient air temperature at the machine.",
        "normal_range": (295.0, 305.0),
    },
    "process_temp": {
        "label": "Process Temperature", "icon": "🔥", "unit": "K",
        "desc": "Internal process temperature.",
        "normal_range": (305.0, 315.0),
    },
    "rotational_speed": {
        "label": "Rotational Speed", "icon": "⚙️", "unit": "rpm",
        "desc": "Spindle rotational speed.",
        "normal_range": (1300.0, 1700.0),
    },
    "torque": {
        "label": "Torque", "icon": "🔧", "unit": "Nm",
        "desc": "Measured torque.",
        "normal_range": (30.0, 50.0),
    },
    "tool_wear": {
        "label": "Tool Wear", "icon": "⏱️", "unit": "min",
        "desc": "Cumulative tool wear.",
        "normal_range": (0.0, 200.0),
    },
}

ACCENT = "#6C5CE7"
ACCENT_2 = "#00D9FF"
GREEN = "#2ecc71"
RED = "#ff4d5e"
AMBER = "#f5b942"


# =========================================================================
# Numeric safety helpers
# =========================================================================
def to_native_float(value: object) -> float:
    return float(value)


def safe_progress_value(value: object) -> float:
    native_value = to_native_float(value)
    return max(0.0, min(native_value, 1.0))


def safe_series_value(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        native_value = to_native_float(value)
    except (TypeError, ValueError):
        return None
    if native_value != native_value:  # NaN check
        return None
    if native_value in (float("inf"), float("-inf")):
        return None
    return native_value


# =========================================================================
# Data model for a single prediction result
# =========================================================================
@dataclass
class PredictionResult:
    status: str
    error: float
    threshold: float
    encrypted_data: object
    iv: object
    decrypted_data: object
    latency_ms: float
    sensor_values: dict = field(default_factory=dict)

    @property
    def risk_pct(self) -> float:
        if self.threshold == 0:
            return 100.0
        ratio = to_native_float(self.error) / to_native_float(self.threshold)
        return round(max(0.0, min(ratio, 1.0)) * 100, 2)

    @property
    def health_pct(self) -> float:
        return round(max(0.0, 100.0 - self.risk_pct), 2)

    @property
    def severity_pct(self) -> float:
        if self.threshold == 0:
            return 100.0
        ratio = to_native_float(self.error) / to_native_float(self.threshold)
        return round(max(0.0, ratio) * 100, 1)

    @property
    def confidence_pct(self) -> float:
        if self.threshold == 0:
            return 100.0
        ratio = to_native_float(self.error) / to_native_float(self.threshold)
        distance = abs(ratio - 1.0)
        return round(max(50.0, min(50.0 + distance * 50.0, 99.9)), 1)

    def to_history_row(self) -> dict:
        return {
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Status": self.status,
            "Reconstruction Error": round(to_native_float(self.error), 6),
            "Threshold": round(to_native_float(self.threshold), 6),
            "Latency (ms)": round(to_native_float(self.latency_ms), 2),
        }


# =========================================================================
# Session state
# =========================================================================
def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "last_error_message" not in st.session_state:
        st.session_state.last_error_message = None

    _ensure_live_series()


def _ensure_live_series() -> None:
    if "live_series" not in st.session_state or not isinstance(st.session_state.live_series, dict):
        st.session_state.live_series = {}
    for key in LIVE_SERIES_KEYS:
        if key not in st.session_state.live_series:
            st.session_state.live_series[key] = deque(maxlen=LIVE_WINDOW)


def history_dataframe() -> pd.DataFrame:
    if not st.session_state.history:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.DataFrame(st.session_state.history)


# =========================================================================
# Cached, read-only filesystem helpers
# =========================================================================
@st.cache_data(ttl=GRAPH_CACHE_TTL_SECONDS, show_spinner=False)
def _path_exists(path: str) -> bool:
    return os.path.exists(path)


@st.cache_data(ttl=GRAPH_CACHE_TTL_SECONDS, show_spinner=False)
def _read_image_bytes(path: str, _mtime: float) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _load_image_safely(path: str) -> Optional[bytes]:
    if not _path_exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
        return _read_image_bytes(path, mtime)
    except OSError as exc:
        st.warning(f"⚠️ Could not read `{path}`: {exc}")
        return None


# =========================================================================
# Styling — dark glassmorphism theme
# =========================================================================
def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 15% 0%, rgba(108,92,231,0.16), transparent 45%),
                radial-gradient(circle at 85% 15%, rgba(0,217,255,0.12), transparent 45%),
                linear-gradient(180deg, #0a0e17 0%, #0d1220 55%, #0a0e17 100%);
        }}

        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }}

        .hero {{
            position: relative;
            padding: 2.1rem 2.4rem;
            border-radius: 20px;
            margin-bottom: 1.4rem;
            overflow: hidden;
            background: linear-gradient(135deg, rgba(108,92,231,0.20), rgba(0,217,255,0.10));
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
        }}
        .hero-title {{
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(90deg, #ffffff 10%, {ACCENT_2} 55%, {ACCENT} 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            letter-spacing: -0.02em;
        }}
        .hero-sub {{
            color: rgba(255,255,255,0.65);
            font-size: 0.98rem;
            margin-top: 0.35rem;
            max-width: 760px;
        }}
        .badge-row {{ margin-top: 1rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }}
        .badge {{
            display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.78rem; font-weight: 600;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            color: rgba(255,255,255,0.85);
            backdrop-filter: blur(6px);
        }}
        .dot {{
            width: 8px; height: 8px; border-radius: 50%; display: inline-block;
            background: {GREEN};
            box-shadow: 0 0 8px {GREEN};
            animation: pulse-dot 1.6s infinite;
        }}
        @keyframes pulse-dot {{
            0% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.55; transform: scale(1.35); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}

        .glass {{
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 18px rgba(0,0,0,0.25);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }}
        .glass:hover {{
            transform: translateY(-3px);
            border-color: rgba(108,92,231,0.45);
            box-shadow: 0 10px 30px rgba(108,92,231,0.18);
        }}

        .kpi {{
            background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015));
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 16px;
            padding: 0.95rem 1.1rem;
            text-align: left;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}
        .kpi:hover {{ transform: translateY(-2px); box-shadow: 0 8px 22px rgba(0,0,0,0.3); }}
        .kpi-label {{ font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase;
                      color: rgba(255,255,255,0.5); font-weight: 600; }}
        .kpi-value {{ font-size: 1.55rem; font-weight: 800; margin-top: 0.15rem; color: #fff; }}
        .kpi-icon {{ font-size: 1.3rem; opacity: 0.9; }}
        .kpi-sub {{ font-size: 0.74rem; color: rgba(255,255,255,0.45); margin-top: 0.1rem; }}

        .sensor-card {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 0.7rem 0.9rem 0.2rem 0.9rem;
            margin-bottom: 0.4rem;
        }}
        .sensor-head {{ display: flex; align-items: center; gap: 0.5rem; font-weight: 700; color: #fff; }}
        .sensor-desc {{ font-size: 0.76rem; color: rgba(255,255,255,0.5); margin: 0.15rem 0 0.4rem 0; }}
        .sensor-range {{ font-size: 0.7rem; color: {ACCENT_2}; font-family: 'JetBrains Mono', monospace; }}

        .result-card {{
            border-radius: 20px;
            padding: 1.6rem 1.8rem;
            margin: 0.6rem 0 1rem 0;
            border: 1px solid rgba(255,255,255,0.1);
            position: relative;
            overflow: hidden;
        }}
        .result-card.normal {{
            background: linear-gradient(135deg, rgba(46,204,113,0.16), rgba(46,204,113,0.03));
            box-shadow: 0 0 40px rgba(46,204,113,0.18), inset 0 1px 0 rgba(255,255,255,0.06);
            animation: glow-green 2.4s ease-in-out infinite;
        }}
        .result-card.anomaly {{
            background: linear-gradient(135deg, rgba(255,77,94,0.18), rgba(255,77,94,0.04));
            box-shadow: 0 0 40px rgba(255,77,94,0.22), inset 0 1px 0 rgba(255,255,255,0.06);
            animation: glow-red 1.4s ease-in-out infinite;
        }}
        @keyframes glow-green {{
            0%, 100% {{ box-shadow: 0 0 26px rgba(46,204,113,0.14); }}
            50% {{ box-shadow: 0 0 46px rgba(46,204,113,0.30); }}
        }}
        @keyframes glow-red {{
            0%, 100% {{ box-shadow: 0 0 26px rgba(255,77,94,0.18); }}
            50% {{ box-shadow: 0 0 52px rgba(255,77,94,0.38); }}
        }}
        .result-title {{ font-size: 1.5rem; font-weight: 800; margin: 0; color: #fff; }}
        .result-caption {{ color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-top: 0.3rem;
                            font-family: 'JetBrains Mono', monospace; }}

        .pipeline {{ display: flex; align-items: center; justify-content: space-between;
                     gap: 0.3rem; flex-wrap: wrap; margin: 0.6rem 0 1rem 0; }}
        .pipe-step {{
            flex: 1; min-width: 96px; text-align: center;
            padding: 0.65rem 0.4rem;
            border-radius: 12px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            font-size: 0.72rem; font-weight: 600; color: rgba(255,255,255,0.55);
            transition: all 0.25s ease;
        }}
        .pipe-step.active {{
            background: linear-gradient(135deg, rgba(108,92,231,0.35), rgba(0,217,255,0.2));
            border-color: {ACCENT_2};
            color: #fff;
            box-shadow: 0 0 18px rgba(0,217,255,0.35);
            transform: scale(1.06);
        }}
        .pipe-step.done {{
            background: rgba(46,204,113,0.14);
            border-color: rgba(46,204,113,0.5);
            color: #d7ffe6;
        }}
        .pipe-arrow {{ color: rgba(255,255,255,0.25); font-size: 1rem; }}

        .pill {{ display: inline-block; padding: 0.18rem 0.6rem; border-radius: 999px;
                 font-size: 0.72rem; font-weight: 700; letter-spacing: 0.02em; }}
        .pill.normal {{ background: rgba(46,204,113,0.16); color: #6ee7a0; border: 1px solid rgba(46,204,113,0.4); }}
        .pill.anomaly {{ background: rgba(255,77,94,0.16); color: #ff9aa4; border: 1px solid rgba(255,77,94,0.4); }}

        h1, h2, h3 {{ color: #f2f2f7 !important; }}
        .section-title {{
            font-size: 1.1rem; font-weight: 700; color: #fff; margin: 1.1rem 0 0.5rem 0;
            display: flex; align-items: center; gap: 0.5rem;
        }}

        .footer {{
            margin-top: 2.2rem; padding: 1.2rem 1.4rem; border-radius: 16px;
            background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
            text-align: center; color: rgba(255,255,255,0.45); font-size: 0.8rem;
        }}
        .footer b {{ color: rgba(255,255,255,0.75); }}

        .glass, .kpi, .result-card, .hero {{ animation: fadeIn 0.45s ease; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0b0f1a 0%, #0d1424 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# Small chart / gauge builders (Plotly)
# =========================================================================
def _dark_layout(fig: go.Figure, height: int = 220, title: str | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.75)", family="Inter"),
        margin=dict(l=10, r=10, t=36 if title else 10, b=10),
        height=height,
        title=dict(text=title, font=dict(size=13)) if title else None,
    )
    return fig


def make_gauge(value: float, title: str, max_value: float = 100.0,
                suffix: str = "%", good_high: bool = True) -> go.Figure:
    value = max(0.0, min(to_native_float(value), max_value))
    ratio = value / max_value if max_value else 0
    if good_high:
        color = GREEN if ratio > 0.7 else (AMBER if ratio > 0.4 else RED)
    else:
        color = GREEN if ratio < 0.4 else (AMBER if ratio < 0.7 else RED)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 26, "color": "#fff"}},
        gauge={
            "axis": {"range": [0, max_value], "tickcolor": "rgba(255,255,255,0.3)"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, max_value * 0.4], "color": "rgba(255,255,255,0.03)"},
                {"range": [max_value * 0.4, max_value * 0.7], "color": "rgba(255,255,255,0.05)"},
                {"range": [max_value * 0.7, max_value], "color": "rgba(255,255,255,0.07)"},
            ],
        },
    ))
    return _dark_layout(fig, height=200, title=title)


def make_live_line(series: dict, keys: list[tuple[str, str, str]], title: str) -> go.Figure:
    fig = go.Figure()
    for key, name, color in keys:
        y = list(series.get(key, []))
        x = list(range(len(y)))
        fig.add_trace(go.Scatter(
            y=y, x=x, mode="lines+markers",
            name=name, line=dict(color=color, width=2, shape="spline"),
            marker=dict(color=color, size=5),
            fill="tozeroy", fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba")
            if color.startswith("rgb") else None,
        ))
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_layout(legend=dict(orientation="h", y=1.18, font=dict(size=10)))
    return _dark_layout(fig, height=210, title=title)


def render_plotly_chart(fig: go.Figure, width: str = "stretch",
                        key: str | None = None, config: dict | None = None) -> None:
    chart_config = {"displayModeBar": False}
    if config:
        chart_config.update(config)

    if key is not None:
        st.plotly_chart(fig, width=width, config=chart_config, key=key)
    else:
        st.plotly_chart(fig, width=width, config=chart_config)


# =========================================================================
# Hero header
# =========================================================================
def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">🤖 Industrial AI Monitoring System</div>
            <div class="hero-sub">
                Autoencoder-based anomaly detection for Industrial IoT sensor
                streams, secured end-to-end with AES-256 encryption and
                evaluated against an adaptive reconstruction-error threshold.
            </div>
            <div class="badge-row">
                <span class="badge"><span class="dot"></span> System Online</span>
                <span class="badge">🧠 Autoencoder AI</span>
                <span class="badge">🔐 AES-256 Secured</span>
                <span class="badge">📡 Real-Time Machine Health</span>
                <span class="badge">🏭 AI4I 2020 Dataset</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# Sidebar
# =========================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.6rem 0 1rem 0;">
                <div style="font-size:2.2rem;">🛰️</div>
                <div style="font-weight:800; font-size:1.05rem; color:#fff;">
                    Industrial AI
                </div>
                <div style="font-size:0.72rem; color:rgba(255,255,255,0.45); letter-spacing:0.05em;">
                    ANOMALY DETECTION PLATFORM
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**📋 Project Details**")
        st.markdown(
            """
            <div style="font-size:0.82rem; line-height:1.9; color:rgba(255,255,255,0.75);">
            🧠 <b>Model:</b> Autoencoder<br>
            🔐 <b>Encryption:</b> AES-256<br>
            📊 <b>Dataset:</b> AI4I 2020<br>
            🎓 <b>Type:</b> IEEE Research Prototype
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        result: Optional[PredictionResult] = st.session_state.latest_result
        total_predictions = len(st.session_state.history)

        st.markdown("**⚙️ System Status**")
        st.metric("Total Predictions (session)", total_predictions)

        if result is not None:
            icon = "🟢" if result.status == "Normal" else "🔴"
            st.metric("Latest Status", f"{icon} {result.status}")
            st.metric("Current Health", f"{result.health_pct:.1f} %")
        else:
            st.info("No predictions yet this session.")

        st.divider()

        if st.button("🗑️ Clear Prediction History", width="stretch"):
            st.session_state.history = []
            st.session_state.latest_result = None
            for dq in st.session_state.live_series.values():
                dq.clear()
            st.toast("Prediction history cleared.", icon="🗑️")
            st.rerun()

        st.caption(
            "Clearing history only resets this dashboard's session view — "
            "it does not affect the trained model, threshold, or encryption."
        )


# =========================================================================
# Top KPI strip
# =========================================================================
def _kpi_card(icon: str, label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="kpi">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def render_kpi_strip() -> None:
    result: Optional[PredictionResult] = st.session_state.latest_result
    df = history_dataframe()
    total = len(df)
    anomalies = int((df["Status"] == "Anomaly").sum()) if total else 0
    normal = total - anomalies

    cols = st.columns(5)
    with cols[0]:
        st.markdown(_kpi_card("💚", "Machine Health",
                               f"{result.health_pct:.1f}%" if result else "—",
                               "Latest reading"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(_kpi_card("🎯", "Prediction",
                               result.status if result else "—",
                               "Autoencoder output"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(_kpi_card("⚠️", "Risk Level",
                               f"{result.risk_pct:.1f}%" if result else "—",
                               "Error vs threshold"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(_kpi_card("📶", "Confidence",
                               f"{result.confidence_pct:.1f}%" if result else "—",
                               "Distance from threshold"), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(_kpi_card("⚡", "Latency",
                               f"{result.latency_ms:.1f} ms" if result else "—",
                               "Inference + AES"), unsafe_allow_html=True)

    cols2 = st.columns(5)
    with cols2[0]:
        st.markdown(_kpi_card("🧮", "Threshold",
                               f"{result.threshold:.4f}" if result else "—",
                               "Adaptive threshold"), unsafe_allow_html=True)
    with cols2[1]:
        st.markdown(_kpi_card("📐", "Recon. Error",
                               f"{result.error:.4f}" if result else "—",
                               "Latest sample"), unsafe_allow_html=True)
    with cols2[2]:
        st.markdown(_kpi_card("🔐", "AES Status",
                               "Active" if result else "Idle",
                               "256-bit encryption"), unsafe_allow_html=True)
    with cols2[3]:
        st.markdown(_kpi_card("📊", "Total / Normal",
                               f"{total} / {normal}",
                               "This session"), unsafe_allow_html=True)
    with cols2[4]:
        st.markdown(_kpi_card("🚨", "Anomalies",
                               f"{anomalies}",
                               "This session"), unsafe_allow_html=True)


# =========================================================================
# Sensor input cards
# =========================================================================
def render_sensor_inputs() -> dict:
    st.markdown('<div class="section-title">🧾 Live Sensor Input Parameters</div>',
                unsafe_allow_html=True)

    widget_specs = {
        "air_temp": dict(min_value=250.0, max_value=350.0, value=298.0, step=0.1),
        "process_temp": dict(min_value=250.0, max_value=400.0, value=308.0, step=0.1),
        "rotational_speed": dict(min_value=0.0, max_value=5000.0, value=1500.0, step=1.0),
        "torque": dict(min_value=0.0, max_value=200.0, value=40.0, step=0.1),
        "tool_wear": dict(min_value=0.0, max_value=300.0, value=0.0, step=1.0),
    }

    cols = st.columns(5)
    values = {}
    for col, key in zip(cols, widget_specs.keys()):
        cfg = SENSOR_CONFIG[key]
        spec = widget_specs[key]
        with col:
            st.markdown(
                f"""
                <div class="sensor-card">
                    <div class="sensor-head">{cfg['icon']} {cfg['label']}</div>
                    <div class="sensor-desc">{cfg['desc']}</div>
                    <div class="sensor-range">Typical: {cfg['normal_range'][0]:.0f}–{cfg['normal_range'][1]:.0f} {cfg['unit']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            values[key] = float(st.number_input(
                f"{cfg['unit']}", key=f"input_{key}",
                help=f"{cfg['desc']} Units: {cfg['unit']}.",
                **spec,
            ))

    return {
        "air_temp": values["air_temp"],
        "process_temp": values["process_temp"],
        "rotational_speed": values["rotational_speed"],
        "torque": values["torque"],
        "tool_wear": values["tool_wear"],
    }


# =========================================================================
# Animated AI pipeline
# =========================================================================
PIPELINE_STAGES = [
    "Sensors", "Scaling", "AES-256", "Autoencoder",
    "Latent Space", "Decoder", "Recon. Error", "Threshold", "Prediction",
]


def _render_pipeline_html(active_index: int) -> str:
    steps_html = []
    for i, stage in enumerate(PIPELINE_STAGES):
        if i < active_index:
            cls = "done"
        elif i == active_index:
            cls = "active"
        else:
            cls = ""
        steps_html.append(f'<div class="pipe-step {cls}">{stage}</div>')
        if i != len(PIPELINE_STAGES) - 1:
            steps_html.append('<div class="pipe-arrow">➜</div>')
    return f'<div class="pipeline">{"".join(steps_html)}</div>'


def render_pipeline_static() -> None:
    st.markdown('<div class="section-title">🔗 AI Detection Pipeline</div>', unsafe_allow_html=True)
    st.markdown(_render_pipeline_html(active_index=-1), unsafe_allow_html=True)


# =========================================================================
# Core detection call
# =========================================================================
def run_detection(sensor_values: dict) -> Optional[PredictionResult]:
    pipeline_placeholder = st.empty()
    try:
        for idx in range(3):
            pipeline_placeholder.markdown(_render_pipeline_html(idx), unsafe_allow_html=True)
            time.sleep(0.08)

        start_time = time.perf_counter()
        pipeline_placeholder.markdown(_render_pipeline_html(3), unsafe_allow_html=True)

        (
            status,
            error,
            encrypted_data,
            iv,
            decrypted_data,
            threshold,
        ) = predict_anomaly(
            sensor_values["air_temp"],
            sensor_values["process_temp"],
            sensor_values["rotational_speed"],
            sensor_values["torque"],
            sensor_values["tool_wear"],
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        for idx in range(4, len(PIPELINE_STAGES)):
            pipeline_placeholder.markdown(_render_pipeline_html(idx), unsafe_allow_html=True)
            time.sleep(0.06)
        pipeline_placeholder.markdown(_render_pipeline_html(len(PIPELINE_STAGES)), unsafe_allow_html=True)

    except FileNotFoundError as exc:
        pipeline_placeholder.empty()
        st.error(
            "🚫 A required file (model, scaler, or threshold) could not be "
            f"found. Please verify your `src/predict.py` artifacts exist.\n\n"
            f"Details: {exc}"
        )
        return None
    except ValueError as exc:
        pipeline_placeholder.empty()
        st.error(
            "🚫 The prediction pipeline rejected the submitted sensor "
            f"values.\n\nDetails: {exc}"
        )
        return None
    except Exception as exc:  # noqa: BLE001
        pipeline_placeholder.empty()
        st.error(
            "🚫 An unexpected error occurred while running detection or "
            f"AES encryption. The dashboard was NOT crashed.\n\nDetails: {exc}"
        )
        return None

    return PredictionResult(
        status=str(status),
        error=to_native_float(error),
        threshold=to_native_float(threshold),
        encrypted_data=encrypted_data,
        iv=iv,
        decrypted_data=decrypted_data,
        latency_ms=to_native_float(elapsed_ms),
        sensor_values=sensor_values,
    )


# =========================================================================
# UI sections — Prediction result
# =========================================================================
def render_result_card(result: PredictionResult) -> None:
    css_class = "normal" if result.status == "Normal" else "anomaly"
    icon = "✅" if result.status == "Normal" else "🚨"
    label = "MACHINE CONDITION: NORMAL" if result.status == "Normal" \
        else "ANOMALY DETECTED"

    severity_note = (
        f" &nbsp;•&nbsp; Severity {result.severity_pct:.0f}% of threshold"
        if result.status != "Normal" else ""
    )

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="result-title">{icon} {label}</div>
            <div class="result-caption">
                Reconstruction error {result.error:.6f} vs. threshold {result.threshold:.6f}
                &nbsp;•&nbsp; Health {result.health_pct:.1f}% &nbsp;•&nbsp;
                Confidence {result.confidence_pct:.1f}% &nbsp;•&nbsp;
                Latency {result.latency_ms:.2f} ms{severity_note}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.status == "Normal":
        st.success(
            "🟢 **Recommendation:** No action required. Continue routine "
            "monitoring — readings are within the learned normal envelope."
        )
    else:
        deviant_key, deviant_dev = None, -1.0
        for key, cfg in SENSOR_CONFIG.items():
            lo, hi = cfg["normal_range"]
            v = result.sensor_values.get(key)
            if v is None:
                continue
            mid = (lo + hi) / 2
            span = max(hi - lo, 1e-6)
            dev = abs(v - mid) / span
            if dev > deviant_dev:
                deviant_dev, deviant_key = dev, key

        with st.container(border=True):
            st.markdown("**🧭 Indicative Root Cause & Recovery**")
            if deviant_key is not None:
                cfg = SENSOR_CONFIG[deviant_key]
                st.write(
                    f"{cfg['icon']} **{cfg['label']}** shows the largest deviation "
                    f"from its typical operating band "
                    f"({cfg['normal_range'][0]:.0f}–{cfg['normal_range'][1]:.0f} {cfg['unit']}) "
                    f"among the submitted readings."
                )
            st.write(
                "**Suggested next steps:** flag the unit for inspection, "
                "cross-check the highlighted sensor against manual readings, "
                "and review recent maintenance/tool-wear logs before resuming "
                "full-speed operation."
            )
            st.caption(
                "Heuristic diagnostic derived from sensor deviation vs. typical "
                "operating bands — not an explanation generated by the "
                "autoencoder model itself."
            )


def render_gauges_row(result: PredictionResult) -> None:
    st.markdown('<div class="section-title">🕹️ Live Gauges</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_plotly_chart(make_gauge(result.health_pct, "Machine Health", good_high=True),
                            key="gauge_result_health")
    with c2:
        render_plotly_chart(make_gauge(result.risk_pct, "Risk Level", good_high=False),
                            key="gauge_result_risk")
    with c3:
        render_plotly_chart(make_gauge(result.confidence_pct, "Confidence", good_high=True),
                            key="gauge_result_confidence")
    with c4:
        threshold_usage = result.risk_pct
        render_plotly_chart(make_gauge(threshold_usage, "Threshold Usage", good_high=False),
                            key="gauge_result_threshold_usage")


def render_prediction_details(result: PredictionResult) -> None:
    with st.expander("📊 Prediction Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", result.status)
        with col2:
            st.metric("Reconstruction Error", f"{result.error:.6f}")
        with col3:
            st.metric("Processing Time", f"{result.latency_ms:.2f} ms")

        st.divider()
        st.markdown("**📥 Sensor Values Submitted**")
        st.write(result.sensor_values)


def render_encryption_details(result: PredictionResult) -> None:
    with st.expander("🔐 AES-256 Encryption Details", expanded=False):
        tab_enc, tab_iv, tab_dec = st.tabs(
            ["Encrypted Payload", "Initialization Vector", "Decrypted Payload"]
        )
        with tab_enc:
            st.code(result.encrypted_data)
        with tab_iv:
            st.code(result.iv)
        with tab_dec:
            st.code(result.decrypted_data)


def render_result_bundle(result: PredictionResult) -> None:
    render_result_card(result)
    render_gauges_row(result)
    render_prediction_details(result)
    render_encryption_details(result)


# =========================================================================
# History & live-series bookkeeping
# =========================================================================
def save_to_history(result: PredictionResult) -> None:
    st.session_state.history.append(result.to_history_row())
    st.session_state.latest_result = result

    _ensure_live_series()
    series = st.session_state.live_series

    for key in SENSOR_SERIES_KEYS:
        safe_value = safe_series_value(result.sensor_values.get(key))
        if safe_value is not None:
            series[key].append(safe_value)

    for key, value in (("error", result.error), ("risk", result.risk_pct), ("health", result.health_pct)):
        safe_value = safe_series_value(value)
        if safe_value is not None:
            series[key].append(safe_value)


# =========================================================================
# Live trend charts (Plotly) — rolling window of last LIVE_WINDOW points
# =========================================================================
def render_live_trend_charts() -> None:
    _ensure_live_series()
    series = st.session_state.live_series

    with st.expander("🐞 Debug: session-state counters", expanded=False):
        st.write({
            "history_len": len(st.session_state.history),
            "latest_result_present": st.session_state.latest_result is not None,
            **{f"deque_len[{k}]": len(v) for k, v in series.items()},
        })

    if not series["error"]:
        st.info("No live data yet — run a detection to start the trend charts.")
        return

    st.markdown('<div class="section-title">📡 Live Sensor & Health Trends</div>',
                unsafe_allow_html=True)
    st.caption(f"Rolling window of the last {LIVE_WINDOW} predictions this session.")

    c1, c2 = st.columns(2)
    with c1:
        render_plotly_chart(
            make_live_line(series, [
                ("air_temp", "Air Temp (K)", "#00D9FF"),
                ("process_temp", "Process Temp (K)", "#6C5CE7"),
            ], "🌡️ Temperature Trend"),
            key="live_trend_temperature",
        )
    with c2:
        render_plotly_chart(
            make_live_line(series, [
                ("rotational_speed", "RPM", "#f5b942"),
            ], "⚙️ Rotational Speed Trend"),
            key="live_trend_rpm",
        )

    c3, c4 = st.columns(2)
    with c3:
        render_plotly_chart(
            make_live_line(series, [
                ("torque", "Torque (Nm)", "#00D9FF"),
                ("tool_wear", "Tool Wear (min)", "#6C5CE7"),
            ], "🔧 Torque & Tool Wear Trend"),
            key="live_trend_torque_wear",
        )
    with c4:
        render_plotly_chart(
            make_live_line(series, [
                ("error", "Recon. Error", "#ff4d5e"),
            ], "📐 Reconstruction Error Trend"),
            key="live_trend_error",
        )

    c5, c6 = st.columns(2)
    with c5:
        render_plotly_chart(
            make_live_line(series, [
                ("risk", "Risk %", "#ff4d5e"),
            ], "⚠️ Risk Trend"),
            key="live_trend_risk",
        )
    with c6:
        render_plotly_chart(
            make_live_line(series, [
                ("health", "Health %", "#2ecc71"),
            ], "💚 Machine Health Trend"),
            key="live_trend_health",
        )


# =========================================================================
# UI sections — History table & stats
# =========================================================================
def render_history_section(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">📜 Prediction History</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No predictions made yet. Run a detection to populate history.")
        return

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search = st.text_input("🔎 Search timestamp", placeholder="e.g. 2026-08-08")
    with fc2:
        status_filter = st.selectbox("Filter status", ["All", "Normal", "Anomaly"])
    with fc3:
        newest_first = st.checkbox("Newest first", value=True)

    filtered = df.copy()
    if search:
        filtered = filtered[filtered["Timestamp"].str.contains(search, case=False, na=False)]
    if status_filter != "All":
        filtered = filtered[filtered["Status"] == status_filter]
    if newest_first:
        filtered = filtered.iloc[::-1]

    st.dataframe(
        filtered,
        width="stretch",
        column_config={
            "Status": st.column_config.TextColumn("Status", help="🟢 Normal or 🔴 Anomaly"),
            "Reconstruction Error": st.column_config.NumberColumn(format="%.6f"),
            "Threshold": st.column_config.NumberColumn(format="%.6f"),
            "Latency (ms)": st.column_config.NumberColumn(format="%.2f"),
        },
        hide_index=True,
    )
    st.caption(f"Showing {len(filtered)} of {len(df)} records. Click any column header to sort.")

    csv_bytes = df.to_csv(index=False).encode()
    file_name = f"prediction_history_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
    st.download_button(
        "⬇️ Download Prediction History (CSV)",
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
        width="stretch",
    )


def render_dashboard_statistics(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">📊 Dashboard Statistics</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No predictions made yet.")
        return

    total_predictions = len(df)
    anomaly_count = int((df["Status"] == "Anomaly").sum())
    normal_count = total_predictions - anomaly_count
    avg_latency = float(df["Latency (ms)"].mean())
    avg_error = float(df["Reconstruction Error"].mean())
    normal_rate = (normal_count / total_predictions) * 100

    cols = st.columns(6)
    stats = [
        ("🔢", "Total Predictions", str(total_predictions)),
        ("🟢", "Normal", str(normal_count)),
        ("🔴", "Anomalies", str(anomaly_count)),
        ("📐", "Avg. Error", f"{avg_error:.6f}"),
        ("⚡", "Avg. Latency", f"{avg_latency:.2f} ms"),
        ("✅", "Normal Rate", f"{normal_rate:.1f}%"),
    ]
    for col, (icon, label, value) in zip(cols, stats):
        with col:
            st.markdown(_kpi_card(icon, label, value), unsafe_allow_html=True)


def render_machine_health_dashboard() -> None:
    result: Optional[PredictionResult] = st.session_state.latest_result
    if result is None:
        return

    st.markdown('<div class="section-title">🖥️ Machine Health Dashboard (Latest Reading)</div>',
                unsafe_allow_html=True)
    render_plotly_chart(make_gauge(result.health_pct, "Machine Health", good_high=True),
                        key="gauge_dashboard_health")

    if result.health_pct > 80:
        st.success("🟢 Excellent Condition")
    elif result.health_pct > 50:
        st.warning("🟡 Moderate Condition")
    else:
        st.error("🔴 Critical Condition")


# =========================================================================
# Analytics tab
# =========================================================================
def render_analytics_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No predictions made yet. Run a detection to populate analytics.")
        return

    st.markdown('<div class="section-title">📊 Prediction Distribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        counts = df["Status"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        fig = px.pie(
            counts, names="Status", values="Count", hole=0.55,
            color="Status", color_discrete_map={"Normal": GREEN, "Anomaly": RED},
        )
        render_plotly_chart(_dark_layout(fig, height=300, title="Normal vs. Anomaly"),
                            key="analytics_pie_distribution")
    with c2:
        fig = px.bar(
            df.reset_index(), x="index", y="Latency (ms)",
            color="Status", color_discrete_map={"Normal": GREEN, "Anomaly": RED},
        )
        fig.update_xaxes(title="Prediction #")
        render_plotly_chart(_dark_layout(fig, height=300, title="Latency per Prediction"),
                            key="analytics_bar_latency")

    st.markdown('<div class="section-title">📈 Reconstruction Error Over Time</div>',
                unsafe_allow_html=True)
    fig = px.area(
        df.reset_index(), x="index", y="Reconstruction Error",
    )
    fig.add_hline(y=float(df["Threshold"].iloc[-1]), line_dash="dash",
                  line_color=AMBER, annotation_text="Latest Threshold")
    fig.update_traces(line_color=ACCENT_2, fillcolor="rgba(0,217,255,0.12)")
    fig.update_xaxes(title="Prediction #")
    render_plotly_chart(_dark_layout(fig, height=320), key="analytics_area_error")

    st.markdown('<div class="section-title">🕒 Prediction Timeline</div>', unsafe_allow_html=True)
    timeline_df = df.copy().reset_index()
    fig = px.scatter(
        timeline_df, x="Timestamp", y="Reconstruction Error",
        color="Status", size="Latency (ms)",
        color_discrete_map={"Normal": GREEN, "Anomaly": RED},
    )
    render_plotly_chart(_dark_layout(fig, height=320), key="analytics_scatter_timeline")


# =========================================================================
# UI sections — Offline evaluation graphs
# =========================================================================
def render_model_performance_graphs() -> None:
    st.markdown('<div class="section-title">📊 Model Performance Graphs</div>', unsafe_allow_html=True)

    any_found = False
    for title, path, caption in MODEL_PERFORMANCE_GRAPHS:
        image_bytes = _load_image_safely(path)
        if image_bytes is None:
            st.warning(f"⚠️ {title} not found (expected at `{path}`).")
            continue

        any_found = True
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.image(image_bytes, width="stretch")
            st.caption(caption)
            with st.expander("🔍 View Fullscreen"):
                st.image(image_bytes, width="stretch")

    if not any_found:
        st.info(
            "No model performance graphs are currently available in the "
            f"`{RESULTS_DIR}/` directory."
        )


def render_feature_distribution_graphs() -> None:
    st.markdown('<div class="section-title">📂 Feature Distribution</div>', unsafe_allow_html=True)

    available: list[tuple[str, str, bytes]] = []
    for title, path, icon in FEATURE_DISTRIBUTION_GRAPHS:
        image_bytes = _load_image_safely(path)
        if image_bytes is None:
            st.warning(f"⚠️ {title} distribution graph not found (expected at `{path}`).")
            continue
        available.append((title, icon, image_bytes))

    if not available:
        st.info(
            "No feature distribution graphs are currently available in the "
            f"`{RESULTS_DIR}/` directory."
        )
        return

    columns_per_row = 2
    for i in range(0, len(available), columns_per_row):
        row_items = available[i:i + columns_per_row]
        cols = st.columns(len(row_items))
        for col, (title, icon, image_bytes) in zip(cols, row_items):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{icon} {title}**")
                    st.image(image_bytes, width="stretch")
                    st.caption(f"Distribution of {title.lower()} across the dataset.")


# =========================================================================
# Footer
# =========================================================================
def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            <b>Industrial AI Monitoring Dashboard</b> — IEEE Research Prototype<br>
            TensorFlow Autoencoder &nbsp;•&nbsp; AES-256 Encryption &nbsp;•&nbsp;
            Industrial IoT &nbsp;•&nbsp; Federated Learning<br><br>
            Developed by <b>Rahul MJ</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# Main app
# =========================================================================
def render_prediction_tab() -> None:
    render_kpi_strip()
    st.divider()
    sensor_values = render_sensor_inputs()
    render_pipeline_static()

    detect_clicked = st.button("🚀 Detect Anomaly", type="primary", width="stretch")

    if detect_clicked:
        result = run_detection(sensor_values)
        if result is not None:
            save_to_history(result)
            st.divider()
            render_result_bundle(result)

    elif st.session_state.latest_result is not None:
        st.info("Showing the most recent detection result below.")
        render_result_bundle(st.session_state.latest_result)


def render_live_dashboard_tab() -> None:
    render_live_trend_charts()
    st.divider()
    render_machine_health_dashboard()


def render_history_tab() -> None:
    df = history_dataframe()
    render_dashboard_statistics(df)
    st.divider()
    render_history_section(df)


def main() -> None:
    st.set_page_config(
        page_title="Industrial AI Monitoring System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    init_session_state()
    render_sidebar()
    render_hero()

    tab_predict, tab_live, tab_history, tab_analytics, tab_model, tab_features = st.tabs(
        [
            "🔍 Live Prediction",
            "📡 Live Dashboard",
            "📜 History",
            "📊 Analytics",
            "🧠 Model Performance",
            "📂 Feature Distribution",
        ]
    )

    with tab_predict:
        render_prediction_tab()

    with tab_live:
        render_live_dashboard_tab()

    with tab_history:
        render_history_tab()

    with tab_analytics:
        render_analytics_tab(history_dataframe())

    with tab_model:
        render_model_performance_graphs()

    with tab_features:
        render_feature_distribution_graphs()

    st.divider()
    render_footer()


if __name__ == "__main__":
    main()