import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Fault Early-Warning Tester", layout="wide")

# ----------------------------------------------------------------------------
# 1. Load artifacts
# ----------------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(ARTIFACT_DIR / "models/scaler_v3.pkl")
    model_2 = joblib.load(ARTIFACT_DIR / "models/xgboost_f2_v3.pkl")
    model_1 = joblib.load(ARTIFACT_DIR / "models/xgboost_baseline_v3.pkl")
    features = joblib.load(ARTIFACT_DIR / "models/features_v3.pkl")
    best_thresh = joblib.load(ARTIFACT_DIR / "models/best_thresh_v3.pkl")
    horizon_sec = joblib.load(ARTIFACT_DIR / "models/horizon_sec_v3.pkl")
    return scaler, model_1, model_2, features, best_thresh, horizon_sec

try:
    scaler, model_1, model_2, FEATURES, DEFAULT_THRESH, HORIZON_SEC = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Missing model artifact: "
        f"{e}\n\nMake sure scaler.pkl, xgboost_baseline.pkl, xgboost_f2.pkl, "
        "features.pkl, best_thresh.pkl and horizon_sec.pkl are in the same "
        "folder as app.py (run the last cell of the notebook to generate them)."
    )
    st.stop()


# ----------------------------------------------------------------------------
# 2. Preprocessing pipeline (mirrors the training notebook exactly)
# ----------------------------------------------------------------------------
def preprocess_flight(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduces the notebook's preprocessing for a single uploaded CSV
    containing one or more flights (identified via 'bag')."""
    data = raw_df.copy()
    data = data.sort_values(by=["bag", "timestamp_ns"]).reset_index(drop=True)

    # flight_id
    gap_ns_threshold = 1000 * 1e9
    data["time_gap_ns"] = data.groupby("bag", sort=False)["timestamp_ns"].diff()
    bag_changed = data["bag"] != data["bag"].shift(1)
    large_gap = data["time_gap_ns"] > gap_ns_threshold
    data["flight_id"] = (bag_changed | large_gap).cumsum() - 1
    data.drop(columns=["time_gap_ns"], inplace=True)

    # flight_time_sec
    data["flight_time_sec"] = data.groupby("flight_id", sort=False)["timestamp_ns"].transform(
        lambda x: (x - x.min()) / 1e9
    )

    # fill NA within each flight
    fill_cols = [c for c in [
        "vel_x", "vel_y", "vel_z", "pitch_cmd", "pitch_meas", "pitch_error",
        "yaw_cmd", "yaw_meas", "yaw_error"
    ] if c in data.columns]
    data[fill_cols] = data.groupby("flight_id", sort=False)[fill_cols].ffill().bfill()

    # yaw wrap
    data["yaw_error_clean"] = ((data["yaw_error"] + 180) % 360) - 180

    # ground speed
    data["ground_speed"] = np.sqrt(data["vel_x"] ** 2 + data["vel_y"] ** 2 + data["vel_z"] ** 2)

    return data


FAULT_COLS = ["engine_fault", "aileron_fault", "elevator_fault", "rudder_fault"]


def add_binary_fault(data: pd.DataFrame) -> pd.DataFrame:
    """Exactly mirrors the notebook: binary_fault = 1 if ANY of the raw
    per-actuator fault columns is > 0."""
    data = data.copy()
    present_faults = [c for c in FAULT_COLS if c in data.columns]
    if not present_faults:
        raise KeyError(
            f"None of the expected fault columns {FAULT_COLS} were found in "
            "the uploaded CSV. Check your raw file's column names."
        )
    data["binary_fault"] = (data[present_faults] > 0).any(axis=1).astype(int)
    return data


def compute_fault_onset(data: pd.DataFrame, min_consecutive_frames: int = 5) -> pd.DataFrame:
    """Filters out single-frame startup glitches (e.g., at t=0) by requiring

    binary_fault to stay 1 for at least N consecutive frames.
    """
    data = data.sort_values(["flight_id", "flight_time_sec"]).reset_index(drop=True)
    sustained_fault = (
        data.groupby("flight_id")["binary_fault"]
        .transform(lambda x: x.rolling(window=min_consecutive_frames, min_periods=1).sum())
    ) >= min_consecutive_frames
    data["fault_onset"] = (
        sustained_fault
        & (~sustained_fault.groupby(data["flight_id"]).shift(1).fillna(False))
    ).astype(int)

    return data

def add_future_fault_label(data: pd.DataFrame, horizon_sec: float = 5.0) -> pd.DataFrame:
    data = data.sort_values(["flight_id", "flight_time_sec"]).reset_index(drop=True)
    labels = np.zeros(len(data), dtype=int)

    for flight_id, group in data.groupby("flight_id"):
        idx = group.index.values
        times = group["flight_time_sec"].values
        onset_times = np.sort(times[group["fault_onset"].values == 1])

        if len(onset_times) == 0:
            continue

        lo = np.searchsorted(onset_times, times, side="right")
        hi = np.searchsorted(onset_times, times + horizon_sec, side="right")
        labels[idx] = (hi > lo).astype(int)

    data["fault_within_5s"] = labels
    return data

def predict_flight(flight_df: pd.DataFrame, model, smooth: bool = True):
    X = flight_df[FEATURES].copy()
    X_scaled = scaler.transform(X)
    raw_probs = model.predict_proba(X_scaled)[:, 1]

    if smooth:
        probs = pd.Series(raw_probs).rolling(window=10, min_periods=1).mean().values
    else:
        probs = raw_probs

    return probs


def fit_pre_onset_regression(time_vals: np.ndarray, prob_vals: np.ndarray):
    """Simple linear regression: probability ~ time. Returns slope, intercept, r2, fitted line."""
    if len(time_vals) < 2:
        return None
    A = np.vstack([time_vals, np.ones_like(time_vals)]).T
    slope, intercept = np.linalg.lstsq(A, prob_vals, rcond=None)[0]
    fitted = slope * time_vals + intercept
    ss_res = np.sum((prob_vals - fitted) ** 2)
    ss_tot = np.sum((prob_vals - prob_vals.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2, fitted


# ----------------------------------------------------------------------------
# Live replay helpers (stock-ticker style colored line)
# ----------------------------------------------------------------------------
STATE_COLORS = {"normal": "#1f77b4", "warning": "#f2c94c", "fault": "#e74c3c"}


def classify_states(sub_df: pd.DataFrame, threshold: float) -> list:
    states = []
    for _, row in sub_df.iterrows():
        if row["binary_fault"] == 1:
            states.append("fault")
        elif row["pred_prob"] >= threshold:
            states.append("warning")
        else:
            states.append("normal")
    return states


def build_colored_segments(sub_df: pd.DataFrame, threshold: float):
    """Splits the line into contiguous same-state segments, bridging the
    boundary point into both segments so the drawn line stays continuous."""
    states = classify_states(sub_df, threshold)
    times = sub_df["flight_time_sec"].values
    probs = sub_df["pred_prob"].values

    segments = []
    cur_x, cur_y, cur_state = [], [], None

    for t, p, s in zip(times, probs, states):
        if cur_state is None:
            cur_state = s
            cur_x, cur_y = [t], [p]
        elif s == cur_state:
            cur_x.append(t)
            cur_y.append(p)
        else:
            # bridge point so consecutive segments connect visually
            cur_x.append(t)
            cur_y.append(p)
            segments.append((cur_x, cur_y, STATE_COLORS[cur_state]))
            cur_state = s
            cur_x, cur_y = [t], [p]

    if cur_x:
        segments.append((cur_x, cur_y, STATE_COLORS[cur_state]))

    return segments


def make_live_figure(full_flight_df, segments, threshold, onset_time, horizon_sec, title):
    fig = go.Figure()

    if onset_time is not None and horizon_sec is not None:
        fig.add_vrect(
            x0=max(onset_time - horizon_sec, full_flight_df["flight_time_sec"].min()),
            x1=onset_time,
            fillcolor="rgba(255, 165, 0, 0.15)",
            line_width=0,
            annotation_text=f"{horizon_sec:.0f}s horizon",
            annotation_position="top left",
        )

    for x_seg, y_seg, color in segments:
        fig.add_trace(go.Scatter(
            x=x_seg, y=y_seg, mode="lines",
            line=dict(color=color, width=3),
            showlegend=False,
        ))

    fig.add_hline(y=threshold, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                  annotation_text=f"Threshold ({threshold:.2f})", annotation_position="bottom right")

    if onset_time is not None:
        fig.add_vline(x=onset_time, line_dash="dot", line_color="white",
                      annotation_text="True fault onset", annotation_position="top right")

    fig.update_layout(
        title=title,
        xaxis=dict(title="Flight time (s)", range=[0, full_flight_df["flight_time_sec"].max()]),
        yaxis=dict(title="Predicted probability", range=[0, 1]),
        template="plotly_dark",
        height=450,
        margin=dict(t=60),
    )
    return fig


# ----------------------------------------------------------------------------
# 3. Sidebar controls
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload raw flight CSV (must include 'bag', 'timestamp_ns', sensor columns, "
    "and a fault/binary_fault column)",
    type=["csv"],
)

model_choice = st.sidebar.radio(
    "Model", ["Model 2 (smoothed, F2-tuned)", "Model 1 (baseline)"], index=0
)
active_model = model_2 if model_choice.startswith("Model 2") else model_1
use_smoothing = model_choice.startswith("Model 2")

threshold = st.sidebar.slider(
    "Warning threshold",
    min_value=0.0, max_value=1.0,
    value=float(DEFAULT_THRESH) if use_smoothing else 0.30,
    step=0.01,
)

st.sidebar.caption(f"Prediction horizon: **{HORIZON_SEC:.0f} seconds** ahead of fault onset.")

st.title("🚁 Fault Early-Warning Model — Tester")
st.write(
    "Pick one **normal** flight and one **faulty** flight to see whether the model "
    f"raises a warning before the fault actually happens (target: {HORIZON_SEC:.0f}s lead time)."
)

if uploaded_file is None:
    st.info("Upload a CSV with one or more flights to begin.")
    st.stop()

raw_df = pd.read_csv(uploaded_file)

missing_fault_cols = [c for c in FAULT_COLS if c not in raw_df.columns]
if missing_fault_cols:
    st.error(
        f"Uploaded CSV is missing expected fault columns: {missing_fault_cols}. "
        f"The pipeline expects all of {FAULT_COLS} (from the raw telemetry) to "
        "compute binary_fault, exactly as in the training notebook."
    )
    st.stop()

processed = preprocess_flight(raw_df)
processed = add_binary_fault(processed)
processed = compute_fault_onset(processed)
processed = add_future_fault_label(processed, horizon_sec=HORIZON_SEC) 

# classify each flight as normal / faulty
flight_summary = processed.groupby("flight_id")["binary_fault"].max()
normal_flights = flight_summary[flight_summary == 0].index.tolist()
fault_flights = flight_summary[flight_summary == 1].index.tolist()

col1, col2 = st.columns(2)
with col1:
    if not normal_flights:
        st.warning("No fully normal flight found in this file.")
        st.stop()
    normal_id = st.selectbox("Normal flight", normal_flights, key="normal_flight")
with col2:
    if not fault_flights:
        st.warning("No faulty flight found in this file.")
        st.stop()
    fault_id = st.selectbox("Faulty flight", fault_flights, key="fault_flight")


# ----------------------------------------------------------------------------
# 4. Run predictions
# ----------------------------------------------------------------------------
def analyze_flight(flight_id, label):
    flight_df = processed[processed["flight_id"] == flight_id].sort_values("flight_time_sec").reset_index(drop=True)
    probs = predict_flight(flight_df, active_model, smooth=use_smoothing)
    flight_df = flight_df.copy()
    flight_df["pred_prob"] = probs
    return flight_df


normal_df = analyze_flight(normal_id, "Normal")
fault_df = analyze_flight(fault_id, "Faulty")

onset_rows = fault_df[fault_df["fault_onset"] == 1]
onset_time = onset_rows["flight_time_sec"].iloc[0] if len(onset_rows) > 0 else None

# ----------------------------------------------------------------------------
# 5. Plot
# ----------------------------------------------------------------------------
def make_plot(flight_df, title, onset_time=None, show_regression=False, horizon_sec=None):
    fig = go.Figure()

    if onset_time is not None and horizon_sec is not None:
        fig.add_vrect(
            x0=max(onset_time - horizon_sec, flight_df["flight_time_sec"].min()),
            x1=onset_time,
            fillcolor="rgba(255, 165, 0, 0.15)",
            line_width=0,
            annotation_text=f"{horizon_sec:.0f}s prediction horizon",
            annotation_position="top left",
        )

    fig.add_trace(go.Scatter(
        x=flight_df["flight_time_sec"], y=flight_df["pred_prob"],
        mode="lines", name="Predicted fault probability", line=dict(color="#1f77b4", width=2),
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold ({threshold:.2f})", annotation_position="top left")

    if onset_time is not None:
        fig.add_vline(x=onset_time, line_dash="dot", line_color="black",
                      annotation_text="True fault onset", annotation_position="top right")

        pre_onset = flight_df[flight_df["flight_time_sec"] < onset_time]
        if show_regression and len(pre_onset) >= 2:
            reg = fit_pre_onset_regression(
                pre_onset["flight_time_sec"].values, pre_onset["pred_prob"].values
            )
            if reg is not None:
                slope, intercept, r2, fitted = reg
                fig.add_trace(go.Scatter(
                    x=pre_onset["flight_time_sec"], y=fitted,
                    mode="lines", name=f"Pre-onset trend (slope={slope:.4f}, R²={r2:.2f})",
                    line=dict(color="orange", width=3, dash="dash"),
                ))

    fig.update_layout(
        title=title,
        xaxis_title="Flight time (s)",
        yaxis_title="Predicted probability of fault within horizon",
        yaxis_range=[0, 1],
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


tab1, tab2 = st.tabs(["📊 Comparison", "📋 Raw predictions", ])

with tab1:
    st.plotly_chart(make_plot(normal_df, f"Normal flight (id={normal_id})", horizon_sec=HORIZON_SEC), use_container_width=True)
    st.plotly_chart(
        make_plot(fault_df, f"Faulty flight (id={fault_id})", onset_time=onset_time,
                  show_regression=True, horizon_sec=HORIZON_SEC),
        use_container_width=True,
    )

    # ---- Metrics ----
    st.subheader("📈 Early-warning metrics (faulty flight)")
    warned_rows = fault_df[fault_df["pred_prob"] >= threshold]
    first_warning_time = warned_rows["flight_time_sec"].iloc[0] if len(warned_rows) > 0 else None

    m1, m2, m3 = st.columns(3)
    m1.metric("True fault onset", f"{onset_time:.1f}s" if onset_time is not None else "N/A")

    if first_warning_time is not None and onset_time is not None:
        lead_time = onset_time - first_warning_time
        if first_warning_time <= onset_time:
            m2.metric("First warning", f"{first_warning_time:.1f}s", delta=f"{lead_time:.1f}s before onset")
        else:
            m2.metric("First warning", f"{first_warning_time:.1f}s", delta="after onset (missed)", delta_color="inverse")
    else:
        m2.metric("First warning", "Never crossed threshold")

    pre_onset_fault = fault_df[fault_df["flight_time_sec"] < onset_time] if onset_time is not None else fault_df
    reg = fit_pre_onset_regression(
        pre_onset_fault["flight_time_sec"].values, pre_onset_fault["pred_prob"].values
    ) if len(pre_onset_fault) >= 2 else None
    if reg is not None:
        slope, intercept, r2, _ = reg
        m3.metric("Pre-onset trend slope", f"{slope:.4f} /s", help="Positive slope = rising risk before the fault, as expected.")

    # ---- Did the model actually fire specifically INSIDE the horizon window? ----
    if onset_time is not None:
        horizon_start = onset_time - HORIZON_SEC
        in_horizon = fault_df[
            (fault_df["flight_time_sec"] >= horizon_start) & (fault_df["flight_time_sec"] < onset_time)
        ]
        fired_in_horizon = (in_horizon["pred_prob"] >= threshold).any() if len(in_horizon) > 0 else False
        # any warning BEFORE the horizon window even started (too early / noisy)
        before_horizon = fault_df[fault_df["flight_time_sec"] < horizon_start]
        fired_before_horizon = (before_horizon["pred_prob"] >= threshold).any() if len(before_horizon) > 0 else False


    false_positive_rate_normal = (normal_df["pred_prob"] >= threshold).mean()
    st.caption(f"Normal flight false-alarm rate at this threshold: **{false_positive_rate_normal:.1%}** of frames.")

with tab2:
    st.write("Faulty flight predictions")
    st.dataframe(fault_df[["flight_time_sec", "pred_prob", "binary_fault", "fault_onset", "fault_within_5s"]])
    st.write("Normal flight predictions")
    st.dataframe(fault_df[["flight_time_sec", "pred_prob", "binary_fault", "fault_onset", "fault_within_5s"]])

    csv_fault = fault_df.to_csv(index=False).encode("utf-8")
    csv_normal = normal_df.to_csv(index=False).encode("utf-8")
    c1, c2 = st.columns(2)
    c1.download_button("Download faulty flight predictions", csv_fault, "fault_flight_predictions.csv", "text/csv")
    c2.download_button("Download normal flight predictions", csv_normal, "normal_flight_predictions.csv", "text/csv")
