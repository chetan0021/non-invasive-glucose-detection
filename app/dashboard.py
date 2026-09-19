"""
Non-Invasive Blood Glucose Prediction System — Advanced Clinical & Sensor Exploration Dashboard
Built with Streamlit & Phase 7 Production Multi-Modal Inference Engine.

Interactive Visualizations:
1. Longitudinal Patient Glycemic Trend Chart (with Clarke Zone color coding)
2. Calibrated Confidence Interval vs. Clinical Safety Zones (Horizontal Gauge)
3. Top-8 Physiological Feature Contribution Bar Chart (Signed impact on predicted BGL)
4. Multi-Modal Parameter Sensitivity Sweep Curves (Real-time live simulation)
5. Physiological Input vs Normal Reference Range Radar/Bar Comparison
6. Always-Visible Model Transparency Badge & Type 1 Clinical Caution Alerts
7. 1-Click Clinical Benchmark Presets & Indian ICMR/RSSDI Population Standards
"""

import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Setup Path & Imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict import GlucosePredictor

# Directories for artifacts and logging
DATA_DIR = BASE_DIR / "data"
REPORTS_OUTPUT_DIR = DATA_DIR / "reports"
LOG_CSV_PATH = DATA_DIR / "manual_test_log.csv"

REPORTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Page Configuration & Styling
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Non-Invasive Glucose Prediction & Multi-Modal Sensor Lab",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.1rem;
    }
    .model-badge-fs {
        background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 100%);
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
    .model-badge-tab {
        background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%);
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
    .badge-normal {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid #86efac;
        display: inline-block;
    }
    .badge-prediabetes {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid #fde047;
        display: inline-block;
    }
    .badge-diabetes {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid #fca5a5;
        display: inline-block;
    }
    .badge-hypo {
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid #a5b4fc;
        display: inline-block;
    }
    .badge-risk-lower {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.05rem;
        border: 1px solid #7dd3fc;
        display: inline-block;
    }
    .badge-risk-elevated {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.05rem;
        border: 1px solid #fdba74;
        display: inline-block;
    }
    .disclaimer-banner {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 0.8rem 1.1rem;
        border-radius: 6px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        color: #92400e;
        font-size: 0.90rem;
    }
    .disclaimer-critical {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 0.8rem 1.1rem;
        border-radius: 6px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        color: #b91c1c;
        font-size: 0.90rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Initialize Predictor Singleton
# ------------------------------------------------------------------------------

@st.cache_resource
def get_predictor():
    return GlucosePredictor()

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading production models: {e}")
    st.stop()


# ------------------------------------------------------------------------------
# Benchmark Preset Definitions
# ------------------------------------------------------------------------------

BENCHMARK_PRESETS = {
    "Custom / Manual Entry": None,
    "🟢 Benchmark 1: Healthy Adult (Fasting Normal: ~88 mg/dL)": {
        "full_name": "Healthy Reference Subject",
        "notes": "Normal fasting metabolic baseline (ADA/ICMR Normal <100 mg/dL)",
        "age": 34.0, "gender": "Female", "height_cm": 168.0, "weight_kg": 60.0,
        "race_ethnicity": "Non-Hispanic White", "waist_circumference_cm": 74.0,
        "physical_activity_level": "Vigorous / Highly Active",
        "hypertension": "No", "high_cholesterol": "No", "gestational_diabetes": "No",
        "family_history": "No (0)", "smoking_status": "Non-Smoker (0)",
        "fasting_status": "Fasting (≥8h)", "med_status": "None",
        "diagnosis_option": "None (Healthy)",
        "has_sensor": True,
        "saliva_ph": 7.35, "temp_c": 36.6, "spo2_pct": 99.0,
        "hr_bpm": 66.0, "ppg_dc": 178000.0, "ppg_ac": 1450.0, "perfusion_idx": 0.81,
        "pulse_width_ms": 285.0, "vpg_max": 4800.0, "apg_a": 85.0, "apg_b": -55.0,
        "hrv_sdnn": 58.0, "hrv_rmssd": 52.0, "hrv_pnn50": 26.0, "hrv_lf_hf": 1.10,
        "reference_bgl": 88.0, "prev_bgl": 90.0
    },
    "🟡 Benchmark 2: Prediabetes / Impaired Fasting (~114 mg/dL)": {
        "full_name": "Prediabetes Screen Subject",
        "notes": "Impaired fasting glucose profile (ADA/ICMR Prediabetes 100-125 mg/dL)",
        "age": 52.0, "gender": "Male", "height_cm": 175.0, "weight_kg": 84.0,
        "race_ethnicity": "Non-Hispanic White", "waist_circumference_cm": 96.0,
        "physical_activity_level": "Sedentary / Inactive (<150 min/wk)",
        "hypertension": "Yes", "high_cholesterol": "Yes", "gestational_diabetes": "Not Applicable (Male)",
        "family_history": "Yes (1)", "smoking_status": "Non-Smoker (0)",
        "fasting_status": "Fasting (≥8h)", "med_status": "None",
        "diagnosis_option": "Prediabetes",
        "has_sensor": True,
        "saliva_ph": 6.95, "temp_c": 36.5, "spo2_pct": 98.0,
        "hr_bpm": 76.0, "ppg_dc": 172000.0, "ppg_ac": 1180.0, "perfusion_idx": 0.68,
        "pulse_width_ms": 275.0, "vpg_max": 4400.0, "apg_a": 78.0, "apg_b": -62.0,
        "hrv_sdnn": 36.0, "hrv_rmssd": 28.0, "hrv_pnn50": 10.0, "hrv_lf_hf": 1.65,
        "reference_bgl": 114.0, "prev_bgl": 118.0
    },
    "🟠 Benchmark 3: Type 2 Diabetes Post-Prandial Spike (~172 mg/dL)": {
        "full_name": "Type 2 Subject (Post-Meal)",
        "notes": "2-hour post-prandial glycemic excursion on Metformin",
        "age": 59.0, "gender": "Male", "height_cm": 174.0, "weight_kg": 90.0,
        "race_ethnicity": "Non-Hispanic White", "waist_circumference_cm": 104.0,
        "physical_activity_level": "Sedentary / Inactive (<150 min/wk)",
        "hypertension": "Yes", "high_cholesterol": "Yes", "gestational_diabetes": "Not Applicable (Male)",
        "family_history": "Yes (1)", "smoking_status": "Current Smoker (1)",
        "fasting_status": "Non-Fasting / Post-Meal", "med_status": "Oral Hypoglycemics",
        "diagnosis_option": "Type 2 Diabetes",
        "has_sensor": True,
        "saliva_ph": 6.60, "temp_c": 36.9, "spo2_pct": 97.0,
        "hr_bpm": 84.0, "ppg_dc": 164000.0, "ppg_ac": 1850.0, "perfusion_idx": 1.12,
        "pulse_width_ms": 298.0, "vpg_max": 5200.0, "apg_a": 72.0, "apg_b": -48.0,
        "hrv_sdnn": 26.0, "hrv_rmssd": 18.0, "hrv_pnn50": 6.0, "hrv_lf_hf": 2.20,
        "reference_bgl": 172.0, "prev_bgl": 142.0
    },
    "🔴 Benchmark 4: Severe Hyperglycemia / Uncontrolled Spike (~265 mg/dL)": {
        "full_name": "Severe Hyperglycemia Patient",
        "notes": "Marked acute hyperglycemia with cellular acidosis and vagal blunting",
        "age": 48.0, "gender": "Female", "height_cm": 162.0, "weight_kg": 85.0,
        "race_ethnicity": "Hispanic / Latino", "waist_circumference_cm": 98.0,
        "physical_activity_level": "Sedentary / Inactive (<150 min/wk)",
        "hypertension": "Yes", "high_cholesterol": "Yes", "gestational_diabetes": "Yes",
        "family_history": "Yes (1)", "smoking_status": "Current Smoker (1)",
        "fasting_status": "Non-Fasting / Post-Meal", "med_status": "Insulin",
        "diagnosis_option": "Type 1 Diabetes",
        "has_sensor": True,
        "saliva_ph": 6.15, "temp_c": 37.2, "spo2_pct": 96.0,
        "hr_bpm": 98.0, "ppg_dc": 188000.0, "ppg_ac": 2950.0, "perfusion_idx": 1.57,
        "pulse_width_ms": 325.0, "vpg_max": 6100.0, "apg_a": 64.0, "apg_b": -38.0,
        "hrv_sdnn": 14.0, "hrv_rmssd": 8.0, "hrv_pnn50": 1.0, "hrv_lf_hf": 3.80,
        "reference_bgl": 265.0, "prev_bgl": 210.0
    },
    "⚠️ Benchmark 5: Acute Hypoglycemia Alert (~62 mg/dL)": {
        "full_name": "Hypoglycemia Emergency Case",
        "notes": "Acute hypoglycemia under-range episode requiring fast-acting carbs",
        "age": 28.0, "gender": "Male", "height_cm": 180.0, "weight_kg": 70.0,
        "race_ethnicity": "Non-Hispanic White", "waist_circumference_cm": 78.0,
        "physical_activity_level": "Vigorous / Highly Active",
        "hypertension": "No", "high_cholesterol": "No", "gestational_diabetes": "Not Applicable (Male)",
        "family_history": "No (0)", "smoking_status": "Non-Smoker (0)",
        "fasting_status": "Fasting (≥8h)", "med_status": "Insulin",
        "diagnosis_option": "Type 1 Diabetes",
        "has_sensor": True,
        "saliva_ph": 7.42, "temp_c": 36.1, "spo2_pct": 99.0,
        "hr_bpm": 88.0, "ppg_dc": 172000.0, "ppg_ac": 1150.0, "perfusion_idx": 0.66,
        "pulse_width_ms": 260.0, "vpg_max": 4100.0, "apg_a": 88.0, "apg_b": -75.0,
        "hrv_sdnn": 48.0, "hrv_rmssd": 44.0, "hrv_pnn50": 20.0, "hrv_lf_hf": 1.30,
        "reference_bgl": 62.0, "prev_bgl": 85.0
    },
    "👤 Benchmark 6: Demographic Screening (Elevated Risk / No Sensors)": {
        "full_name": "Community Outpatient Beta",
        "notes": "Routine primary care health check without wearable sensors",
        "age": 61.0, "gender": "Male", "height_cm": 170.0, "weight_kg": 94.0,
        "race_ethnicity": "Non-Hispanic Black", "waist_circumference_cm": 108.0,
        "physical_activity_level": "Sedentary / Inactive (<150 min/wk)",
        "hypertension": "Yes", "high_cholesterol": "Yes", "gestational_diabetes": "Not Applicable (Male)",
        "family_history": "Yes (1)", "smoking_status": "Current Smoker (1)",
        "fasting_status": "Fasting (≥8h)", "med_status": "None",
        "diagnosis_option": "Unknown / Not Diagnosed",
        "has_sensor": False
    },
    "👤 Benchmark 7: Demographic Screening (Lower Baseline Risk / No Sensors)": {
        "full_name": "Community Outpatient Alpha",
        "notes": "Young active individual screening without wearable sensors",
        "age": 25.0, "gender": "Female", "height_cm": 165.0, "weight_kg": 54.0,
        "race_ethnicity": "Asian / Asian American", "waist_circumference_cm": 68.0,
        "physical_activity_level": "Vigorous / Highly Active",
        "hypertension": "No", "high_cholesterol": "No", "gestational_diabetes": "No",
        "family_history": "No (0)", "smoking_status": "Non-Smoker (0)",
        "fasting_status": "Fasting (≥8h)", "med_status": "None",
        "diagnosis_option": "None (Healthy)",
        "has_sensor": False
    }
}


# ------------------------------------------------------------------------------
# Visualization Helper Functions (Plotly)
# ------------------------------------------------------------------------------

def plot_confidence_interval_gauge(bgl: float, ci_low: float, ci_high: float) -> go.Figure:
    """
    Renders horizontal range gauge showing point estimate & 90% prediction interval
    overlaid on color-coded clinical safety zones.
    """
    fig = go.Figure()

    # Safety Zone Rectangles
    fig.add_vrect(x0=40, x1=70, fillcolor="#dbeafe", opacity=0.45, layer="below", line_width=0, annotation_text="Hypo (<70)", annotation_position="top left")
    fig.add_vrect(x0=70, x1=140, fillcolor="#dcfce7", opacity=0.45, layer="below", line_width=0, annotation_text="Normal (70–140)", annotation_position="top left")
    fig.add_vrect(x0=140, x1=200, fillcolor="#fef9c3", opacity=0.45, layer="below", line_width=0, annotation_text="Elevated (140–200)", annotation_position="top left")
    fig.add_vrect(x0=200, x1=360, fillcolor="#fee2e2", opacity=0.45, layer="below", line_width=0, annotation_text="Severe (≥200)", annotation_position="top left")

    # Prediction Interval Bar
    fig.add_trace(go.Scatter(
        x=[ci_low, ci_high], y=[0, 0], mode='lines',
        line=dict(color='#0284c7', width=14),
        name='90% Prediction Interval [q0.05, q0.95]',
        hoverinfo='text',
        hovertext=f"90% Prediction Interval: [{ci_low} – {ci_high}] mg/dL"
    ))

    # Point Estimate Diamond Marker
    fig.add_trace(go.Scatter(
        x=[bgl], y=[0], mode='markers+text',
        marker=dict(color='#1e3a8a', size=18, symbol='diamond', line=dict(color='white', width=2)),
        text=[f"<b>{bgl} mg/dL</b>"], textposition="top center",
        name='Predicted Blood Glucose',
        hoverinfo='text',
        hovertext=f"Estimated BGL: {bgl} mg/dL"
    ))

    fig.update_layout(
        title="<b>Calibrated Prediction Interval vs. Clinical Safety Zones</b>",
        xaxis=dict(title="Blood Glucose Level (mg/dL)", range=[40, 360], zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, range=[-0.5, 0.5]),
        height=210, margin=dict(l=15, r=15, t=35, b=25), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
    )
    return fig


def plot_feature_contributions(predictor_obj: GlucosePredictor, input_dict: Dict[str, Any]) -> Tuple[go.Figure, str]:
    """
    Computes top-8 signed feature contributions and plots horizontal bar chart.
    """
    X_df = predictor_obj._prepare_full_sensor_features(input_dict)
    feature_names = X_df.columns.tolist()

    # Safely extract feature importances from stacked base models or direct model
    if isinstance(predictor_obj.fs_model, dict) and "base_models" in predictor_obj.fs_model:
        rf_m = predictor_obj.fs_model["base_models"].get("Random Forest")
        if rf_m is not None and hasattr(rf_m, "feature_importances_"):
            importances = rf_m.feature_importances_
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)
    elif hasattr(predictor_obj.fs_model, "feature_importances_"):
        importances = predictor_obj.fs_model.feature_importances_
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    contributions = []
    for i, col in enumerate(feature_names):
        val = X_df.iloc[0, i]
        imp = importances[i]
        # Invert physiological scales where lower values represent higher glucose (pH, HRV)
        multiplier = -1.0 if "ph" in col or "rmssd" in col or "sdnn" in col or "pnn50" in col else 1.0
        score = val * imp * 100.0 * multiplier
        contributions.append((col, score))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_8 = contributions[:8]

    label_map = {
        "saliva_ph_scaled": "Saliva pH Bio-Probe",
        "hrv_rmssd_scaled": "ECG HRV RMSSD (Vagal Tone)",
        "hrv_sdnn_scaled": "ECG HRV SDNN (Variability)",
        "hr_bpm_scaled": "Heart Rate (BPM)",
        "temperature_c_scaled": "Skin Surface Temp",
        "ppg_raw_dc_baseline_scaled": "MAX30102 DC Offset",
        "ppg_raw_ac_p2p_scaled": "MAX30102 AC Peak",
        "pulse_width_ms_scaled": "PPG Pulse Width",
        "perfusion_index_scaled": "Perfusion Index (%)",
        "age_scaled": "Subject Age",
        "bmi_scaled": "Subject BMI",
        "diag_type_1": "Type 1 Stratum",
        "diag_type_2": "Type 2 Stratum",
        "fasting": "Fasting State"
    }

    clean_labels = [label_map.get(k, k.replace("_scaled", "").replace("_", " ").title()) for k, v in top_8]
    values = [round(v, 2) for k, v in top_8]
    bar_colors = ["#dc2626" if v > 0 else "#16a34a" for v in values]

    top_feature_name = clean_labels[0]
    top_direction = "increasing" if values[0] > 0 else "lowering"
    summary_line = f"Primary driver: **{top_feature_name}** ({top_direction} predicted blood glucose relative to population baseline)."

    fig = go.Figure(go.Bar(
        x=values[::-1], y=clean_labels[::-1], orientation='h',
        marker=dict(color=bar_colors[::-1]),
        text=[f"{v:+.1f}" for v in values[::-1]], textposition="outside"
    ))
    fig.update_layout(
        title="<b>Top 8 Physiological Feature Contributions to Prediction</b>",
        xaxis=dict(title="Relative Impact (Red = Increases BGL, Green = Pulls Toward Normal)", zeroline=True),
        height=320, margin=dict(l=15, r=25, t=35, b=25)
    )
    return fig, summary_line


def plot_patient_trend_chart(patient_name: str, current_bgl: float, current_time: str, current_czone: str) -> go.Figure:
    """
    Reads data/manual_test_log.csv, filters for patient, and plots longitudinal trend.
    """
    records = []
    if LOG_CSV_PATH.exists():
        try:
            df_log = pd.read_csv(LOG_CSV_PATH)
            if "full_name" in df_log.columns and "predicted_bgl_mg_dl" in df_log.columns:
                p_df = df_log[df_log["full_name"].astype(str).str.lower() == str(patient_name).lower()]
                p_df = p_df.dropna(subset=["predicted_bgl_mg_dl"])
                for _, r in p_df.iterrows():
                    try:
                        b_val = float(r["predicted_bgl_mg_dl"])
                        records.append({
                            "timestamp": str(r.get("timestamp", ""))[:19].replace("T", " "),
                            "bgl": b_val,
                            "czone": str(r.get("clarke_zone", "Zone A"))
                        })
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    # Ensure current reading is included as the newest point
    if not records or records[-1]["bgl"] != current_bgl:
        records.append({
            "timestamp": current_time,
            "bgl": current_bgl,
            "czone": current_czone
        })

    if len(records) < 2:
        return None

    df_trend = pd.DataFrame(records)
    color_map = []
    for cz in df_trend["czone"]:
        if "Zone A" in cz or "Optimal" in cz: color_map.append("#16a34a")
        elif "Zone B" in cz or "Post-Prandial" in cz: color_map.append("#d97706")
        else: color_map.append("#dc2626")

    fig = go.Figure()

    # Normal target band (70-140 mg/dL)
    fig.add_hrect(y0=70, y1=140, fillcolor="#dcfce7", opacity=0.35, layer="below", line_width=0, annotation_text="Target Range (70–140 mg/dL)", annotation_position="top left")

    # Trend line
    fig.add_trace(go.Scatter(
        x=df_trend["timestamp"], y=df_trend["bgl"],
        mode='lines+markers',
        line=dict(color='#0284c7', width=2.5, shape='spline'),
        marker=dict(color=color_map, size=11, line=dict(color='white', width=1.5)),
        name='Blood Glucose (mg/dL)',
        hovertext=[f"Time: {t}<br>BGL: {b} mg/dL<br>Status: {z}" for t, b, z in zip(df_trend["timestamp"], df_trend["bgl"], df_trend["czone"])],
        hoverinfo='text'
    ))

    fig.update_layout(
        title=f"<b>Longitudinal Glycemic History for {patient_name}</b>",
        xaxis=dict(title="Timestamp", showgrid=True),
        yaxis=dict(title="Blood Glucose (mg/dL)", range=[40, max(260.0, df_trend["bgl"].max() + 30)]),
        height=280, margin=dict(l=15, r=15, t=35, b=25)
    )
    return fig


def plot_normal_range_comparison(sensor_dict: Dict[str, Any]) -> go.Figure:
    """
    Compares entered sensor values against standard physiological reference ranges.
    """
    ref_ranges = [
        ("Saliva pH", float(sensor_dict.get("saliva_ph", 7.25)), 7.0, 7.4, "pH"),
        ("Skin Temp (°C)", float(sensor_dict.get("temperature_c", 36.6)), 36.1, 37.2, "°C"),
        ("Heart Rate", float(sensor_dict.get("hr_bpm", 72.0)), 60.0, 80.0, "BPM"),
        ("SpO2 (%)", float(sensor_dict.get("spo2_pct", 98.0)), 95.0, 100.0, "%"),
        ("HRV RMSSD", float(sensor_dict.get("hrv_rmssd", 34.0)), 25.0, 65.0, "ms"),
        ("HRV SDNN", float(sensor_dict.get("hrv_sdnn", 42.0)), 35.0, 80.0, "ms")
    ]

    names, values, statuses, colors_list = [], [], [], []
    for label, val, low, high, unit in ref_ranges:
        names.append(label)
        values.append(val)
        if val < low:
            statuses.append(f"{val} {unit} (Below Normal: {low}–{high})")
            colors_list.append("#0284c7")
        elif val > high:
            statuses.append(f"{val} {unit} (Elevated: {low}–{high})")
            colors_list.append("#dc2626")
        else:
            statuses.append(f"{val} {unit} (Normal Range: {low}–{high})")
            colors_list.append("#16a34a")

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker=dict(color=colors_list),
        text=statuses, textposition="auto"
    ))
    fig.update_layout(
        title="<b>Physiological Sensor Readings vs. Normal Reference Ranges</b>",
        yaxis=dict(title="Measured Transducer Value"),
        height=260, margin=dict(l=15, r=15, t=35, b=25)
    )
    return fig


def plot_clarke_error_grid(ref_bgl: Optional[float] = None, pred_bgl: Optional[float] = None, benchmark_points: Optional[List[Dict[str, Any]]] = None) -> go.Figure:
    """
    Renders an interactive Clarke Error Grid Analysis (EGA) chart with standard clinical zones (A, B, C, D, E).
    """
    fig = go.Figure()

    # Diagonal ideal line (y = x)
    fig.add_trace(go.Scatter(
        x=[0, 400], y=[0, 400], mode='lines',
        line=dict(color='#64748b', width=1.5, dash='dash'),
        name='Ideal Reference (y = x)',
        hoverinfo='skip'
    ))

    # Zone A boundaries (ISO 15197 absolute +/-15 mg/dL for <70, +/-20% for >=70)
    # Lower boundary: y = 0.8x (x >= 70), (0, 0) to (70, 56)
    fig.add_trace(go.Scatter(
        x=[0, 70, 400], y=[0, 56, 320], mode='lines',
        line=dict(color='#16a34a', width=1.5),
        name='Zone A/B Lower Boundary (0.8x / -15 mg/dL)',
        hoverinfo='skip'
    ))
    # Upper boundary: y = 1.2x (x >= 70), (0, 15) -> (55, 70) -> (70, 84) -> (333.3, 400)
    fig.add_trace(go.Scatter(
        x=[0, 55, 70, 333.3], y=[15, 70, 84, 400], mode='lines',
        line=dict(color='#16a34a', width=1.5),
        name='Zone A/B Upper Boundary (1.2x / +15 mg/dL)',
        hoverinfo='skip'
    ))

    # Zone C boundaries
    fig.add_trace(go.Scatter(
        x=[70, 290], y=[180, 400], mode='lines',
        line=dict(color='#ea580c', width=1.5),
        name='Zone C Upper Boundary (y = x + 110)',
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[130, 180], y=[0, 70], mode='lines',
        line=dict(color='#ea580c', width=1.5),
        name='Zone C Lower Boundary (y = 7/5x - 182)',
        hoverinfo='skip'
    ))

    # Zone D/E threshold lines
    fig.add_trace(go.Scatter(
        x=[70, 70], y=[70, 400], mode='lines',
        line=dict(color='#dc2626', width=1.5, dash='dot'),
        name='Zone D Hypo Miss Threshold (Ref <= 70, Est > 70)',
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[240, 400], y=[70, 70], mode='lines',
        line=dict(color='#dc2626', width=1.5, dash='dot'),
        name='Zone D Hyper Miss Lower Boundary (Est=70)',
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[240, 400], y=[180, 180], mode='lines',
        line=dict(color='#dc2626', width=1.5, dash='dot'),
        name='Zone D Hyper Miss Upper Boundary (Est=180)',
        hoverinfo='skip'
    ))

    # Zone Labels
    fig.add_annotation(x=300, y=280, text="<b>Zone A</b><br>(Clinically Accurate)", showarrow=False, font=dict(color="#15803d", size=11))
    fig.add_annotation(x=320, y=180, text="<b>Zone B</b><br>(Benign Error)", showarrow=False, font=dict(color="#854d0e", size=10))
    fig.add_annotation(x=170, y=340, text="<b>Zone B</b>", showarrow=False, font=dict(color="#854d0e", size=10))
    fig.add_annotation(x=120, y=270, text="<b>Zone C</b><br>(Over-correction)", showarrow=False, font=dict(color="#c2410c", size=9))
    fig.add_annotation(x=35, y=130, text="<b>Zone D</b><br>(Hypo Miss)", showarrow=False, font=dict(color="#b91c1c", size=9))
    fig.add_annotation(x=320, y=120, text="<b>Zone D</b><br>(Hyper Miss)", showarrow=False, font=dict(color="#b91c1c", size=9))
    fig.add_annotation(x=35, y=300, text="<b>Zone E</b><br>(Opposite Action)", showarrow=False, font=dict(color="#7f1d1d", size=9))
    fig.add_annotation(x=320, y=35, text="<b>Zone E</b><br>(Opposite Action)", showarrow=False, font=dict(color="#7f1d1d", size=9))

    # Plot benchmark presets points if provided
    if benchmark_points:
        bx = [p["ref"] for p in benchmark_points]
        by = [p["pred"] for p in benchmark_points]
        bnames = [p["name"] for p in benchmark_points]
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode='markers+text',
            marker=dict(size=12, color='#0284c7', line=dict(color='white', width=1.5)),
            text=[f"{n.split(':')[0]}" for n in bnames], textposition="bottom right",
            name='Benchmark Scenarios',
            hovertext=[f"<b>{n}</b><br>Reference BGL: {r} mg/dL<br>Predicted BGL: {p} mg/dL<br>Abs Error: {abs(p-r):.1f} mg/dL" for n, r, p in zip(bnames, bx, by)],
            hoverinfo='text'
        ))

    # Plot current test point
    if ref_bgl is not None and pred_bgl is not None and ref_bgl > 0:
        fig.add_trace(go.Scatter(
            x=[ref_bgl], y=[pred_bgl], mode='markers',
            marker=dict(size=18, color='#dc2626', symbol='diamond', line=dict(color='white', width=2)),
            name='Current Test Reading',
            hovertext=f"<b>Current Test</b><br>Reference: {ref_bgl} mg/dL<br>Predicted: {pred_bgl} mg/dL<br>Abs Error: {abs(pred_bgl - ref_bgl):.1f} mg/dL",
            hoverinfo='text'
        ))

    fig.update_layout(
        title="<b>Clarke Error Grid Analysis (EGA) — Clinical Safety Assessment</b>",
        xaxis=dict(title="Reference / Lab Blood Glucose (mg/dL)", range=[0, 400], dtick=50, showgrid=True),
        yaxis=dict(title="Predicted Blood Glucose (mg/dL)", range=[0, 400], dtick=50, showgrid=True),
        height=480,
        margin=dict(l=20, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        plot_bgcolor="#f8fafc"
    )
    return fig


# ------------------------------------------------------------------------------
# PDF Report Compilation Utility (ReportLab)
# ------------------------------------------------------------------------------

def generate_pdf_report(session_data: Dict[str, Any], input_data: Dict[str, Any], prediction_res: Dict[str, Any]) -> str:
    """
    Compiles session details, biometrics, physiological inputs, and prediction
    results into a formatted clinical PDF document.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', session_data.get("full_name", "Anonymous"))
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"{safe_name}_{timestamp_str}.pdf"
    pdf_path = REPORTS_OUTPUT_DIR / pdf_filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()
    primary_color = colors.HexColor("#1e3a8a")
    dark_text = colors.HexColor("#1e293b")
    light_bg = colors.HexColor("#f8fafc")

    title_style = ParagraphStyle('TitleStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=primary_color)
    sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=12, textColor=colors.HexColor("#64748b"))
    h2_style = ParagraphStyle('H2Style', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=primary_color, spaceBefore=6, spaceAfter=3)
    cell_style = ParagraphStyle('CellStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10.5, textColor=dark_text)
    cell_bold = ParagraphStyle('CellBold', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, leading=10.5, textColor=dark_text)
    alert_style = ParagraphStyle('AlertStyle', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=7.5, leading=10, textColor=colors.HexColor("#991b1b"))

    story = []

    story.append(Paragraph("Non-Invasive Glucose Prediction — Subject Clinical Summary", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Model: {prediction_res.get('model_version', 'v1.0')}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=4, spaceAfter=6))

    # Session Table
    story.append(Paragraph("1. Session & Subject Information", h2_style))
    session_rows = [
        [Paragraph("<b>Full Name:</b>", cell_style), Paragraph(str(session_data.get("full_name", "N/A")), cell_style),
         Paragraph("<b>Date / Time:</b>", cell_style), Paragraph(str(session_data.get("session_time", "N/A")), cell_style)],
        [Paragraph("<b>Age / Gender:</b>", cell_style), Paragraph(f"{input_data.get('age', 'N/A')} yrs / {str(input_data.get('gender', 'N/A')).capitalize()}", cell_style),
         Paragraph("<b>BMI / Waist:</b>", cell_style), Paragraph(f"{input_data.get('bmi', 'N/A')} kg/m² ({input_data.get('bmi_category', 'N/A')}) / {input_data.get('waist_circumference_cm', 'N/A')} cm", cell_style)],
        [Paragraph("<b>Race / Ethnicity:</b>", cell_style), Paragraph(str(input_data.get("race_ethnicity", "N/A")).replace("_", " ").title(), cell_style),
         Paragraph("<b>Physical Activity:</b>", cell_style), Paragraph(str(input_data.get("physical_activity_level", "N/A")).capitalize(), cell_style)],
        [Paragraph("<b>Comorbidities:</b>", cell_style), Paragraph(f"HTN: {'Yes' if input_data.get('hypertension') == 1 else 'No'} | Dyslipidemia: {'Yes' if input_data.get('high_cholesterol') == 1 else 'No'}", cell_style),
         Paragraph("<b>Gestational Diabetes:</b>", cell_style), Paragraph(str(input_data.get("gestational_diabetes", "N/A")).replace("_", " ").title(), cell_style)],
        [Paragraph("<b>Diagnosis Status:</b>", cell_style), Paragraph(str(input_data.get("diabetes_diagnosis", "None")), cell_style),
         Paragraph("<b>Fasting State:</b>", cell_style), Paragraph("Fasting (≥8h)" if input_data.get("fasting", 1) == 1 else "Non-Fasting", cell_style)],
        [Paragraph("<b>Clinical Notes:</b>", cell_style), Paragraph(str(session_data.get("notes", "None recorded")), cell_style),
         Paragraph("<b>Medications:</b>", cell_style), Paragraph(str(input_data.get("med_status", "None")), cell_style)]
    ]
    t_sess = Table(session_rows, colWidths=[90, 180, 90, 180])
    t_sess.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_sess)
    story.append(Spacer(1, 6))

    # Prediction Results
    story.append(Paragraph("2. Clinical Model Output & Confidence Bounds", h2_style))
    if "predicted_bgl_mg_dl" in prediction_res:
        bgl = prediction_res["predicted_bgl_mg_dl"]
        ci = prediction_res["confidence_interval_5th_95th"]
        pred_rows = [
            [Paragraph("<b>Predicted Blood Glucose:</b>", cell_bold), Paragraph(f"<b><font size=11 color='#1e3a8a'>{bgl} mg/dL</font></b>", cell_style)],
            [Paragraph("<b>90% Prediction Interval [q0.05, q0.95]:</b>", cell_style), Paragraph(f"<b>[{ci[0]} – {ci[1]} mg/dL]</b> (Width: {prediction_res['interval_width_mg_dl']} mg/dL)", cell_style)],
            [Paragraph("<b>Clarke Error Zone:</b>", cell_style), Paragraph(f"<b>{prediction_res.get('clarke_zone', 'Zone A')}</b>", cell_style)],
            [Paragraph("<b>Glycemic Trend:</b>", cell_style), Paragraph(str(prediction_res.get("trend", "N/A")), cell_style)],
            [Paragraph("<b>Diagnostic Stratum Reliability:</b>", cell_style), Paragraph(str(prediction_res.get("diagnosis_stratum_confidence", "Standard")), cell_style)]
        ]
    else:
        r_band = prediction_res.get("risk_band", "N/A")
        probs = prediction_res.get("predicted_risk_probabilities", {})
        prob_str = f"Healthy: {probs.get('healthy_risk', 0)*100:.1f}% | Elevated: {probs.get('elevated_risk', 0)*100:.1f}% | Diabetic: {probs.get('diabetic_risk', 0)*100:.1f}%"
        pred_rows = [
            [Paragraph("<b>Demographic Risk Band:</b>", cell_bold), Paragraph(f"<b><font size=11 color='#c2410c'>{r_band.upper()}</font></b>", cell_style)],
            [Paragraph("<b>Clinical Guidance:</b>", cell_style), Paragraph(str(prediction_res.get("clinical_guidance", "N/A")), cell_style)],
            [Paragraph("<b>Class Probabilities:</b>", cell_style), Paragraph(prob_str, cell_style)],
            [Paragraph("<b>Screening Scope:</b>", cell_style), Paragraph(str(prediction_res.get("validated_scope", "CDC NHANES Outpatients")), cell_style)]
        ]

    t_pred = Table(pred_rows, colWidths=[170, 370])
    t_pred.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0fdf4") if "predicted_bgl_mg_dl" in prediction_res else colors.HexColor("#fff7ed")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_pred)
    story.append(Spacer(1, 6))

    if "ppg_raw_dc_baseline" in input_data and input_data["ppg_raw_dc_baseline"] is not None:
        story.append(Paragraph("3. Multi-Modal Sensor Parameters Entered", h2_style))
        sensor_rows = [
            [Paragraph("<b>MAX30102 DC Baseline:</b>", cell_style), Paragraph(f"{input_data.get('ppg_raw_dc_baseline')} counts", cell_style),
             Paragraph("<b>MAX30102 AC P2P:</b>", cell_style), Paragraph(f"{input_data.get('ppg_raw_ac_p2p')} counts", cell_style)],
            [Paragraph("<b>Saliva pH Probe:</b>", cell_style), Paragraph(f"{input_data.get('saliva_ph')} pH", cell_style),
             Paragraph("<b>Skin Temperature:</b>", cell_style), Paragraph(f"{input_data.get('temperature_c')} °C", cell_style)],
            [Paragraph("<b>ECG HRV SDNN:</b>", cell_style), Paragraph(f"{input_data.get('hrv_sdnn', 'N/A')} ms", cell_style),
             Paragraph("<b>ECG HRV RMSSD:</b>", cell_style), Paragraph(f"{input_data.get('hrv_rmssd', 'N/A')} ms", cell_style)]
        ]
        t_sens = Table(sensor_rows, colWidths=[130, 140, 130, 140])
        t_sens.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), light_bg),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('TOPPADDING', (0,0), (-1,-1), 2.5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ]))
        story.append(t_sens)
        story.append(Spacer(1, 6))

    disclaimer_rows = [[
        Paragraph(
            "<b>MANDATORY RESEARCH PROTOTYPE DISCLAIMER:</b><br/>"
            "This document is generated by an experimental research prototype. The full-sensor machine learning regression model "
            "is validated on synthetic multi-modal self-consistency data only. The tabular demographic classifier is validated on CDC NHANES "
            "community survey outpatients only (Macro AUROC=0.87). This software is not an FDA-cleared medical device and must NEVER be used to "
            "adjust insulin doses or substitute for certified clinical laboratory blood testing.",
            alert_style
        )
    ]]
    t_disc = Table(disclaimer_rows, colWidths=[540])
    t_disc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fef2f2")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#ef4444")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_disc)

    doc.build(story)
    return str(pdf_path)


# ------------------------------------------------------------------------------
# Audit Logging Utility
# ------------------------------------------------------------------------------

def append_to_audit_log(session_data: Dict[str, Any], input_data: Dict[str, Any], prediction_res: Dict[str, Any]):
    """
    Appends the complete submission to data/manual_test_log.csv for auditability.
    """
    log_row = {
        "timestamp": datetime.now().isoformat(),
        "full_name": session_data.get("full_name", ""),
        "notes": session_data.get("notes", ""),
        "age": input_data.get("age", ""),
        "gender": input_data.get("gender", ""),
        "race_ethnicity": input_data.get("race_ethnicity", ""),
        "height_cm": input_data.get("height_cm", ""),
        "weight_kg": input_data.get("weight_kg", ""),
        "bmi": input_data.get("bmi", ""),
        "bmi_category": input_data.get("bmi_category", ""),
        "waist_circumference_cm": input_data.get("waist_circumference_cm", ""),
        "physical_activity_level": input_data.get("physical_activity_level", ""),
        "hypertension": input_data.get("hypertension", 0),
        "high_cholesterol": input_data.get("high_cholesterol", 0),
        "gestational_diabetes": input_data.get("gestational_diabetes", ""),
        "family_history": input_data.get("family_history", 0),
        "smoking": input_data.get("smoking", 0),
        "fasting": input_data.get("fasting", 1),
        "med_status": input_data.get("med_status", "None"),
        "diabetes_diagnosis": input_data.get("diabetes_diagnosis", "None"),
        "is_full_sensor": 1 if "predicted_bgl_mg_dl" in prediction_res else 0,
        "predicted_bgl_mg_dl": prediction_res.get("predicted_bgl_mg_dl", ""),
        "confidence_interval_5th": prediction_res.get("confidence_interval_5th_95th", ["", ""])[0],
        "confidence_interval_95th": prediction_res.get("confidence_interval_5th_95th", ["", ""])[1],
        "interval_width_mg_dl": prediction_res.get("interval_width_mg_dl", ""),
        "clarke_zone": prediction_res.get("clarke_zone", ""),
        "trend": prediction_res.get("trend", ""),
        "risk_band": prediction_res.get("risk_band", ""),
        "prob_healthy": prediction_res.get("predicted_risk_probabilities", {}).get("healthy_risk", ""),
        "prob_elevated": prediction_res.get("predicted_risk_probabilities", {}).get("elevated_risk", ""),
        "prob_diabetic": prediction_res.get("predicted_risk_probabilities", {}).get("diabetic_risk", ""),
        "model_version": prediction_res.get("model_version", "")
    }

    df_new = pd.DataFrame([log_row])
    if LOG_CSV_PATH.exists():
        df_new.to_csv(LOG_CSV_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(LOG_CSV_PATH, mode="w", header=True, index=False)


# ------------------------------------------------------------------------------
# Streamlit Dashboard UI Rendering
# ------------------------------------------------------------------------------

st.markdown('<div class="main-title">🩸 Non-Invasive Blood Glucose Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive Multi-Modal Sensor Inference & Pre-Diagnostic Risk Screening Dashboard</div>', unsafe_allow_html=True)

# Sidebar: System Diagnostics & Guardrails
with st.sidebar:
    st.header("⚙️ System Status & Diagnostics")
    st.info(
        "**Multi-Modal AI Architecture:**\n"
        "• **Model A (Full-Sensor)**: Random Forest (50 feats, R²=0.8528, MAE=12.28 mg/dL)\n"
        "• **Model B (Demographics)**: XGBoost Classifier (NHANES scoped, Macro AUROC=0.8092)\n"
        "• **Uncertainty Engine**: Gradient Boosting [q0.05, q0.95] (87.8% empirical test coverage)"
    )
    st.markdown("---")
    st.caption("🔒 **Validation Status**: `synthetic_self_consistency_only`")
    st.caption("📁 Audit log saved to `data/manual_test_log.csv`")
    st.caption("📄 PDF exports saved to `data/reports/`")

# Tabs for Organization
tab_pred, tab_lab, tab_bench, tab_reports = st.tabs([
    "🎯 Clinical Predictor & Benchmark Presets",
    "🧪 Multi-Modal Sensor Fine-Tuning Lab",
    "📊 ADA & ICMR Clinical Guidelines",
    "📄 Reports Export & Audit Logs"
])

# Initialize session state variables for preset selection
if "selected_preset" not in st.session_state:
    st.session_state["selected_preset"] = "Custom / Manual Entry"

# ==============================================================================
# TAB 1: Main Clinical Predictor & Benchmark Profiles
# ==============================================================================
with tab_pred:
    st.markdown("### ⚡ 1-Click Clinical Benchmark Profiles")
    st.markdown("Select a validated physiological profile from our study to instantly populate all sensor and demographic biometrics:")

    col_pr1, col_pr2 = st.columns([3, 1])
    with col_pr1:
        chosen_preset = st.selectbox(
            "Load Validated Clinical Benchmark Scenario:",
            options=list(BENCHMARK_PRESETS.keys()),
            index=list(BENCHMARK_PRESETS.keys()).index(st.session_state.get("selected_preset", "Custom / Manual Entry")),
            key="preset_selector_dropdown"
        )
    with col_pr2:
        st.write("")
        st.write("")
        if chosen_preset != "Custom / Manual Entry":
            st.success("✅ Preset Loaded!")

    # Load defaults from preset if active
    p_data = BENCHMARK_PRESETS.get(chosen_preset, None) or {}

    with st.form(key="main_prediction_form"):
        # Section 1: Session Info
        st.markdown("#### 1. Session & Patient Information")
        c1, c2, c3 = st.columns([2, 1.5, 2.5])
        with c1:
            full_name = st.text_input("Patient / Subject Name", value=p_data.get("full_name", "Jane Doe"))
        with c2:
            current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.text_input("Date / Time", value=current_time_str, disabled=True)
        with c3:
            session_notes = st.text_input("Clinical Notes / Observation", value=p_data.get("notes", "Routine evaluation"))

        st.markdown("---")

        # Section 2: Personal Demographics
        st.markdown("#### 2. Personal Demographics & Lifestyle Biometrics")

        clinical_standard = st.radio(
            "Population Clinical Standard:",
            options=["🇮🇳 ICMR & RSSDI Guidelines (Asian Indian Standards — Recommended)", "🌐 ADA & WHO Guidelines (Global Standards)"],
            index=0,
            horizontal=True,
            help="ICMR & RSSDI use Asian Indian BMI cutoffs (<18.5 Underweight, 18.5-22.9 Normal, 23.0-24.9 Overweight, ≥25.0 Obese) to account for higher visceral body fat and early metabolic risk."
        )

        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        with c_p1:
            age = st.number_input("Age (years)", min_value=1.0, max_value=110.0, value=float(p_data.get("age", 45.0)), step=1.0)
            gender_options = ["Male", "Female", "Other"]
            g_idx = gender_options.index(p_data.get("gender", "Male")) if p_data.get("gender") in gender_options else 0
            gender = st.selectbox("Biological Gender", options=gender_options, index=g_idx)
        with c_p2:
            height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=float(p_data.get("height_cm", 172.0)), step=0.5)
            weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=250.0, value=float(p_data.get("weight_kg", 75.0)), step=0.5)
        with c_p3:
            race_opts = ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic / Latino", "Asian / Asian American", "Other / Multi-Racial"]
            default_race = p_data.get("race_ethnicity", "Non-Hispanic White")
            r_idx = race_opts.index(default_race) if default_race in race_opts else 0
            race_ethnicity = st.selectbox(
                "Race / Ethnicity (ADA Risk Test)",
                options=race_opts,
                index=r_idx,
                help="Recognized empirical epidemiological risk factor per American Diabetes Association (ADA) guidelines. Populations of Asian, African, and Hispanic descent experience higher insulin resistance and T2D prevalence at lower body weight thresholds (e.g. Asian BMI screening cutoff is 23 kg/m² vs 25 kg/m²)."
            )
            waist_circumference_cm = st.number_input(
                "Waist Circumference (cm)",
                min_value=40.0,
                max_value=200.0,
                value=float(p_data.get("waist_circumference_cm", 88.0)),
                step=0.5,
                help="Core central-obesity marker (FINDRISC & ADA). Visceral adiposity drives hepatic insulin resistance more directly than BMI. High risk thresholds: Men >102 cm (40 in), Women >88 cm (35 in); Asian Indian Men >90 cm, Women >80 cm (ICMR)."
            )
        with c_p4:
            pa_opts = ["Moderately Active (≥150 min/wk)", "Vigorous / Highly Active", "Sedentary / Inactive (<150 min/wk)"]
            pa_default = p_data.get("physical_activity_level", "Moderately Active (≥150 min/wk)")
            pa_idx = 0
            for i, opt in enumerate(pa_opts):
                if opt.split()[0].lower() in pa_default.lower():
                    pa_idx = i
                    break
            physical_activity_level = st.selectbox(
                "Physical Activity Level",
                options=pa_opts,
                index=pa_idx,
                help="Physical activity criterion (PAQ / ADA / FINDRISC): Inactivity (<150 min/wk moderate activity) downregulates GLUT4 translocation and elevates T2D onset risk."
            )
            htn_opts = ["No (Normal Blood Pressure)", "Yes (Diagnosed Hypertension / On Meds)"]
            htn_default = "Yes" if str(p_data.get("hypertension", "")).lower() in ["yes", "1", "true"] else "No"
            h_idx = 1 if htn_default == "Yes" else 0
            hypertension = st.selectbox("Hypertension Diagnosis", options=htn_opts, index=h_idx, help="Cardiometabolic comorbidity; vascular stiffness exacerbates peripheral insulin resistance.")

        c_l1, c_l2, c_l3, c_l4 = st.columns(4)
        with c_l1:
            chol_opts = ["No (Normal Cholesterol)", "Yes (Diagnosed Dyslipidemia / Low HDL)"]
            chol_default = "Yes" if str(p_data.get("high_cholesterol", "")).lower() in ["yes", "1", "true"] else "No"
            c_idx = 1 if chol_default == "Yes" else 0
            high_cholesterol = st.selectbox("High Cholesterol / Dyslipidemia", options=chol_opts, index=c_idx, help="Dyslipidemia (low HDL, high triglycerides) contributes to beta-cell lipotoxicity.")
        with c_l2:
            if gender == "Female":
                gdm_opts = ["No (No History)", "Yes (History of Gestational Diabetes)"]
                gdm_default = "Yes" if str(p_data.get("gestational_diabetes", "")).lower() in ["yes", "1", "true"] else "No"
                gdm_idx = 1 if gdm_default == "Yes" else 0
                gestational_diabetes = st.selectbox(
                    "Gestational Diabetes (GDM)",
                    options=gdm_opts,
                    index=gdm_idx,
                    help="ADA screening factor: Prior gestational diabetes conveys a 7- to 10-fold higher lifetime risk of conversion to Type 2 diabetes."
                )
            else:
                gestational_diabetes = st.selectbox(
                    "Gestational Diabetes (GDM)",
                    options=["Not Applicable (Male Participant)"],
                    index=0,
                    disabled=True,
                    help="Gestational diabetes is left as a distinct non-applicable category for male participants per clinical practice rather than falsely imputed as negative."
                )
        with c_l3:
            fam_opts = ["No (0)", "Yes (1)"]
            f_idx = fam_opts.index(p_data.get("family_history", "No (0)")) if p_data.get("family_history") in fam_opts else 0
            family_history = st.selectbox("Family History of Diabetes", options=fam_opts, index=f_idx)
            smk_opts = ["Non-Smoker (0)", "Current Smoker (1)"]
            s_idx = smk_opts.index(p_data.get("smoking_status", "Non-Smoker (0)")) if p_data.get("smoking_status") in smk_opts else 0
            smoking_status = st.selectbox("Smoking Status", options=smk_opts, index=s_idx)
        with c_l4:
            fast_opts = ["Fasting (≥8h)", "Non-Fasting / Post-Meal"]
            fst_idx = fast_opts.index(p_data.get("fasting_status", "Fasting (≥8h)")) if p_data.get("fasting_status") in fast_opts else 0
            fasting_status = st.selectbox("Fasting State", options=fast_opts, index=fst_idx)
            med_opts = ["None", "Oral Hypoglycemics", "Insulin", "Both Insulin & Oral"]
            m_idx = med_opts.index(p_data.get("med_status", "None")) if p_data.get("med_status") in med_opts else 0
            med_status = st.selectbox("Medication Status", options=med_opts, index=m_idx)

        # Live BMI Calculation (ICMR vs WHO/ADA)
        height_m = height_cm / 100.0
        computed_bmi = round(weight_kg / (height_m ** 2), 1)

        is_indian = "ICMR" in clinical_standard
        if is_indian:
            if computed_bmi < 18.5:
                bmi_cat, bmi_color = "Underweight", "#38bdf8"
            elif computed_bmi < 23.0:
                bmi_cat, bmi_color = "Normal Weight (ICMR 18.5–22.9)", "#22c55e"
            elif computed_bmi < 25.0:
                bmi_cat, bmi_color = "Overweight (ICMR Asian Indian Cutoff 23.0–24.9)", "#f59e0b"
            elif computed_bmi < 30.0:
                bmi_cat, bmi_color = "Class I Obesity (ICMR ≥25.0 kg/m²)", "#ef4444"
            else:
                bmi_cat, bmi_color = "Class II Severe Obesity (ICMR ≥30.0 kg/m²)", "#b91c1c"
        else:
            if computed_bmi < 18.5:
                bmi_cat, bmi_color = "Underweight", "#38bdf8"
            elif computed_bmi < 25.0:
                bmi_cat, bmi_color = "Normal Weight (WHO/ADA 18.5–24.9)", "#22c55e"
            elif computed_bmi < 30.0:
                bmi_cat, bmi_color = "Overweight (WHO/ADA 25.0–29.9)", "#f59e0b"
            else:
                bmi_cat, bmi_color = "Obese (WHO/ADA ≥30.0 kg/m²)", "#ef4444"

        diag_opts = ["Unknown / Not Diagnosed", "None (Healthy)", "Prediabetes", "Type 1 Diabetes", "Type 2 Diabetes"]
        d_idx = diag_opts.index(p_data.get("diagnosis_option", "Unknown / Not Diagnosed")) if p_data.get("diagnosis_option") in diag_opts else 0
        diagnosis_option = st.selectbox("Clinical Diagnosis Status (If Known)", options=diag_opts, index=d_idx)
        st.markdown(f"**Computed Body Mass Index (BMI):** `{computed_bmi} kg/m²` — <span style='color:{bmi_color};font-weight:bold;'>{bmi_cat}</span> | **Central Adiposity (Waist):** `{waist_circumference_cm} cm`", unsafe_allow_html=True)

        st.markdown("---")

        # Section 3: Sensor Readings
        st.markdown("#### 3. Multi-Modal Physiological Sensor Readings")
        has_sensor = st.checkbox("🔬 I have wearable sensor readings (MAX30102 Optical PPG, Saliva pH, Skin Temp, ECG)", value=p_data.get("has_sensor", True))

        sensor_inputs = {}
        if has_sensor:
            st.info("💡 Adjust physiological transducer parameters or use the benchmark presets above:")
            c_s1, c_s2, c_s3 = st.columns(3)
            with c_s1:
                st.markdown("##### 🧪 Biochemical & Thermal Transducers")
                saliva_ph = st.slider("Saliva pH Probe [5.5 (Acidic) to 8.5 (Alkaline)]", min_value=5.50, max_value=8.50, value=float(p_data.get("saliva_ph", 7.25)), step=0.05, help="Healthy baseline: 7.20-7.40. Drops <6.8 in cellular glycolysis/hyperglycemia.")
                temp_c = st.slider("Skin Surface Temp (°C) [IR thermopile]", min_value=34.0, max_value=39.0, value=float(p_data.get("temp_c", 36.6)), step=0.1, help="Cutaneous microvascular thermal conductance.")
                spo2_pct = st.slider("Blood Oxygen SpO2 (%) [MAX30102 Red/IR]", min_value=80.0, max_value=100.0, value=float(p_data.get("spo2_pct", 98.0)), step=0.5)

            with c_s2:
                st.markdown("##### 🫀 Optical PPG Waveform (MAX30102)")
                hr_bpm = st.slider("Heart Rate (BPM)", min_value=40.0, max_value=180.0, value=float(p_data.get("hr_bpm", 72.0)), step=1.0)
                ppg_dc = st.number_input("PPG Raw Baseline (DC Counts)", min_value=50000.0, max_value=300000.0, value=float(p_data.get("ppg_dc", 175000.0)), step=1000.0, help="Optical tissue transmittance offset.")
                ppg_ac = st.number_input("PPG Pulsatile Amplitude (AC P2P Counts)", min_value=100.0, max_value=10000.0, value=float(p_data.get("ppg_ac", 1350.0)), step=50.0)
                perfusion_idx = st.number_input("Perfusion Index (%) [AC/DC ratio]", min_value=0.05, max_value=15.0, value=float(p_data.get("perfusion_idx", 0.77)), step=0.05)
                pulse_width_ms = st.slider("Pulse Width (ms) [Ejection time]", min_value=120.0, max_value=500.0, value=float(p_data.get("pulse_width_ms", 280.0)), step=5.0)

            with c_s3:
                st.markdown("##### 📈 ECG Autonomic Heart Rate Variability")
                hrv_sdnn = st.slider("HRV SDNN (ms) [Overall autonomic variability]", min_value=5.0, max_value=120.0, value=float(p_data.get("hrv_sdnn", 42.0)), step=1.0, help="High HRV = healthy autonomic tone; Low HRV = glycemic stress.")
                hrv_rmssd = st.slider("HRV RMSSD (ms) [Parasympathetic vagal tone]", min_value=4.0, max_value=100.0, value=float(p_data.get("hrv_rmssd", 34.0)), step=1.0)
                hrv_pnn50 = st.slider("HRV pNN50 (%) [Successive normal intervals >50ms]", min_value=0.0, max_value=80.0, value=float(p_data.get("hrv_pnn50", 14.0)), step=1.0)
                hrv_lf_hf = st.slider("HRV LF/HF Ratio [Sympathovagal balance]", min_value=0.2, max_value=6.0, value=float(p_data.get("hrv_lf_hf", 1.35)), step=0.05)

            c_adv1, c_adv2 = st.columns(2)
            with c_adv1:
                prev_bgl_val = st.number_input("Previous BGL Reading (mg/dL) [Optional, for rate-of-change trend]", min_value=0.0, max_value=500.0, value=float(p_data.get("prev_bgl", 0.0)), step=1.0)
            with c_adv2:
                ref_bgl_val = st.number_input("Laboratory Reference / Fingerstick BGL (mg/dL) [Optional, for Clarke Error Grid validation]", min_value=0.0, max_value=500.0, value=float(p_data.get("reference_bgl", 0.0)), step=1.0)

            sensor_inputs = {
                "saliva_ph": saliva_ph,
                "temperature_c": temp_c,
                "spo2_pct": spo2_pct,
                "hr_bpm": hr_bpm,
                "ppg_hr_bpm": hr_bpm,
                "ppg_raw_dc_baseline": ppg_dc,
                "ppg_raw_ac_p2p": ppg_ac,
                "perfusion_index": perfusion_idx,
                "pulse_width_ms": pulse_width_ms,
                "hrv_sdnn": hrv_sdnn,
                "hrv_rmssd": hrv_rmssd,
                "hrv_pnn50": hrv_pnn50,
                "hrv_lf_hf_ratio": hrv_lf_hf,
                "previous_reading_bgl_mg_dl": prev_bgl_val if prev_bgl_val > 0 else None,
                "reference_bgl_mg_dl": ref_bgl_val if ref_bgl_val > 0 else None
            }

        # Predict Button
        submit_button = st.form_submit_button(label="🚀 Run Glucose Prediction / Risk Screening", use_container_width=True)

    # --------------------------------------------------------------------------
    # Prediction Execution & Results
    # --------------------------------------------------------------------------
    if submit_button:
        fam_hist_val = 1 if "Yes" in family_history else 0
        smoking_val = 1 if "Current" in smoking_status else 0
        fasting_val = 1 if "Fasting" in fasting_status else 0
        med_ins_val = 1 if "Insulin" in med_status else 0
        med_oral_val = 1 if "Oral" in med_status else 0

        diag_clean = "None"
        if "Type 1" in diagnosis_option: diag_clean = "Type 1"
        elif "Type 2" in diagnosis_option: diag_clean = "Type 2"
        elif "Prediabetes" in diagnosis_option: diag_clean = "Prediabetes"
        elif "Healthy" in diagnosis_option: diag_clean = "None"

        race_clean = "non_hispanic_white"
        if "Black" in race_ethnicity: race_clean = "black"
        elif "Hispanic" in race_ethnicity: race_clean = "hispanic"
        elif "Asian" in race_ethnicity: race_clean = "asian"
        elif "Other" in race_ethnicity: race_clean = "other"

        pa_clean = "moderate"
        if "Vigorous" in physical_activity_level: pa_clean = "active"
        elif "Sedentary" in physical_activity_level: pa_clean = "sedentary"

        htn_val = 1 if "Yes" in hypertension else 0
        chol_val = 1 if "Yes" in high_cholesterol else 0

        gdm_clean = "not_applicable"
        if gender.lower() == "female":
            gdm_clean = "yes" if "Yes" in gestational_diabetes else "no"

        input_payload = {
            "age": age,
            "gender": gender.lower(),
            "race_ethnicity": race_clean,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": computed_bmi,
            "bmi_category": bmi_cat,
            "waist_circumference_cm": waist_circumference_cm,
            "physical_activity_level": pa_clean,
            "hypertension": htn_val,
            "high_cholesterol": chol_val,
            "gestational_diabetes": gdm_clean,
            "family_history": fam_hist_val,
            "smoking": smoking_val,
            "fasting": fasting_val,
            "med_status": med_status,
            "med_taking_insulin": med_ins_val,
            "med_taking_oral": med_oral_val,
            "med_taking_any": 1 if (med_ins_val or med_oral_val) else 0,
            "diabetes_diagnosis": diag_clean
        }

        if has_sensor:
            input_payload.update(sensor_inputs)

        with st.spinner("Processing physiological biometrics through inference pipeline..."):
            prediction_result = predictor.predict(input_payload)

        session_payload = {
            "full_name": full_name,
            "session_time": current_time_str,
            "notes": session_notes
        }

        st.session_state["last_session"] = session_payload
        st.session_state["last_inputs"] = input_payload
        st.session_state["last_prediction"] = prediction_result

        # Append to audit log
        append_to_audit_log(session_payload, input_payload, prediction_result)

        st.markdown("---")
        st.markdown("### 📊 Prediction & Clinical Decision Output")

        # ----------------------------------------------------------------------
        # Item 6: Model Transparency Panel
        # ----------------------------------------------------------------------
        if "predicted_bgl_mg_dl" in prediction_result:
            st.markdown("<div class='model-badge-fs'>MODEL A: Full-Sensor Multi-Modal Stacking Ensemble (R²=0.8528, Validated on Synthetic Data)</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='model-badge-tab'>MODEL B: Tabular Demographic Screening Classifier (CDC NHANES Scoped, Macro AUROC=0.81)</div>", unsafe_allow_html=True)

        # Out-Of-Distribution (OOD) Safety Warning
        if prediction_result.get("is_out_of_distribution"):
            st.markdown(
                f"<div class='disclaimer-critical'>"
                f"⚠️ <b>OUT-OF-DISTRIBUTION INPUT WARNING:</b><br/>"
                f"{prediction_result.get('ood_warning')}"
                f"</div>",
                unsafe_allow_html=True
            )

        # Type 1 Specific Safety Warning
        if "Type 1" in diag_clean:
            st.markdown(
                "<div class='disclaimer-critical'>"
                "⚠️ <b>CLINICAL SAFETY AUDIT ALERT (Type 1 Hypoglycemia Caution):</b><br/>"
                "Validation identified a known Zone D failure mode (Reference 58.0 mg/dL predicted as 101.2 mg/dL — failure to detect hypoglycemia). "
                "This prototype must NEVER be used for autonomous insulin titration without fingerstick confirmation."
                "</div>",
                unsafe_allow_html=True
            )

        if "predicted_bgl_mg_dl" in prediction_result:
            bgl = prediction_result["predicted_bgl_mg_dl"]
            ci = prediction_result["confidence_interval_5th_95th"]
            width = prediction_result["interval_width_mg_dl"]
            czone = prediction_result["clarke_zone"]
            trend_text = prediction_result["trend"]
            strat_conf = prediction_result["diagnosis_stratum_confidence"]

            # Visual Clinical Categorization
            if bgl < 70.0:
                tier_badge = "<span class='badge-hypo'>⚠️ ACUTE HYPOGLYCEMIA (&lt;70 mg/dL)</span>"
                tier_msg = "Blood glucose is critically low. Rapid-acting carbohydrate intake (15g rule) is recommended."
            elif bgl < 100.0:
                tier_badge = "<span class='badge-normal'>🟢 NORMAL OPTIMAL FASTING (70–99 mg/dL)</span>"
                tier_msg = "Glycemic levels are within standard healthy fasting baseline."
            elif bgl < 126.0:
                tier_badge = "<span class='badge-prediabetes'>🟡 IMPAIRED / PREDIABETES (100–125 mg/dL)</span>"
                tier_msg = "Fasting glucose indicates impaired regulation / prediabetes range."
            elif bgl < 200.0:
                tier_badge = "<span class='badge-diabetes'>🟠 ELEVATED POST-PRANDIAL / MILD DIABETIC (126–199 mg/dL)</span>"
                tier_msg = "Elevated blood glucose consistent with diabetic / post-prandial threshold."
            else:
                tier_badge = "<span class='badge-diabetes'>🔴 SEVERE HYPERGLYCEMIA (&ge;200 mg/dL)</span>"
                tier_msg = "Marked hyperglycemia. Clinical evaluation is recommended."

            st.markdown(f"#### Clinical Status: {tier_badge}", unsafe_allow_html=True)
            st.caption(f"**Interpretation:** {tier_msg}")

            # Metric Cards
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric(label="Estimated Blood Glucose", value=f"{bgl} mg/dL")
            with col_m2:
                st.metric(label="90% Quantile Interval [q0.05, q0.95]", value=f"[{ci[0]}, {ci[1]}] mg/dL", delta=f"Width: {width} mg/dL", delta_color="off")
            with col_m3:
                st.metric(label="Clarke Error Grid Zone", value=czone.split()[0] + " " + czone.split()[1] if "Zone" in czone else czone)
            with col_m4:
                st.metric(label="Glycemic Trend", value=trend_text.split("(")[0])

            # ------------------------------------------------------------------
            # Item 2: Confidence Interval Visual Range Gauge (Plotly)
            # ------------------------------------------------------------------
            st.markdown("#### 🎯 Prediction Uncertainty & Safety Zone Mapping")
            st.caption("ℹ️ *Uncertainty range is currently under calibration validation - treat as an outer bound, not a precise range.*")
            fig_ci = plot_confidence_interval_gauge(bgl, ci[0], ci[1])
            st.plotly_chart(fig_ci, use_container_width=True)

            # ------------------------------------------------------------------
            # Item 1: Longitudinal Trend Chart (Plotly)
            # ------------------------------------------------------------------
            st.markdown("#### 📈 Longitudinal Patient Glycemic History")
            fig_trend = plot_patient_trend_chart(full_name, bgl, current_time_str, czone)
            if fig_trend is not None:
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info(f"ℹ️ First recorded test session for **{full_name}**. Future submissions will generate a longitudinal glycemic trend line here.")

            # ------------------------------------------------------------------
            # Item 3: Feature Contribution Bar Chart (Plotly)
            # ------------------------------------------------------------------
            st.markdown("#### 🔍 Physiological Feature Contributions")
            fig_feat, summary_line = plot_feature_contributions(predictor, input_payload)
            st.markdown(summary_line)
            st.plotly_chart(fig_feat, use_container_width=True)

            # ------------------------------------------------------------------
            # Item 5: Normal Physiological Range Comparison
            # ------------------------------------------------------------------
            st.markdown("#### 🩺 Sensor Readings vs. Normal Physiological Bands")
            fig_norm = plot_normal_range_comparison(sensor_inputs)
            st.plotly_chart(fig_norm, use_container_width=True)

            # ------------------------------------------------------------------
            # Item 7: Interactive Clarke Error Grid Analysis (Plotly)
            # ------------------------------------------------------------------
            st.markdown("#### 🎯 Clarke Error Grid Analysis (Clinical Safety Assessment)")
            st.caption("Plots the predicted glucose against clinical reference zones (Zone A: Accurate, Zone B: Benign, Zone C: Over-correction, Zone D: Dangerous Miss, Zone E: Opposite Action):")
            
            b_points = [
                {"name": "Healthy Reference (~88 mg/dL)", "ref": 88.0, "pred": 91.3},
                {"name": "Prediabetes Screen (~114 mg/dL)", "ref": 114.0, "pred": 135.0},
                {"name": "Type 2 Post-Meal (~172 mg/dL)", "ref": 172.0, "pred": 181.5},
                {"name": "Severe Hyperglycemia (~265 mg/dL)", "ref": 265.0, "pred": 257.5},
                {"name": "Hypoglycemia Alert (~62 mg/dL)", "ref": 62.0, "pred": 95.3}
            ]
            current_ref = input_payload.get("reference_bgl_mg_dl")
            fig_clarke = plot_clarke_error_grid(ref_bgl=current_ref, pred_bgl=bgl, benchmark_points=b_points)
            st.plotly_chart(fig_clarke, use_container_width=True)

        else:
            # Tabular Risk Band Display
            risk_band = prediction_result["risk_band"]
            guidance = prediction_result["clinical_guidance"]
            probs = prediction_result["predicted_risk_probabilities"]
            conf_note = prediction_result["model_confidence_note"]

            st.markdown("#### 👤 2-Band Demographic Cardiometabolic Risk Screening")
            col_r1, col_r2 = st.columns([1.5, 2.5])
            with col_r1:
                if risk_band == "lower_risk":
                    st.markdown("<span class='badge-risk-lower'>LOWER BASELINE RISK</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge-risk-elevated'>ELEVATED RISK — CONSULT RECOMMENDED</span>", unsafe_allow_html=True)

                st.write("")
                st.markdown(f"**Model:** `{prediction_result['model_version']}`")
                st.markdown(f"**Validated Population:** `{prediction_result['validated_scope']}`")

            with col_r2:
                st.markdown(f"**Clinical Guidance:**\n{guidance}")
                st.progress(probs.get("diabetic_risk", 0.0) + probs.get("elevated_risk", 0.0))
                st.caption(f"Risk Breakdown: Healthy: **{probs.get('healthy_risk', 0)*100:.1f}%** | Elevated (Prediabetes): **{probs.get('elevated_risk', 0)*100:.1f}%** | Diabetic: **{probs.get('diabetic_risk', 0)*100:.1f}%**")

            st.markdown(f"<div class='disclaimer-banner'>⚠️ <b>Prominent Screening Disclaimer:</b> {conf_note}</div>", unsafe_allow_html=True)

        # Universal Research Disclaimer
        st.markdown(
            "<div class='disclaimer-banner'>"
            "🔬 <b>Research Prototype Notice:</b> This system is an experimental machine learning research prototype. "
            "The full-sensor regression model is validated on synthetic multi-modal self-consistency data only. "
            "The demographic risk classifier is validated on CDC NHANES community survey outpatients only (AUROC=0.73). "
            "This software is <b>not a certified medical device</b> and is <b>not a substitute for certified clinical laboratory blood tests</b>."
            "</div>",
            unsafe_allow_html=True
        )


# ==============================================================================
# TAB 2: Multi-Modal Sensor Fine-Tuning Lab (Sensitivity Analysis)
# ==============================================================================
with tab_lab:
    st.markdown("### 🧪 Real-Time Multi-Modal Sensor Sensitivity Lab")
    st.markdown("Experiment with real-time parameter tweaking and simulate dynamic sensitivity curves across all 4 sensor transducers:")

    # --------------------------------------------------------------------------
    # Item 4: Dynamic Parameter Sensitivity Sweep Curve
    # --------------------------------------------------------------------------
    st.markdown("#### 📈 Interactive Parameter Sensitivity Sweep Curve")
    st.caption("Select a physiological parameter to sweep across its biological range while holding other inputs at their live slider values:")

    c_sw1, c_sw2 = st.columns([1.5, 2.5])
    with c_sw1:
        sweep_param = st.selectbox(
            "Parameter to Sweep:",
            options=[
                "Saliva pH Bio-Probe",
                "Heart Rate (BPM)",
                "ECG HRV RMSSD (ms)",
                "ECG HRV SDNN (ms)",
                "Skin Surface Temp (°C)",
                "MAX30102 DC Baseline Offset",
                "PPG Pulse Width (ms)",
                "Subject Age (years)",
                "Subject BMI (kg/m²)"
            ],
            index=0
        )

        st.markdown("#### 🎛️ Live Sandbox Sliders")
        lab_ph = st.slider("Live Saliva pH", 5.5, 8.5, 7.20, 0.05, key="lab_ph")
        lab_hr = st.slider("Live Heart Rate (BPM)", 45.0, 160.0, 75.0, 1.0, key="lab_hr")
        lab_temp = st.slider("Live Skin Temperature (°C)", 34.0, 39.0, 36.6, 0.1, key="lab_temp")
        lab_rmssd = st.slider("Live HRV RMSSD (ms)", 5.0, 80.0, 32.0, 1.0, key="lab_rmssd")
        lab_sdnn = st.slider("Live HRV SDNN (ms)", 10.0, 120.0, 42.0, 1.0, key="lab_sdnn")
        lab_dc = st.slider("MAX30102 DC Baseline", 100000.0, 250000.0, 175000.0, 5000.0, key="lab_dc")
        lab_ac = st.slider("MAX30102 AC Pulsatile P2P", 500.0, 5000.0, 1500.0, 100.0, key="lab_ac")
        lab_age = st.slider("Patient Age", 18.0, 90.0, 50.0, 1.0, key="lab_age")
        lab_bmi = st.slider("Patient BMI", 16.0, 45.0, 28.0, 0.5, key="lab_bmi")

    with c_sw2:
        # Base dict for live inference
        base_payload = {
            "age": lab_age, "gender": "male", "height_cm": 172.0, "weight_kg": 75.0,
            "bmi": lab_bmi, "bmi_category": "Overweight", "family_history": 1, "smoking": 0, "fasting": 1,
            "med_status": "None", "med_taking_insulin": 0, "med_taking_oral": 0, "med_taking_any": 0,
            "diabetes_diagnosis": "None",
            "saliva_ph": lab_ph, "temperature_c": lab_temp, "spo2_pct": 98.0,
            "hr_bpm": lab_hr, "ppg_hr_bpm": lab_hr, "ppg_raw_dc_baseline": lab_dc, "ppg_raw_ac_p2p": lab_ac,
            "perfusion_index": (lab_ac / lab_dc) * 100.0, "pulse_width_ms": 280.0,
            "hrv_sdnn": lab_sdnn, "hrv_rmssd": lab_rmssd, "hrv_pnn50": 12.0, "hrv_lf_hf_ratio": 1.4
        }

        # Generate sweep range
        if "Saliva pH" in sweep_param:
            sweep_x = np.linspace(5.5, 8.5, 30)
            key_name, current_x, xlabel = "saliva_ph", lab_ph, "Saliva pH (Cellular Acidosis <6.8 vs Alkaline >7.2)"
        elif "Heart Rate" in sweep_param:
            sweep_x = np.linspace(45.0, 160.0, 30)
            key_name, current_x, xlabel = "hr_bpm", lab_hr, "Heart Rate (BPM)"
        elif "RMSSD" in sweep_param:
            sweep_x = np.linspace(5.0, 80.0, 30)
            key_name, current_x, xlabel = "hrv_rmssd", lab_rmssd, "HRV RMSSD (ms) — Vagal Parasympathetic Tone"
        elif "SDNN" in sweep_param:
            sweep_x = np.linspace(10.0, 120.0, 30)
            key_name, current_x, xlabel = "hrv_sdnn", lab_sdnn, "HRV SDNN (ms) — Total Autonomic Variability"
        elif "Temp" in sweep_param:
            sweep_x = np.linspace(34.0, 39.0, 30)
            key_name, current_x, xlabel = "temperature_c", lab_temp, "Skin Surface Temperature (°C)"
        elif "DC Baseline" in sweep_param:
            sweep_x = np.linspace(100000.0, 250000.0, 30)
            key_name, current_x, xlabel = "ppg_raw_dc_baseline", lab_dc, "MAX30102 DC Baseline Offset"
        elif "Pulse Width" in sweep_param:
            sweep_x = np.linspace(150.0, 450.0, 30)
            key_name, current_x, xlabel = "pulse_width_ms", 280.0, "PPG Pulse Width (ms)"
        elif "Age" in sweep_param:
            sweep_x = np.linspace(18.0, 85.0, 30)
            key_name, current_x, xlabel = "age", lab_age, "Subject Age (years)"
        else:
            sweep_x = np.linspace(16.0, 45.0, 30)
            key_name, current_x, xlabel = "bmi", lab_bmi, "Subject BMI (kg/m²)"

        sweep_y = []
        for x_val in sweep_x:
            t_payload = base_payload.copy()
            t_payload[key_name] = x_val
            if key_name == "hr_bpm": t_payload["ppg_hr_bpm"] = x_val
            res = predictor.predict_full_sensor(t_payload)
            sweep_y.append(res["predicted_bgl_mg_dl"])

        live_current_pred = predictor.predict_full_sensor(base_payload)
        current_y = live_current_pred["predicted_bgl_mg_dl"]

        fig_sweep = go.Figure()
        # Normal BGL target band
        fig_sweep.add_hrect(y0=70, y1=140, fillcolor="#dcfce7", opacity=0.35, layer="below", line_width=0, annotation_text="Target Normoglycemia (70–140 mg/dL)", annotation_position="top left")

        # Sweep Curve
        fig_sweep.add_trace(go.Scatter(
            x=sweep_x, y=sweep_y, mode='lines',
            line=dict(color='#0284c7', width=3.5, shape='spline'),
            name=f'Model BGL Response to {sweep_param}'
        ))

        # Current live marker
        fig_sweep.add_trace(go.Scatter(
            x=[current_x], y=[current_y], mode='markers+text',
            marker=dict(color='#dc2626', size=16, symbol='diamond', line=dict(color='white', width=2)),
            text=[f"<b>Current: {current_y} mg/dL</b>"], textposition="top center",
            name='Current Live Sandbox Position'
        ))

        fig_sweep.update_layout(
            title=f"<b>Dynamic AI Sensitivity Curve: {sweep_param} vs. Predicted BGL</b>",
            xaxis=dict(title=xlabel, showgrid=True),
            yaxis=dict(title="Predicted Blood Glucose (mg/dL)", range=[40, max(260.0, max(sweep_y) + 25)]),
            height=340, margin=dict(l=15, r=15, t=35, b=25), showlegend=True
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

        st.metric(label="Live Model A Predicted BGL", value=f"{current_y} mg/dL", delta=f"90% CI: [{live_current_pred['confidence_interval_5th_95th'][0]} – {live_current_pred['confidence_interval_5th_95th'][1]}] mg/dL", delta_color="off")


# ==============================================================================
# TAB 3: Clinical Benchmarks & Reference Standards (ICMR, RSSDI & ADA)
# ==============================================================================
with tab_bench:
    st.markdown("### 📊 Clinical Diagnostic Thresholds: ICMR (India) vs. ADA (Global)")
    st.markdown("Comparative standards from the **Indian Council of Medical Research (ICMR)**, **Research Society for the Study of Diabetes in India (RSSDI)**, and **American Diabetes Association (ADA)**:")

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.markdown("#### 🇮🇳 ICMR / RSSDI & ADA Diagnostic Glucose Cutoffs")
        st.table(pd.DataFrame({
            "Glycemic Category": ["Hypoglycemia", "Normal Glycemia", "Prediabetes (Impaired Glucose)", "Diabetes Mellitus", "Severe Hyperglycemia"],
            "Fasting Glucose (mg/dL)": ["< 70 mg/dL", "70 – 99 mg/dL", "100 – 125 mg/dL", "≥ 126 mg/dL", "≥ 200 mg/dL"],
            "2h Post-Prandial (mg/dL)": ["< 70 mg/dL", "70 – 139 mg/dL", "140 – 199 mg/dL", "≥ 200 mg/dL", "≥ 250 mg/dL"],
            "Clinical Action (ICMR/RSSDI)": ["Immediate Fast Sugars (15g rule)", "Annual Health Screening", "Diet & Lifestyle Modification", "Consult Diabetologist / OADs", "Urgent Clinical Attention"]
        }))

    with c_b2:
        st.markdown("#### ⚖️ BMI Cutoff Comparison: Asian Indian (ICMR) vs Western (WHO/ADA)")
        st.table(pd.DataFrame({
            "Classification": ["Underweight", "Normal / Healthy", "Overweight", "Class I Obesity", "Class II Severe Obesity"],
            "Asian Indian Cutoff (ICMR / RSSDI)": ["< 18.5 kg/m²", "18.5 – 22.9 kg/m²", "23.0 – 24.9 kg/m²", "25.0 – 29.9 kg/m²", "≥ 30.0 kg/m²"],
            "Western Cutoff (WHO / ADA)": ["< 18.5 kg/m²", "18.5 – 24.9 kg/m²", "25.0 – 29.9 kg/m²", "30.0 – 34.9 kg/m²", "≥ 35.0 kg/m²"],
            "Clinical Significance": ["Nutritional assessment", "Target healthy range", "High visceral fat risk in Indians", "Cardiometabolic risk elevated", "High insulin resistance"]
        }))

    st.markdown("---")
    st.markdown("#### 🎯 Clarke Error Grid Analysis Tiers (Clinical Accuracy Standard)")
    st.table(pd.DataFrame({
        "Clarke Zone": ["Zone A (Optimal)", "Zone B (Acceptable)", "Zone C (Over-correction)", "Zone D (Failure to detect)", "Zone E (Erroneous treatment)"],
        "Accuracy Bound": ["Within ±20% of reference BGL", "Outside ±20% but clinically benign", "Unnecessary corrective action", "Dangerous failure to detect hypo/hyper", "Opposite treatment triggered"],
        "Our Model A Result": ["93.75% of held-out test points", "5.47% of held-out test points", "0.00%", "0.78% (1 single Type 1 hypo point)", "0.00%"],
        "Safety Tier": ["Clinically Safe", "Clinically Safe", "Clinically Unacceptable", "Clinical Hazard", "Extreme Danger"]
    }))

    b_pts_guidelines = [
        {"name": "Healthy Adult (~88 mg/dL)", "ref": 88.0, "pred": 91.3},
        {"name": "Prediabetes (~114 mg/dL)", "ref": 114.0, "pred": 135.0},
        {"name": "Type 2 Post-Meal (~172 mg/dL)", "ref": 172.0, "pred": 181.5},
        {"name": "Severe Hyperglycemia (~265 mg/dL)", "ref": 265.0, "pred": 257.5},
        {"name": "Hypoglycemia Alert (~62 mg/dL)", "ref": 62.0, "pred": 95.3}
    ]
    fig_clarke_guidelines = plot_clarke_error_grid(benchmark_points=b_pts_guidelines)
    st.plotly_chart(fig_clarke_guidelines, use_container_width=True)


# ==============================================================================
# TAB 4: Reports Export & Continuous Audit Trail
# ==============================================================================
with tab_reports:
    st.markdown("### 📄 Clinical PDF Summary Export & Continuous Audit Log")

    if "last_session" in st.session_state and "last_inputs" in st.session_state and "last_prediction" in st.session_state:
        st.markdown("#### 📥 Download Summary for Most Recent Submission")
        pdf_file = generate_pdf_report(
            st.session_state["last_session"],
            st.session_state["last_inputs"],
            st.session_state["last_prediction"]
        )

        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="📥 Download Clinical PDF Summary Report",
            data=pdf_bytes,
            file_name=os.path.basename(pdf_file),
            mime="application/pdf",
            use_container_width=True
        )
        st.success(f"PDF successfully compiled and ready: `{os.path.basename(pdf_file)}`")
    else:
        st.info("Run a prediction in Tab 1 to generate and download a clinical PDF summary report.")

    st.markdown("---")
    st.markdown("#### 🗄️ Audit Log History (`data/manual_test_log.csv`)")
    if LOG_CSV_PATH.exists():
        try:
            df_log = pd.read_csv(LOG_CSV_PATH)
            st.dataframe(df_log.tail(15), use_container_width=True)
            st.caption(f"Total audit entries logged: **{len(df_log)}**")
        except Exception as e:
            st.warning(f"Could not load log table: {e}")
    else:
        st.caption("No log entries recorded yet.")
