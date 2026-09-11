"""
Non-Invasive Blood Glucose Prediction System — Interactive Clinical Dashboard
Built with Streamlit & Phase 7 Production Inference Engine.

Sections:
1. Session Information (Patient/Subject metadata, timestamp, clinical notes)
2. Personal Parameters (Age, gender, height/weight with live BMI calculation, diagnosis, meds)
3. Sensor Readings (Collapsible multi-modal inputs: MAX30102 PPG, Saliva pH, Skin Temp, ECG-HRV)
4. Prediction & Clinical Risk Tiers (Full-Sensor BGL regression with 85.16% CI vs. 2-Band Demographic Screening)
5. Clinical PDF Report Generation (Instant PDF compilation saved to data/reports/)
6. Audit Logging (Appends all submissions to data/manual_test_log.csv)
"""

import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import streamlit as st

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
    page_title="Non-Invasive Blood Glucose Prediction System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .badge-zone-a {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid #86efac;
        display: inline-block;
    }
    .badge-zone-b {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid #fde047;
        display: inline-block;
    }
    .badge-zone-d {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        border: 1px solid #fca5a5;
        display: inline-block;
    }
    .badge-risk-lower {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.1rem;
        border: 1px solid #7dd3fc;
        display: inline-block;
    }
    .badge-risk-elevated {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.1rem;
        border: 1px solid #fdba74;
        display: inline-block;
    }
    .disclaimer-banner {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
        color: #92400e;
        font-size: 0.92rem;
    }
    .disclaimer-critical {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        color: #b91c1c;
        font-size: 0.92rem;
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
    from reportlab.lib.units import inch

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
    secondary_color = colors.HexColor("#0284c7")
    dark_text = colors.HexColor("#1e293b")
    light_bg = colors.HexColor("#f8fafc")

    title_style = ParagraphStyle('TitleStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=primary_color)
    sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=13, textColor=colors.HexColor("#64748b"))
    h2_style = ParagraphStyle('H2Style', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=primary_color, spaceBefore=8, spaceAfter=4)
    cell_style = ParagraphStyle('CellStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, leading=11, textColor=dark_text)
    cell_bold = ParagraphStyle('CellBold', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=dark_text)
    alert_style = ParagraphStyle('AlertStyle', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8, leading=11, textColor=colors.HexColor("#991b1b"))

    story = []

    # Title & Metadata
    story.append(Paragraph("Non-Invasive Glucose Prediction — Subject Clinical Summary", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Model Engine: {prediction_res.get('model_version', 'v1.0')}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=4, spaceAfter=8))

    # Session Table
    story.append(Paragraph("1. Session & Subject Information", h2_style))
    session_rows = [
        [Paragraph("<b>Full Name:</b>", cell_style), Paragraph(str(session_data.get("full_name", "N/A")), cell_style),
         Paragraph("<b>Date / Time:</b>", cell_style), Paragraph(str(session_data.get("session_time", "N/A")), cell_style)],
        [Paragraph("<b>Age:</b>", cell_style), Paragraph(f"{input_data.get('age', 'N/A')} yrs", cell_style),
         Paragraph("<b>Gender:</b>", cell_style), Paragraph(str(input_data.get("gender", "N/A")).capitalize(), cell_style)],
        [Paragraph("<b>BMI:</b>", cell_style), Paragraph(f"{input_data.get('bmi', 'N/A')} kg/m² ({input_data.get('bmi_category', 'N/A')})", cell_style),
         Paragraph("<b>Diagnosis:</b>", cell_style), Paragraph(str(input_data.get("diabetes_diagnosis", "None")), cell_style)],
        [Paragraph("<b>Family History:</b>", cell_style), Paragraph("Yes" if input_data.get("family_history", 0) == 1 else "No", cell_style),
         Paragraph("<b>Fasting Status:</b>", cell_style), Paragraph("Fasting (≥8h)" if input_data.get("fasting", 1) == 1 else "Non-Fasting", cell_style)],
        [Paragraph("<b>Notes:</b>", cell_style), Paragraph(str(session_data.get("notes", "None recorded")), cell_style),
         Paragraph("<b>Medications:</b>", cell_style), Paragraph(str(input_data.get("med_status", "None")), cell_style)]
    ]
    t_sess = Table(session_rows, colWidths=[100, 170, 100, 170])
    t_sess.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), light_bg),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_sess)
    story.append(Spacer(1, 8))

    # Prediction Results
    story.append(Paragraph("2. Clinical Model Output & Confidence Bounds", h2_style))

    if "predicted_bgl_mg_dl" in prediction_res:
        # Full Sensor Output
        bgl = prediction_res["predicted_bgl_mg_dl"]
        ci = prediction_res["confidence_interval_5th_95th"]
        pred_rows = [
            [Paragraph("<b>Predicted Blood Glucose:</b>", cell_bold), Paragraph(f"<b><font size=11 color='#1e3a8a'>{bgl} mg/dL</font></b>", cell_style)],
            [Paragraph("<b>5th–95th Percentile CI:</b>", cell_style), Paragraph(f"<b>[{ci[0]} – {ci[1]} mg/dL]</b> (Width: {prediction_res['interval_width_mg_dl']} mg/dL)", cell_style)],
            [Paragraph("<b>Interval Empirical Test Coverage:</b>", cell_style), Paragraph(prediction_res.get("empirical_interval_coverage", "85.16%"), cell_style)],
            [Paragraph("<b>Clarke Zone Tier:</b>", cell_style), Paragraph(f"<b>{prediction_res.get('clarke_zone', 'Zone A')}</b>", cell_style)],
            [Paragraph("<b>Trend (Rate of Change):</b>", cell_style), Paragraph(str(prediction_res.get("trend", "N/A")), cell_style)],
            [Paragraph("<b>Diagnostic Stratum Confidence:</b>", cell_style), Paragraph(str(prediction_res.get("diagnosis_stratum_confidence", "Standard")), cell_style)]
        ]
    else:
        # Tabular Risk Band Output
        r_band = prediction_res.get("risk_band", "N/A")
        probs = prediction_res.get("predicted_risk_probabilities", {})
        prob_str = f"Healthy: {probs.get('healthy_risk', 0)*100:.1f}% | Elevated (Prediabetes): {probs.get('elevated_risk', 0)*100:.1f}% | Diabetic: {probs.get('diabetic_risk', 0)*100:.1f}%"
        pred_rows = [
            [Paragraph("<b>Demographic Risk Band:</b>", cell_bold), Paragraph(f"<b><font size=11 color='#c2410c'>{r_band.upper()}</font></b>", cell_style)],
            [Paragraph("<b>Clinical Guidance:</b>", cell_style), Paragraph(str(prediction_res.get("clinical_guidance", "N/A")), cell_style)],
            [Paragraph("<b>Model Probabilities:</b>", cell_style), Paragraph(prob_str, cell_style)],
            [Paragraph("<b>Screening Scope:</b>", cell_style), Paragraph(str(prediction_res.get("validated_scope", "CDC NHANES Outpatients")), cell_style)],
            [Paragraph("<b>Clinical Confidence Note:</b>", cell_style), Paragraph(str(prediction_res.get("model_confidence_note", "N/A")), cell_style)]
        ]

    t_pred = Table(pred_rows, colWidths=[180, 360])
    t_pred.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0fdf4") if "predicted_bgl_mg_dl" in prediction_res else colors.HexColor("#fff7ed")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_pred)
    story.append(Spacer(1, 8))

    # Sensor Inputs Breakdown if Full-Sensor was used
    if "ppg_raw_dc_baseline" in input_data and input_data["ppg_raw_dc_baseline"] is not None:
        story.append(Paragraph("3. Multi-Modal Sensor Parameters Entered", h2_style))
        sensor_rows = [
            [Paragraph("<b>MAX30102 DC Baseline:</b>", cell_style), Paragraph(f"{input_data.get('ppg_raw_dc_baseline')} counts", cell_style),
             Paragraph("<b>MAX30102 AC P2P:</b>", cell_style), Paragraph(f"{input_data.get('ppg_raw_ac_p2p')} counts", cell_style)],
            [Paragraph("<b>Perfusion Index:</b>", cell_style), Paragraph(f"{input_data.get('perfusion_index', 'N/A')}%", cell_style),
             Paragraph("<b>Heart Rate:</b>", cell_style), Paragraph(f"{input_data.get('hr_bpm')} BPM", cell_style)],
            [Paragraph("<b>Saliva pH Sensor:</b>", cell_style), Paragraph(f"{input_data.get('saliva_ph')} pH", cell_style),
             Paragraph("<b>Skin Temperature:</b>", cell_style), Paragraph(f"{input_data.get('temperature_c')} °C", cell_style)],
            [Paragraph("<b>ECG HRV SDNN:</b>", cell_style), Paragraph(f"{input_data.get('hrv_sdnn', 'N/A')} ms", cell_style),
             Paragraph("<b>ECG HRV RMSSD:</b>", cell_style), Paragraph(f"{input_data.get('hrv_rmssd', 'N/A')} ms", cell_style)]
        ]
        t_sens = Table(sensor_rows, colWidths=[140, 130, 140, 130])
        t_sens.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), light_bg),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t_sens)
        story.append(Spacer(1, 8))

    # Mandatory Legal & Clinical Disclaimers Box
    disclaimer_rows = [[
        Paragraph(
            "<b>MANDATORY RESEARCH PROTOTYPE DISCLAIMER:</b><br/>"
            "This document is generated by an experimental research prototype. The full-sensor machine learning regression model "
            "is validated on synthetic multi-modal self-consistency data only. The tabular demographic classifier is validated on CDC NHANES "
            "community survey outpatients only (AUROC=0.73) and cannot reliably isolate prediabetes. This software is not an FDA-cleared "
            "medical diagnostic device and must NEVER be used to adjust insulin doses or substitute for certified clinical laboratory blood testing.",
            alert_style
        )
    ]]
    t_disc = Table(disclaimer_rows, colWidths=[540])
    t_disc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fef2f2")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#ef4444")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
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
        "height_cm": input_data.get("height_cm", ""),
        "weight_kg": input_data.get("weight_kg", ""),
        "bmi": input_data.get("bmi", ""),
        "bmi_category": input_data.get("bmi_category", ""),
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
# Streamlit UI Rendering
# ------------------------------------------------------------------------------

st.markdown('<div class="main-title">🩸 Non-Invasive Blood Glucose Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive Multi-Modal Sensor Inference & Pre-Diagnostic Risk Screening Dashboard</div>', unsafe_allow_html=True)

# Sidebar with Model Integrity Summary
with st.sidebar:
    st.header("⚙️ System Status & Guardrails")
    st.info(
        "**Model Architecture:**\n"
        "• **Model A**: Multi-Modal Random Forest (50 features, R²=0.8557, MAE=12.13 mg/dL)\n"
        "• **Model B**: XGBoost Risk Classifier (NHANES scoped, Macro AUROC=0.7296)\n"
        "• **Quantile Model**: Gradient Boosting [q0.05, q0.95] (85.16% empirical test coverage)"
    )
    st.markdown("---")
    st.caption("🔒 **Validation Status**: `synthetic_self_consistency_only`")
    st.caption("📁 Audit log saved to `data/manual_test_log.csv`")
    st.caption("📄 PDF exports saved to `data/reports/`")

# Main Form Container
with st.form(key="prediction_form"):

    # --------------------------------------------------------------------------
    # SECTION 1: Session Information
    # --------------------------------------------------------------------------
    st.subheader("1. Session Information")
    col1, col2, col3 = st.columns([2, 1.5, 2.5])
    with col1:
        full_name = st.text_input("Full Name", value="Jane Doe", help="Subject or patient name for reporting")
    with col2:
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.text_input("Date / Time", value=current_time_str, disabled=True)
    with col3:
        session_notes = st.text_input("Clinical Notes / Observation (Optional)", value="Routine morning screening", help="Context for the test session")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECTION 2: Personal Parameters
    # --------------------------------------------------------------------------
    st.subheader("2. Personal Demographics & Lifestyle Biometrics")
    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
    with c_p1:
        age = st.number_input("Age (years)", min_value=1.0, max_value=110.0, value=45.0, step=1.0)
        gender = st.selectbox("Biological Gender", options=["Male", "Female", "Other"], index=0)
    with c_p2:
        height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=172.0, step=0.5)
        weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=250.0, value=75.0, step=0.5)
    with c_p3:
        family_history = st.selectbox("Family History of Diabetes", options=["No (0)", "Yes (1)"], index=0)
        smoking_status = st.selectbox("Smoking Status", options=["Non-Smoker (0)", "Current Smoker (1)"], index=0)
    with c_p4:
        fasting_status = st.selectbox("Fasting State", options=["Fasting (≥8h)", "Non-Fasting / Post-Meal"], index=0)
        med_status = st.selectbox("Medication Status", options=["None", "Oral Hypoglycemics", "Insulin", "Both Insulin & Oral"], index=0)

    # Live BMI Calculation
    height_m = height_cm / 100.0
    computed_bmi = round(weight_kg / (height_m ** 2), 1)
    if computed_bmi < 18.5:
        bmi_cat = "Underweight"
        bmi_color = "#38bdf8"
    elif computed_bmi < 25.0:
        bmi_cat = "Normal Weight"
        bmi_color = "#22c55e"
    elif computed_bmi < 30.0:
        bmi_cat = "Overweight"
        bmi_color = "#f59e0b"
    else:
        bmi_cat = "Obese (Class I-III)"
        bmi_color = "#ef4444"

    # Diabetes Diagnosis Option
    st.markdown(f"**Computed Body Mass Index (BMI):** `{computed_bmi} kg/m²` — <span style='color:{bmi_color};font-weight:bold;'>{bmi_cat}</span>", unsafe_allow_html=True)
    diagnosis_option = st.selectbox(
        "Clinical Diagnosis Status (If Known)",
        options=["Unknown / Not Diagnosed", "None (Healthy)", "Prediabetes", "Type 1 Diabetes", "Type 2 Diabetes"],
        index=0,
        help="Optional: Enables stratified diagnostic uncertainty notes in Model A"
    )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # SECTION 3: Sensor Readings (Collapsible Toggle)
    # --------------------------------------------------------------------------
    st.subheader("3. Physiological Sensor Readings (Optional)")
    has_sensor = st.checkbox("🔬 I have sensor readings (MAX30102 Optical PPG, Saliva pH, Skin Temp, ECG)", value=False)

    sensor_inputs = {}
    if has_sensor:
        st.info("Enter measured or simulated transducer parameters. Values default to physiological baseline ranges.")
        
        c_s1, c_s2, c_s3 = st.columns(3)
        with c_s1:
            st.markdown("##### 🧪 Biochemical & Thermal Transducers")
            saliva_ph = st.number_input("Saliva pH (pH probe / strip)", min_value=5.0, max_value=9.0, value=7.25, step=0.05, help="Baseline mean: 7.255")
            temp_c = st.number_input("Skin Surface Temp (°C) [IR thermopile]", min_value=30.0, max_value=42.0, value=36.6, step=0.1)
            spo2_pct = st.slider("SpO2 Blood Oxygen (%) [MAX30102]", min_value=75.0, max_value=100.0, value=98.0, step=0.5)

        with c_s2:
            st.markdown("##### 🫀 Optical PPG Waveform (MAX30102)")
            hr_bpm = st.number_input("Heart Rate (BPM)", min_value=35.0, max_value=220.0, value=72.0, step=1.0)
            ppg_dc = st.number_input("PPG Raw Baseline (DC Counts) [MAX30102 IR]", min_value=50000.0, max_value=300000.0, value=175000.0, step=1000.0)
            ppg_ac = st.number_input("PPG Pulsatile Amplitude (AC P2P Counts)", min_value=100.0, max_value=10000.0, value=1200.0, step=50.0)
            perfusion_idx = st.number_input("Perfusion Index (%) [AC/DC ratio]", min_value=0.05, max_value=15.0, value=0.70, step=0.05)
            pulse_width_ms = st.number_input("Pulse Width (ms)", min_value=100.0, max_value=600.0, value=280.0, step=5.0)

        with c_s3:
            st.markdown("##### 📈 ECG Autonomic Heart Rate Variability")
            hrv_sdnn = st.number_input("HRV SDNN (ms) [Overall variability]", min_value=5.0, max_value=180.0, value=42.0, step=1.0)
            hrv_rmssd = st.number_input("HRV RMSSD (ms) [Parasympathetic tone]", min_value=5.0, max_value=150.0, value=34.0, step=1.0)
            hrv_pnn50 = st.number_input("HRV pNN50 (%)", min_value=0.0, max_value=100.0, value=12.0, step=1.0)
            hrv_lf_hf = st.number_input("HRV LF/HF Ratio [Sympathovagal balance]", min_value=0.1, max_value=10.0, value=1.35, step=0.05)

        # Dynamic Trend input
        st.markdown("##### ⏱️ Glycemic Trend Baseline")
        prev_bgl_val = st.number_input("Previous BGL Reading (mg/dL) [Optional, for rate-of-change trend]", min_value=0.0, max_value=500.0, value=0.0, step=1.0)

        sensor_inputs = {
            "saliva_ph": saliva_ph,
            "temperature_c": temp_c,
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
            "previous_reading_bgl_mg_dl": prev_bgl_val if prev_bgl_val > 0 else None
        }

    # Submit Button
    submit_button = st.form_submit_button(label="🚀 Run Glucose Prediction / Risk Screening", use_container_width=True)


# ------------------------------------------------------------------------------
# SECTION 4: Prediction Execution & Clinical Output Display
# ------------------------------------------------------------------------------

if submit_button:
    # Build payload
    fam_hist_val = 1 if "Yes" in family_history else 0
    smoking_val = 1 if "Current" in smoking_status else 0
    fasting_val = 1 if "Fasting" in fasting_status else 0
    
    med_ins_val = 1 if "Insulin" in med_status else 0
    med_oral_val = 1 if "Oral" in med_status else 0
    
    diag_clean = "None"
    if "Type 1" in diagnosis_option:
        diag_clean = "Type 1"
    elif "Type 2" in diagnosis_option:
        diag_clean = "Type 2"
    elif "Prediabetes" in diagnosis_option:
        diag_clean = "Prediabetes"
    elif "Healthy" in diagnosis_option:
        diag_clean = "None"

    input_payload = {
        "age": age,
        "gender": gender.lower(),
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": computed_bmi,
        "bmi_category": bmi_cat,
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

    # Run Prediction Dispatcher
    with st.spinner("Processing physiological biometrics through inference pipeline..."):
        prediction_result = predictor.predict(input_payload)

    # Session metadata container
    session_payload = {
        "full_name": full_name,
        "session_time": current_time_str,
        "notes": session_notes
    }

    # Store in session state for PDF download
    st.session_state["last_session"] = session_payload
    st.session_state["last_inputs"] = input_payload
    st.session_state["last_prediction"] = prediction_result

    # Log to CSV
    append_to_audit_log(session_payload, input_payload, prediction_result)

    # --------------------------------------------------------------------------
    # Output Display
    # --------------------------------------------------------------------------
    st.markdown("### 📊 Prediction & Clinical Assessment")

    if "predicted_bgl_mg_dl" in prediction_result:
        # Full-Sensor Path Display
        bgl = prediction_result["predicted_bgl_mg_dl"]
        ci = prediction_result["confidence_interval_5th_95th"]
        width = prediction_result["interval_width_mg_dl"]
        czone = prediction_result["clarke_zone"]
        trend_text = prediction_result["trend"]
        strat_conf = prediction_result["diagnosis_stratum_confidence"]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric(label="Estimated Blood Glucose", value=f"{bgl} mg/dL")
        with col_m2:
            st.metric(label="85.16% Prediction Interval [q0.05, q0.95]", value=f"[{ci[0]}, {ci[1]}] mg/dL", delta=f"Width: {width} mg/dL", delta_color="off")
        with col_m3:
            st.markdown(f"**Clinical Status Tier:**")
            if "Zone A" in czone or "Optimal" in czone:
                st.markdown(f"<span class='badge-zone-a'>{czone}</span>", unsafe_allow_html=True)
            elif "Zone B" in czone or "Post-Prandial" in czone:
                st.markdown(f"<span class='badge-zone-b'>{czone}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='badge-zone-d'>{czone}</span>", unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"**Glycemic Trend:**\n`{trend_text}`")

        # Diagnosis Stratum Specific Callout
        if "Type 1" in diag_clean:
            st.markdown(
                f"<div class='disclaimer-critical'>⚠️ <b>Type 1 Clinical Safety Alert:</b> {strat_conf}</div>",
                unsafe_allow_html=True
            )
        else:
            st.info(f"📌 **Stratum Reliability Note:** {strat_conf}")

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
            st.markdown(f"**Model Version:** `{prediction_result['model_version']}`")
            st.markdown(f"**Validated Scope:** `{prediction_result['validated_scope']}`")

        with col_r2:
            st.markdown(f"**Clinical Guidance:**\n{guidance}")
            st.progress(probs.get("diabetic_risk", 0.0) + probs.get("elevated_risk", 0.0))
            st.caption(f"Risk Probabilities: Healthy: **{probs.get('healthy_risk', 0)*100:.1f}%** | Elevated (Prediabetes): **{probs.get('elevated_risk', 0)*100:.1f}%** | Diabetic: **{probs.get('diabetic_risk', 0)*100:.1f}%**")

        st.markdown(
            f"<div class='disclaimer-banner'>⚠️ <b>Prominent Screening Disclaimer:</b> {conf_note}</div>",
            unsafe_allow_html=True
        )

    # Universal Research Prototype Disclaimer (Always visible, prominent)
    st.markdown(
        "<div class='disclaimer-banner'>"
        "🔬 <b>Research Prototype Disclaimer:</b> This system is an experimental research prototype. "
        "The full-sensor regression model is validated on synthetic multi-modal self-consistency data only. "
        "The tabular risk classifier is validated strictly on CDC NHANES community survey outpatients only (AUROC=0.73). "
        "This software is <b>not a certified medical device</b> and is <b>not a substitute for certified clinical laboratory testing</b> or fingerstick blood glucose monitoring."
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------------------------
    # SECTION 5: Report Generation
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("5. Clinical Report Export")
    pdf_file_path = generate_pdf_report(session_payload, input_payload, prediction_result)

    with open(pdf_file_path, "rb") as f:
        pdf_bytes = f.read()

    st.download_button(
        label="📥 Download Clinical PDF Summary Report",
        data=pdf_bytes,
        file_name=os.path.basename(pdf_file_path),
        mime="application/pdf",
        use_container_width=True
    )
    st.success(f"Report compiled and saved to `{pdf_file_path}`")
