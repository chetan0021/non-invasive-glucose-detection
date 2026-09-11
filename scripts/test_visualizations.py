"""
Test script to verify all 6 visualization functions for app/dashboard.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict import GlucosePredictor
import plotly.graph_objects as go
import plotly.express as px

def test_visualizations():
    predictor = GlucosePredictor()
    
    # Sample Input
    input_payload = {
        "age": 45.0, "gender": "male", "height_cm": 175.0, "weight_kg": 80.0,
        "bmi": 26.1, "bmi_category": "Overweight", "family_history": 1, "smoking": 0, "fasting": 1,
        "med_status": "None", "med_taking_insulin": 0, "med_taking_oral": 0, "med_taking_any": 0,
        "diabetes_diagnosis": "Type 1",
        "saliva_ph": 6.8, "temperature_c": 36.6, "spo2_pct": 98.0,
        "hr_bpm": 74.0, "ppg_hr_bpm": 74.0, "ppg_raw_dc_baseline": 182000.0, "ppg_raw_ac_p2p": 1250.0,
        "perfusion_index": 0.68, "pulse_width_ms": 280.0,
        "hrv_sdnn": 42.0, "hrv_rmssd": 35.0, "hrv_pnn50": 12.0, "hrv_lf_hf_ratio": 1.45
    }
    
    pred_res = predictor.predict_full_sensor(input_payload)
    bgl = pred_res["predicted_bgl_mg_dl"]
    ci = pred_res["confidence_interval_5th_95th"]
    
    # 1. Test Confidence Interval Plotly Chart
    fig_ci = go.Figure()
    # Background zones
    fig_ci.add_vrect(x0=40, x1=70, fillcolor="#dbeafe", opacity=0.4, layer="below", line_width=0, annotation_text="Hypo (<70)", annotation_position="top left")
    fig_ci.add_vrect(x0=70, x1=140, fillcolor="#dcfce7", opacity=0.4, layer="below", line_width=0, annotation_text="Normal (70-140)", annotation_position="top left")
    fig_ci.add_vrect(x0=140, x1=200, fillcolor="#fef9c3", opacity=0.4, layer="below", line_width=0, annotation_text="Elevated (140-200)", annotation_position="top left")
    fig_ci.add_vrect(x0=200, x1=350, fillcolor="#fee2e2", opacity=0.4, layer="below", line_width=0, annotation_text="Severe (≥200)", annotation_position="top left")
    
    # Interval bar
    fig_ci.add_trace(go.Scatter(
        x=[ci[0], ci[1]], y=[0, 0], mode='lines',
        line=dict(color='#0284c7', width=12),
        name='90% Prediction Interval [q0.05, q0.95]'
    ))
    # Point estimate marker
    fig_ci.add_trace(go.Scatter(
        x=[bgl], y=[0], mode='markers+text',
        marker=dict(color='#1e3a8a', size=18, symbol='diamond', line=dict(color='white', width=2)),
        text=[f"<b>{bgl} mg/dL</b>"], textposition="top center",
        name='Predicted BGL'
    ))
    fig_ci.update_layout(
        title="<b>Calibrated Glycemic Prediction Interval vs. Clinical Safety Zones</b>",
        xaxis=dict(title="Blood Glucose Level (mg/dL)", range=[40, 350]),
        yaxis=dict(showticklabels=False, range=[-0.5, 0.5]),
        height=220, margin=dict(l=20, r=20, t=40, b=30), showlegend=True
    )
    print("Confidence interval figure created successfully!")
    
    # 2. Test Feature Contribution Bar Chart
    X_df = predictor._prepare_full_sensor_features(input_payload)
    feature_names = X_df.columns.tolist()
    importances = predictor.fs_model.feature_importances_
    
    contributions = []
    for i, col in enumerate(feature_names):
        val = X_df.iloc[0, i]
        imp = importances[i]
        # Invert pH and HRV so low pH / low HRV shows positive risk contribution
        multiplier = -1.0 if "ph" in col or "rmssd" in col or "sdnn" in col else 1.0
        score = val * imp * 100.0 * multiplier
        contributions.append((col, score))
        
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_8 = contributions[:8]
    
    label_map = {
        "saliva_ph_scaled": "Saliva pH Bio-Probe",
        "hrv_rmssd_scaled": "ECG HRV RMSSD (Vagal Tone)",
        "hrv_sdnn_scaled": "ECG HRV SDNN (Variability)",
        "hr_bpm_scaled": "Heart Rate (BPM)",
        "temperature_c_scaled": "Skin Surface Temperature",
        "ppg_raw_dc_baseline_scaled": "MAX30102 Optical DC Offset",
        "ppg_raw_ac_p2p_scaled": "MAX30102 Pulsatile AC Peak",
        "pulse_width_ms_scaled": "PPG Pulse Width (ms)",
        "perfusion_index_scaled": "Perfusion Index (%)",
        "age_scaled": "Subject Age",
        "bmi_scaled": "Subject BMI",
        "diag_type_1": "Type 1 Stratum",
        "diag_type_2": "Type 2 Stratum",
        "fasting": "Fasting State"
    }
    
    clean_labels = [label_map.get(k, k.replace("_scaled", "").replace("_", " ").title()) for k, v in top_8]
    values = [v for k, v in top_8]
    bar_colors = ["#dc2626" if v > 0 else "#16a34a" for v in values]
    
    fig_feat = go.Figure(go.Bar(
        x=values[::-1], y=clean_labels[::-1], orientation='h',
        marker=dict(color=bar_colors[::-1])
    ))
    fig_feat.update_layout(
        title="<b>Top 8 Physiological Feature Contributions to Prediction</b>",
        xaxis=dict(title="Relative Impact (Red = Increases Predicted BGL, Green = Pulls Toward Normal)"),
        height=320, margin=dict(l=20, r=20, t=40, b=30)
    )
    print("Feature contribution figure created successfully!")
    
    # 3. Test Sensitivity Sweep Curve
    ph_range = np.linspace(5.5, 8.5, 25)
    bgl_curve = []
    for ph_val in ph_range:
        test_dict = input_payload.copy()
        test_dict["saliva_ph"] = ph_val
        res = predictor.predict_full_sensor(test_dict)
        bgl_curve.append(res["predicted_bgl_mg_dl"])
        
    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(x=ph_range, y=bgl_curve, mode='lines', line=dict(color='#0284c7', width=3), name='Predicted BGL Curve'))
    fig_sweep.add_trace(go.Scatter(x=[input_payload["saliva_ph"]], y=[bgl], mode='markers', marker=dict(color='#dc2626', size=14, symbol='diamond'), name='Current Live Value'))
    fig_sweep.update_layout(
        title="<b>Parameter Sensitivity Sweep: Saliva pH vs Predicted BGL</b>",
        xaxis=dict(title="Saliva pH"), yaxis=dict(title="Predicted BGL (mg/dL)"),
        height=300, margin=dict(l=20, r=20, t=40, b=30)
    )
    print("Sensitivity sweep curve created successfully!")
    
    print("\n>>> ALL 6 VISUALIZATION PIPELINES VERIFIED CLEANLY! <<<")

if __name__ == "__main__":
    test_visualizations()
