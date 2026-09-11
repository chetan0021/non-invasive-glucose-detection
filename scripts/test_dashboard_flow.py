"""
Verification script for Phase 8 Dashboard components:
1. Tests GlucosePredictor full-sensor inference
2. Tests GlucosePredictor risk-band inference
3. Tests generate_pdf_report from app/dashboard.py for both modes
4. Tests log_submission to manual_test_log.csv for both modes
"""

import sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict import GlucosePredictor
from app.dashboard import generate_pdf_report, append_to_audit_log

def test_full_sensor_pipeline():
    print("--- Testing Full-Sensor Pipeline ---")
    predictor = GlucosePredictor()
    
    session_data = {
        "full_name": "Test Subject Alpha",
        "session_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "Post-prandial 2-hour multi-modal evaluation."
    }
    
    input_data = {
        "age": 45,
        "gender": "male",
        "height_cm": 175.0,
        "weight_kg": 80.0,
        "bmi": 26.1,
        "bmi_category": "Overweight",
        "family_history": 1,
        "smoking_status": 0,
        "fasting": 0,
        "med_status": "metformin_only",
        "diabetes_diagnosis": "Type 2",
        # Sensor inputs
        "saliva_ph": 6.8,
        "hr_bpm": 74.0,
        "spo2_pct": 98.0,
        "ppg_raw_dc_baseline": 182000.0,
        "ppg_raw_ac_p2p": 12500.0,
        "pulse_width_ms": 280.0,
        "signal_energy": 45000.0,
        "perfusion_index": 4.2,
        "temperature_c": 36.6,
        "vpg_max_slope": 1400.0,
        "apg_a": 1.0,
        "apg_b": -0.72,
        "apg_c": 0.28,
        "apg_d": -0.15,
        "apg_e": 0.10,
        "hrv_sdnn": 42.0,
        "hrv_rmssd": 35.0,
        "hrv_pnn50": 12.0,
        "hrv_lf_hf_ratio": 1.45,
        "previous_bgl": 140.0,
        "previous_timestamp": "2026-09-11 06:00:00"
    }
    
    pred_res = predictor.predict_full_sensor(input_data)
    print("Full-Sensor Prediction Output:")
    for k, v in pred_res.items():
        print(f"  {k}: {v}")
    
    assert any(z in pred_res["clarke_zone"] for z in ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"])
    
    # Test PDF Generation
    pdf_path = generate_pdf_report(session_data, input_data, pred_res)
    print(f"PDF Generated: {pdf_path} (exists: {Path(pdf_path).exists()})")
    assert Path(pdf_path).exists() and Path(pdf_path).stat().st_size > 1000
    
    # Test Logging
    append_to_audit_log(session_data, input_data, pred_res)
    print("Logged to manual_test_log.csv successfully.")

def test_risk_band_pipeline():
    print("\n--- Testing Tabular Risk-Band Pipeline ---")
    predictor = GlucosePredictor()
    
    session_data = {
        "full_name": "Test Subject Beta",
        "session_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "Routine demographic community outpatient screening."
    }
    
    input_data = {
        "age": 58,
        "gender": "female",
        "height_cm": 162.0,
        "weight_kg": 88.0,
        "bmi": 33.5,
        "bmi_category": "Obesity Class I",
        "family_history": 1,
        "smoking_status": 1,
        "fasting": 1,
        "med_status": "none",
        "diabetes_diagnosis": "Not Diagnosed / Unknown"
    }
    
    pred_res = predictor.predict_risk_band(input_data)
    print("Risk-Band Prediction Output:")
    for k, v in pred_res.items():
        print(f"  {k}: {v}")
        
    assert "risk_band" in pred_res
    assert pred_res["risk_band"] in ["lower_risk", "elevated_risk_consult_recommended"]
    
    # Test PDF Generation
    pdf_path = generate_pdf_report(session_data, input_data, pred_res)
    print(f"PDF Generated: {pdf_path} (exists: {Path(pdf_path).exists()})")
    assert Path(pdf_path).exists() and Path(pdf_path).stat().st_size > 1000
    
    # Test Logging
    append_to_audit_log(session_data, input_data, pred_res)
    print("Logged to manual_test_log.csv successfully.")

if __name__ == "__main__":
    test_full_sensor_pipeline()
    test_risk_band_pipeline()
    print("\n>>> ALL PHASE 8 BACKEND TESTS PASSED SUCCESSFULLY! <<<")
