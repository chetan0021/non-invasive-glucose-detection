import sys
import os
from pathlib import Path

# Add scripts directory to path to import predict.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from predict import GlucosePredictor

predictor = GlucosePredictor()

hypo_cases = [
    {
        "name": "Case 1: Insulin-induced acute (Sympathetic Dominant)",
        "desc": "Tachycardia, low HRV, peripheral vasoconstriction (sweating/cold).",
        "data": {
            "hr_bpm": 105.0, "hrv_sdnn": 25.0, "hrv_rmssd": 18.0, "hrv_pnn50": 5.0, "hrv_lf_hf_ratio": 3.8,
            "ppg_raw_dc_baseline": 175000.0, "ppg_raw_ac_p2p": 1050.0, "perfusion_index": 0.60,
            "saliva_ph": 7.42, "temperature_c": 35.8, "age": 28.0, "bmi": 21.6,
            "diabetes_diagnosis": "Type 1", "med_taking_insulin": 1,
            "reference_bgl_mg_dl": 58.0
        }
    },
    {
        "name": "Case 2: Exercise-induced (Vasodilated, High HR)",
        "desc": "High HR, very low HRV, high perfusion (post-exercise vasodilation), elevated temp.",
        "data": {
            "hr_bpm": 115.0, "hrv_sdnn": 15.0, "hrv_rmssd": 10.0, "hrv_pnn50": 2.0, "hrv_lf_hf_ratio": 4.5,
            "ppg_raw_dc_baseline": 165000.0, "ppg_raw_ac_p2p": 4500.0, "perfusion_index": 2.7,
            "saliva_ph": 7.35, "temperature_c": 37.5, "age": 28.0, "bmi": 21.6,
            "diabetes_diagnosis": "Type 1", "med_taking_insulin": 1,
            "reference_bgl_mg_dl": 63.0
        }
    },
    {
        "name": "Case 3: Missed meal / gradual onset (Parasympathetic Dominant)",
        "desc": "Normal/low HR, high HRV, normal perfusion.",
        "data": {
            "hr_bpm": 62.0, "hrv_sdnn": 65.0, "hrv_rmssd": 55.0, "hrv_pnn50": 25.0, "hrv_lf_hf_ratio": 1.2,
            "ppg_raw_dc_baseline": 172000.0, "ppg_raw_ac_p2p": 2050.0, "perfusion_index": 1.19,
            "saliva_ph": 7.38, "temperature_c": 36.5, "age": 28.0, "bmi": 21.6,
            "diabetes_diagnosis": "Type 1", "med_taking_insulin": 1,
            "reference_bgl_mg_dl": 65.0
        }
    },
    {
        "name": "Case 4: Nocturnal hypoglycemia (Sleeping)",
        "desc": "Bradycardia, very high HRV, high perfusion (sleep vasodilation).",
        "data": {
            "hr_bpm": 52.0, "hrv_sdnn": 85.0, "hrv_rmssd": 78.0, "hrv_pnn50": 45.0, "hrv_lf_hf_ratio": 0.8,
            "ppg_raw_dc_baseline": 170000.0, "ppg_raw_ac_p2p": 2800.0, "perfusion_index": 1.64,
            "saliva_ph": 7.40, "temperature_c": 36.3, "age": 28.0, "bmi": 21.6,
            "diabetes_diagnosis": "Type 1", "med_taking_insulin": 1,
            "reference_bgl_mg_dl": 55.0
        }
    },
    {
        "name": "Case 5: Autonomic Unawareness",
        "desc": "Normal HR, normal HRV, normal perfusion (clinically difficult case).",
        "data": {
            "hr_bpm": 75.0, "hrv_sdnn": 45.0, "hrv_rmssd": 40.0, "hrv_pnn50": 15.0, "hrv_lf_hf_ratio": 1.8,
            "ppg_raw_dc_baseline": 172000.0, "ppg_raw_ac_p2p": 2200.0, "perfusion_index": 1.28,
            "saliva_ph": 7.35, "temperature_c": 36.6, "age": 28.0, "bmi": 21.6,
            "diabetes_diagnosis": "Type 1", "med_taking_insulin": 1,
            "reference_bgl_mg_dl": 59.0
        }
    }
]

print("="*80)
print("HYPOGLYCEMIA PHYSIOLOGICAL DIVERSITY TEST")
print("="*80)

for case in hypo_cases:
    print(f"\n{case['name']}")
    print(f"Description: {case['desc']}")
    res = predictor.predict(case['data'])
    pred = res['predicted_bgl_mg_dl']
    ref = case['data']['reference_bgl_mg_dl']
    zone = res['clarke_zone']
    print(f"Reference: {ref} mg/dL | Predicted: {pred} mg/dL")
    print(f"Error: {pred - ref:.1f} mg/dL")
    print(f"Zone: {zone}")
