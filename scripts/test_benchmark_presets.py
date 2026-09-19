"""
Test the 5 original benchmark presets through predict.py
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path to import predict.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from predict import GlucosePredictor

predictor = GlucosePredictor()

benchmark_cases = [
    {
        "name": "Healthy Adult",
        "reference": 88.0,
        "data": {
            "saliva_ph": 7.35, "temperature_c": 36.6, "hr_bpm": 66.0,
            "hrv_sdnn": 58.0, "hrv_rmssd": 52.0, "hrv_pnn50": 26.0, "hrv_lf_hf_ratio": 1.10,
            "perfusion_index": 0.81, "pulse_width_ms": 285.0,
            "ppg_raw_dc_baseline": 178000.0, "ppg_raw_ac_p2p": 1450.0,
            "age": 34.0, "bmi": 21.0, "diabetes_diagnosis": "None", "fasting": 1,
            "family_history": 0, "smoking": 0, "gender": "Female"
        }
    },
    {
        "name": "Prediabetes",
        "reference": 114.0,
        "data": {
            "saliva_ph": 6.95, "temperature_c": 36.5, "hr_bpm": 76.0,
            "hrv_sdnn": 36.0, "hrv_rmssd": 28.0, "hrv_pnn50": 10.0, "hrv_lf_hf_ratio": 1.65,
            "perfusion_index": 0.68, "pulse_width_ms": 275.0,
            "ppg_raw_dc_baseline": 172000.0, "ppg_raw_ac_p2p": 1180.0,
            "age": 52.0, "bmi": 27.4, "diabetes_diagnosis": "Prediabetes", "fasting": 1,
            "family_history": 1, "smoking": 0, "gender": "Male"
        }
    },
    {
        "name": "Type 2 Diabetes",
        "reference": 172.0,
        "data": {
            "saliva_ph": 6.60, "temperature_c": 36.9, "hr_bpm": 84.0,
            "hrv_sdnn": 26.0, "hrv_rmssd": 18.0, "hrv_pnn50": 6.0, "hrv_lf_hf_ratio": 2.20,
            "perfusion_index": 1.12, "pulse_width_ms": 298.0,
            "ppg_raw_dc_baseline": 164000.0, "ppg_raw_ac_p2p": 1850.0,
            "age": 59.0, "bmi": 29.8, "diabetes_diagnosis": "Type 2", "fasting": 0,
            "family_history": 1, "smoking": 1, "gender": "Male"
        }
    },
    {
        "name": "Severe Hyperglycemia",
        "reference": 265.0,
        "data": {
            "saliva_ph": 6.15, "temperature_c": 37.2, "hr_bpm": 98.0,
            "hrv_sdnn": 14.0, "hrv_rmssd": 8.0, "hrv_pnn50": 1.0, "hrv_lf_hf_ratio": 3.80,
            "perfusion_index": 1.57, "pulse_width_ms": 325.0,
            "ppg_raw_dc_baseline": 188000.0, "ppg_raw_ac_p2p": 2950.0,
            "age": 48.0, "bmi": 32.4, "diabetes_diagnosis": "Type 1", "fasting": 0,
            "family_history": 1, "smoking": 1, "gender": "Female"
        }
    },
    {
        "name": "Hypoglycemia",
        "reference": 62.0,
        "data": {
            "saliva_ph": 7.42, "temperature_c": 36.1, "hr_bpm": 88.0,
            "hrv_sdnn": 48.0, "hrv_rmssd": 44.0, "hrv_pnn50": 20.0, "hrv_lf_hf_ratio": 1.30,
            "perfusion_index": 0.66, "pulse_width_ms": 260.0,
            "ppg_raw_dc_baseline": 172000.0, "ppg_raw_ac_p2p": 1150.0,
            "age": 28.0, "bmi": 21.6, "diabetes_diagnosis": "Type 1", "fasting": 1,
            "family_history": 0, "smoking": 0, "gender": "Male"
        }
    }
]

print("="*80)
print("BENCHMARK PRESET VALIDATION")
print("="*80)

for case in benchmark_cases:
    res = predictor.predict(case['data'])
    pred = res['predicted_bgl_mg_dl']
    ref = case['reference']
    zone = res['clarke_zone']
    print(f"\n{case['name']}")
    print(f"Reference: {ref} mg/dL | Predicted: {pred} mg/dL")
    print(f"Error: {pred - ref:.1f} mg/dL")
    print(f"Zone: {zone}")