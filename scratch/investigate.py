import sys
sys.path.append(".")
import pandas as pd
import json
from app.dashboard import BENCHMARK_PRESETS, get_predictor

print("=== CI Investigation across Presets ===")
predictor = get_predictor()

for name, raw_data in BENCHMARK_PRESETS.items():
    if raw_data is None or not raw_data.get("has_sensor", False):
        continue
    
    # Map raw preset data to what predict.py actually expects
    data = raw_data.copy()
    data["family_history"] = 1 if "Yes" in str(data.get("family_history", "")) else 0
    data["smoking"] = 1 if "Smoker (1)" in str(data.get("smoking_status", "")) else 0
    data["hypertension"] = 1 if data.get("hypertension") == "Yes" else 0
    data["high_cholesterol"] = 1 if data.get("high_cholesterol") == "Yes" else 0
    data["gestational_diabetes"] = data.get("gestational_diabetes", "No")
    
    # Convert physical activity level
    pal = data.get("physical_activity_level", "")
    if "Vigorous" in pal: data["physical_activity_level"] = "active"
    elif "Sedentary" in pal: data["physical_activity_level"] = "sedentary"
    else: data["physical_activity_level"] = "moderate"
    
    # Fasting
    data["fasting"] = 1 if "Fasting (≥8h)" in data.get("fasting_status", "") else 0
    
    # Diabetes diagnosis mapping
    diag = data.get("diagnosis_option", "")
    if "Healthy" in diag: data["diabetes_diagnosis"] = "None"
    elif "Prediabetes" in diag: data["diabetes_diagnosis"] = "Prediabetes"
    elif "Type 1" in diag: data["diabetes_diagnosis"] = "Type 1"
    elif "Type 2" in diag: data["diabetes_diagnosis"] = "Type 2"
    else: data["diabetes_diagnosis"] = "None"
    
    res = predictor.predict(data)
    if "predicted_bgl_mg_dl" in res:
        ci = res["confidence_interval_5th_95th"]
        width = res["interval_width_mg_dl"]
        safe_name = name.split(':')[0].encode('ascii', 'ignore').decode()
        print(f"Preset: {safe_name}")
        print(f"  Point: {res['predicted_bgl_mg_dl']}, CI: [{ci[0]}, {ci[1]}], Width: {width}")
