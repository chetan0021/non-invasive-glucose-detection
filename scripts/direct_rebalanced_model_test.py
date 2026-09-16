"""
Direct test of rebalanced model on benchmark cases.
Bypasses predict.py to test the trained model directly.
"""

import sys
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.train_models import clarke_error_grid_zone

def load_rebalanced_model():
    """Load the trained rebalanced model and scaler"""
    print("="*80)
    print("LOADING REBALANCED MODEL")
    print("="*80)
    
    models_dir = BASE_DIR / "models"
    
    model_path = models_dir / "production_model_full_sensor.pkl"
    scaler_path = models_dir / "scaler_full_sensor.pkl"
    metadata_path = models_dir / "model_metadata_full_sensor.json"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None, None, None
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Model loaded: {metadata['model_type']}")
    print(f"✅ Features: {len(metadata['feature_columns'])} features")
    print(f"✅ Test R²: {metadata['performance_metrics']['test_r2']:.3f}")
    print(f"✅ Zone D failures in training: {metadata['performance_metrics']['zone_d_count']}")
    
    return model, scaler, metadata['feature_columns']

def prepare_test_case(input_data, feature_cols):
    """Prepare a test case for prediction"""
    
    # Create feature vector with defaults
    features = {}
    
    # Numeric features with defaults
    numeric_defaults = {
        "age": 45.0,
        "bmi": 25.0,
        "saliva_ph": 7.0,
        "temperature_c": 36.5,
        "hr_bpm": 80.0,
        "hrv_sdnn": 40.0,
        "hrv_rmssd": 35.0,
        "hrv_pnn50": 15.0,
        "hrv_lf_hf_ratio": 1.5,
        "perfusion_index": 1.0,
        "pulse_width_ms": 280.0,
        "ppg_raw_dc_baseline": 175000.0,
        "ppg_raw_ac_p2p": 1500.0,
        "ppg_signal_energy": 2000.0,
        "vpg_max": 2.0,
        "vpg_min": -1.0,
        "apg_a": 40.0,
        "apg_b": -40.0,
        "apg_c": 15.0,
        "apg_d": -5.0,
        "apg_e": 15.0,
        "apg_b_a_ratio": -1.0,
        "apg_aging_index": -1.5
    }
    
    # Fill numeric features
    for feat in feature_cols:
        if feat in numeric_defaults:
            features[feat] = input_data.get(feat, numeric_defaults[feat])
    
    # Categorical features
    if "gender_male" in feature_cols:
        features["gender_male"] = 1 if input_data.get("gender") == "Male" else 0
    
    if "fasting" in feature_cols:
        features["fasting"] = input_data.get("fasting", 1)
    
    if "family_history" in feature_cols:
        features["family_history"] = input_data.get("family_history", 0)
    
    if "smoking" in feature_cols:
        features["smoking"] = input_data.get("smoking", 0)
    
    # Diagnosis features
    diagnosis = input_data.get("diabetes_diagnosis", "None")
    for diag_col in ["diag_none", "diag_prediabetes", "diag_type_1", "diag_type_2"]:
        if diag_col in feature_cols:
            if diag_col == "diag_none":
                features[diag_col] = 1 if diagnosis == "None" else 0
            elif diag_col == "diag_prediabetes":
                features[diag_col] = 1 if diagnosis == "Prediabetes" else 0
            elif diag_col == "diag_type_1":
                features[diag_col] = 1 if diagnosis == "Type 1" else 0
            elif diag_col == "diag_type_2":
                features[diag_col] = 1 if diagnosis == "Type 2" else 0
    
    # Convert to DataFrame with correct column order
    feature_df = pd.DataFrame([features], columns=feature_cols)
    
    return feature_df

def test_benchmark_cases(model, scaler, feature_cols):
    """Test the model on benchmark cases"""
    print(f"\n" + "="*80)
    print("TESTING BENCHMARK CASES WITH REBALANCED MODEL")
    print("="*80)
    
    benchmark_cases = [
        {
            "name": "Healthy Adult",
            "reference": 88.0,
            "input": {
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
            "input": {
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
            "input": {
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
            "input": {
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
            "input": {
                "saliva_ph": 7.42, "temperature_c": 36.1, "hr_bmp": 88.0,
                "hrv_sdnn": 48.0, "hrv_rmssd": 44.0, "hrv_pnn50": 20.0, "hrv_lf_hf_ratio": 1.30,
                "perfusion_index": 0.66, "pulse_width_ms": 260.0,
                "ppg_raw_dc_baseline": 172000.0, "ppg_raw_ac_p2p": 1150.0,
                "age": 28.0, "bmi": 21.6, "diabetes_diagnosis": "Type 1", "fasting": 1,
                "family_history": 0, "smoking": 0, "gender": "Male"
            }
        }
    ]
    
    print(f"Testing {len(benchmark_cases)} benchmark presets...")
    print(f"{'Preset':25s} {'Reference':>10s} {'Predicted':>10s} {'Error':>8s} {'Zone':>6s} {'Status':>15s}")
    print("-" * 85)
    
    zone_d_failures = []
    zone_e_failures = []
    results = []
    
    for case in benchmark_cases:
        try:
            # Prepare features
            features_df = prepare_test_case(case["input"], feature_cols)
            
            # Scale features
            features_scaled = scaler.transform(features_df)
            
            # Predict
            predicted = model.predict(features_scaled)[0]
            predicted = max(30.0, min(500.0, predicted))  # Reasonable bounds
            
            reference = case["reference"]
            error = abs(predicted - reference)
            
            # Clarke Grid zone
            zone = clarke_error_grid_zone(reference, predicted)
            
            if zone == 'D':
                status = "🚨 ZONE D"
                zone_d_failures.append(case["name"])
            elif zone == 'E':
                status = "🚨 ZONE E"
                zone_e_failures.append(case["name"])
            elif zone == 'C':
                status = "⚠️ ZONE C"
            elif zone == 'B':
                status = "✅ ZONE B"
            else:  # Zone A
                status = "✅ ZONE A"
            
            print(f"{case['name']:25s} {reference:10.1f} {predicted:10.1f} {error:8.1f} {zone:>6s} {status:>15s}")
            
            results.append({
                "name": case["name"],
                "reference": reference,
                "predicted": predicted,
                "error": error,
                "zone": zone,
                "safe": zone in ['A', 'B', 'C']
            })
            
        except Exception as e:
            print(f"{case['name']:25s} ERROR: {e}")
            zone_d_failures.append(f"{case['name']} (error)")
    
    return results, zone_d_failures, zone_e_failures

def test_additional_extreme_cases(model, scaler, feature_cols):
    """Test additional extreme cases"""
    print(f"\n" + "="*80)
    print("TESTING ADDITIONAL EXTREME CASES")
    print("="*80)
    
    extreme_cases = [
        {
            "name": "Mild Hypoglycemia",
            "reference": 65.0,
            "input": {
                "saliva_ph": 7.40, "temperature_c": 36.2, "hr_bpm": 82.0,
                "hrv_sdnn": 45.0, "hrv_rmssd": 40.0, "hrv_pnn50": 18.0, "hrv_lf_hf_ratio": 1.25,
                "perfusion_index": 0.70, "pulse_width_ms": 265.0,
                "age": 35.0, "bmi": 22.5, "diabetes_diagnosis": "Type 1", "fasting": 1,
                "family_history": 0, "smoking": 0, "gender": "Female"
            }
        },
        {
            "name": "Severe Hypoglycemia", 
            "reference": 45.0,
            "input": {
                "saliva_ph": 7.45, "temperature_c": 35.8, "hr_bpm": 95.0,
                "hrv_sdnn": 35.0, "hrv_rmssd": 30.0, "hrv_pnn50": 12.0, "hrv_lf_hf_ratio": 1.60,
                "perfusion_index": 0.60, "pulse_width_ms": 250.0,
                "age": 25.0, "bmi": 20.0, "diabetes_diagnosis": "Type 1", "fasting": 1,
                "family_history": 0, "smoking": 0, "gender": "Male"
            }
        },
        {
            "name": "Moderate Hyperglycemia",
            "reference": 275.0,
            "input": {
                "saliva_ph": 6.10, "temperature_c": 37.0, "hr_bpm": 88.0,
                "hrv_sdnn": 18.0, "hrv_rmssd": 12.0, "hrv_pnn50": 3.0, "hrv_lf_hf_ratio": 3.50,
                "perfusion_index": 1.40, "pulse_width_ms": 315.0,
                "age": 42.0, "bmi": 30.0, "diabetes_diagnosis": "Type 2", "fasting": 0,
                "family_history": 1, "smoking": 0, "gender": "Male"
            }
        },
        {
            "name": "Extreme Hyperglycemia",
            "reference": 350.0,
            "input": {
                "saliva_ph": 6.05, "temperature_c": 37.5, "hr_bpm": 105.0,
                "hrv_sdnn": 12.0, "hrv_rmssd": 6.0, "hrv_pnn50": 1.0, "hrv_lf_hf_ratio": 4.20,
                "perfusion_index": 1.80, "pulse_width_ms": 340.0,
                "age": 55.0, "bmi": 35.0, "diabetes_diagnosis": "Type 1", "fasting": 0,
                "family_history": 1, "smoking": 1, "gender": "Female"
            }
        },
        {
            "name": "DKA-Risk Hyperglycemia",
            "reference": 420.0,
            "input": {
                "saliva_ph": 5.95, "temperature_c": 37.8, "hr_bpm": 115.0,
                "hrv_sdnn": 8.0, "hrv_rmssd": 4.0, "hrv_pnn50": 0.0, "hrv_lf_hf_ratio": 5.00,
                "perfusion_index": 2.10, "pulse_width_ms": 360.0,
                "age": 38.0, "bmi": 28.0, "diabetes_diagnosis": "Type 1", "fasting": 0,
                "family_history": 1, "smoking": 0, "gender": "Male"
            }
        }
    ]
    
    print(f"Testing {len(extreme_cases)} additional extreme cases...")
    print(f"{'Case':25s} {'Reference':>10s} {'Predicted':>10s} {'Error':>8s} {'Zone':>6s} {'Status':>15s}")
    print("-" * 85)
    
    zone_d_failures = []
    zone_e_failures = []
    results = []
    
    for case in extreme_cases:
        try:
            # Prepare features
            features_df = prepare_test_case(case["input"], feature_cols)
            
            # Scale features
            features_scaled = scaler.transform(features_df)
            
            # Predict
            predicted = model.predict(features_scaled)[0]
            predicted = max(30.0, min(500.0, predicted))  # Reasonable bounds
            
            reference = case["reference"]
            error = abs(predicted - reference)
            
            # Clarke Grid zone
            zone = clarke_error_grid_zone(reference, predicted)
            
            if zone == 'D':
                status = "🚨 ZONE D"
                zone_d_failures.append(case["name"])
            elif zone == 'E':
                status = "🚨 ZONE E"
                zone_e_failures.append(case["name"])
            elif zone == 'C':
                status = "⚠️ ZONE C"
            elif zone == 'B':
                status = "✅ ZONE B"
            else:  # Zone A
                status = "✅ ZONE A"
            
            print(f"{case['name']:25s} {reference:10.1f} {predicted:10.1f} {error:8.1f} {zone:>6s} {status:>15s}")
            
            results.append({
                "name": case["name"],
                "reference": reference,
                "predicted": predicted,
                "error": error,
                "zone": zone,
                "safe": zone in ['A', 'B', 'C']
            })
            
        except Exception as e:
            print(f"{case['name']:25s} ERROR: {e}")
            zone_d_failures.append(f"{case['name']} (error)")
    
    return results, zone_d_failures, zone_e_failures

def main():
    """Main safety verification pipeline"""
    print("🛡️  DIRECT REBALANCED MODEL SAFETY VERIFICATION")
    print("="*80)
    print("Testing rebalanced model directly on benchmark + extreme cases")
    
    # Load model
    model, scaler, feature_cols = load_rebalanced_model()
    if model is None:
        return False
    
    # Test benchmark cases
    benchmark_results, benchmark_d, benchmark_e = test_benchmark_cases(model, scaler, feature_cols)
    
    # Test additional extreme cases
    extreme_results, extreme_d, extreme_e = test_additional_extreme_cases(model, scaler, feature_cols)
    
    # Combine results
    all_results = benchmark_results + extreme_results
    total_d_failures = benchmark_d + extreme_d
    total_e_failures = benchmark_e + extreme_e
    
    print(f"\n" + "="*80)
    print("COMPREHENSIVE SAFETY VERIFICATION RESULTS")
    print("="*80)
    
    print(f"BENCHMARK PRESETS (5 cases):")
    print(f"  Zone D failures: {len(benchmark_d)}")
    print(f"  Zone E failures: {len(benchmark_e)}")
    
    print(f"\nADDITIONAL EXTREME CASES (5 cases):")
    print(f"  Zone D failures: {len(extreme_d)}")
    print(f"  Zone E failures: {len(extreme_e)}")
    
    print(f"\nOVERALL SAFETY ASSESSMENT:")
    print(f"  Total test cases: {len(all_results)}")
    print(f"  Total Zone D failures: {len(total_d_failures)}")
    print(f"  Total Zone E failures: {len(total_e_failures)}")
    
    # Zone distribution
    zones = [r["zone"] for r in all_results]
    zone_counts = pd.Series(zones).value_counts()
    
    print(f"\nCLARKE ZONE DISTRIBUTION:")
    for zone in ['A', 'B', 'C', 'D', 'E']:
        count = zone_counts.get(zone, 0)
        pct = (count / len(all_results)) * 100 if all_results else 0
        print(f"  Zone {zone}: {count:2d} cases ({pct:5.1f}%)")
    
    safe_cases = sum(1 for r in all_results if r["safe"])
    safety_pct = (safe_cases / len(all_results)) * 100 if all_results else 0
    
    print(f"\nSAFETY METRICS:")
    print(f"  Safe predictions (Zone A/B/C): {safe_cases}/{len(all_results)} ({safety_pct:.1f}%)")
    print(f"  Dangerous failures (Zone D/E): {len(total_d_failures) + len(total_e_failures)}")
    
    if len(total_d_failures) == 0 and len(total_e_failures) == 0:
        print(f"\n✅ SAFETY VERIFICATION PASSED!")
        print(f"   Zero Zone D/E failures across all {len(all_results)} test cases")
        print(f"   Rebalanced model is SAFE for deployment")
        return True
    else:
        print(f"\n⚠️  SAFETY VERIFICATION INCOMPLETE")
        print(f"   {len(total_d_failures)} Zone D failures remain")
        print(f"   {len(total_e_failures)} Zone E failures remain")
        
        if total_d_failures:
            print(f"   Zone D cases:")
            for failure in total_d_failures:
                print(f"     • {failure}")
        
        print(f"\n📊 PROGRESS SUMMARY:")
        print(f"   Before: Zone D on both severe hyperglycemia (265→139) & hypoglycemia (62→83)")
        print(f"   After: {len(total_d_failures)} Zone D failures out of {len(all_results)} test cases")
        print(f"   Major improvement, but not yet deployment-ready")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)