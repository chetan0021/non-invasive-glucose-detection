"""
Debug the feature assembly bug causing constant 95.3 predictions.
Compare with working direct model test to identify the difference.
"""

import sys
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

def load_model_components():
    """Load model, scaler, and metadata"""
    print("="*80)
    print("LOADING MODEL COMPONENTS")
    print("="*80)
    
    models_dir = BASE_DIR / "models"
    
    with open(models_dir / "production_model_full_sensor.pkl", 'rb') as f:
        model = pickle.load(f)
    
    with open(models_dir / "scaler_full_sensor.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    with open(models_dir / "model_metadata_full_sensor.json", 'r') as f:
        metadata = json.load(f)
    
    feature_cols = metadata['feature_columns']
    
    print(f"✅ Model: {metadata['model_type']}")
    print(f"✅ Features: {len(feature_cols)}")
    
    return model, scaler, feature_cols

def test_feature_assembly_methods():
    """Test different feature assembly methods to find the bug"""
    print(f"\n" + "="*80)
    print("TESTING FEATURE ASSEMBLY METHODS")
    print("="*80)
    
    model, scaler, feature_cols = load_model_components()
    
    # Test case that worked in direct model test
    test_input = {
        "saliva_ph": 7.35, "temperature_c": 36.6, "hr_bmp": 66.0,
        "hrv_sdnn": 58.0, "hrv_rmssd": 52.0, "hrv_pnn50": 26.0, "hrv_lf_hf_ratio": 1.10,
        "perfusion_index": 0.81, "pulse_width_ms": 285.0,
        "ppg_raw_dc_baseline": 178000.0, "ppg_raw_ac_p2p": 1450.0,
        "age": 34.0, "bmi": 21.0, "diabetes_diagnosis": "None", "fasting": 1,
        "family_history": 0, "smoking": 0, "gender": "Female"
    }
    
    print("Expected features:")
    for feat in feature_cols:
        print(f"  {feat}")
    
    # Method 1: Working method from direct test
    print(f"\n--- METHOD 1: Working Direct Test Method ---")
    features_method1 = {}
    
    numeric_defaults = {
        "age": 45.0, "bmi": 25.0, "saliva_ph": 7.0, "temperature_c": 36.5, "hr_bpm": 80.0,
        "hrv_sdnn": 40.0, "hrv_rmssd": 35.0, "hrv_pnn50": 15.0, "hrv_lf_hf_ratio": 1.5,
        "perfusion_index": 1.0, "pulse_width_ms": 280.0, "ppg_raw_dc_baseline": 175000.0,
        "ppg_raw_ac_p2p": 1500.0, "ppg_signal_energy": 2000.0, "vpg_max": 2.0, "vpg_min": -1.0,
        "apg_a": 40.0, "apg_b": -40.0, "apg_c": 15.0, "apg_d": -5.0, "apg_e": 15.0,
        "apg_b_a_ratio": -1.0, "apg_aging_index": -1.5
    }
    
    # Fill numeric features
    for feat in feature_cols:
        if feat in numeric_defaults:
            features_method1[feat] = test_input.get(feat, numeric_defaults[feat])
    
    # Categorical features
    if "gender_male" in feature_cols:
        features_method1["gender_male"] = 1 if test_input.get("gender") == "Male" else 0
    
    if "fasting" in feature_cols:
        features_method1["fasting"] = test_input.get("fasting", 1)
    
    if "family_history" in feature_cols:
        features_method1["family_history"] = test_input.get("family_history", 0)
    
    if "smoking" in feature_cols:
        features_method1["smoking"] = test_input.get("smoking", 0)
    
    # Diagnosis features
    diagnosis = test_input.get("diabetes_diagnosis", "None")
    for diag_col in ["diag_none", "diag_prediabetes", "diag_type_1", "diag_type_2"]:
        if diag_col in feature_cols:
            if diag_col == "diag_none":
                features_method1[diag_col] = 1 if diagnosis == "None" else 0
            elif diag_col == "diag_prediabetes":
                features_method1[diag_col] = 1 if diagnosis == "Prediabetes" else 0
            elif diag_col == "diag_type_1":
                features_method1[diag_col] = 1 if diagnosis == "Type 1" else 0
            elif diag_col == "diag_type_2":
                features_method1[diag_col] = 1 if diagnosis == "Type 2" else 0
    
    # Convert to DataFrame with correct column order
    df_method1 = pd.DataFrame([features_method1], columns=feature_cols)
    
    # Scale and predict
    scaled_method1 = scaler.transform(df_method1)
    pred_method1 = model.predict(scaled_method1)[0]
    
    print(f"Method 1 prediction: {pred_method1:.1f}")
    print(f"Method 1 feature sample:")
    for i, feat in enumerate(feature_cols[:10]):  # Show first 10
        print(f"  {feat}: {features_method1[feat]}")
    
    # Method 2: Broken method from stratified report
    print(f"\n--- METHOD 2: Broken Stratified Report Method ---")
    
    # Load test data
    test_path = BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv"
    test_df = pd.read_csv(test_path)
    
    # Take first row as example
    test_row = test_df.iloc[0]
    
    prepared_features = {}
    
    # Same numeric defaults as method 1
    for feat in feature_cols:
        if feat in numeric_defaults:
            prepared_features[feat] = test_df.get(feat, numeric_defaults[feat]).fillna(numeric_defaults[feat])
    
    # Categorical features - this is likely where the bug is
    if "gender_male" in feature_cols:
        prepared_features["gender_male"] = (test_df.get("gender", "F") == "M").astype(int)
    
    if "fasting" in feature_cols:
        prepared_features["fasting"] = test_df.get("fasting", 1).fillna(1)
    
    if "family_history" in feature_cols:
        prepared_features["family_history"] = test_df.get("family_history", 0).fillna(0)
    
    if "smoking" in feature_cols:
        prepared_features["smoking"] = test_df.get("smoking", 0).fillna(0)
    
    # Diagnosis features
    diagnosis_series = test_df.get("diabetes_diagnosis", "None").fillna("None")
    for diag_col in ["diag_none", "diag_prediabetes", "diag_type_1", "diag_type_2"]:
        if diag_col in feature_cols:
            if diag_col == "diag_none":
                prepared_features[diag_col] = (diagnosis_series == "None").astype(int)
            elif diag_col == "diag_prediabetes":
                prepared_features[diag_col] = (diagnosis_series == "Prediabetes").astype(int)
            elif diag_col == "diag_type_1":
                prepared_features[diag_col] = (diagnosis_series == "Type 1").astype(int)
            elif diag_col == "diag_type_2":
                prepared_features[diag_col] = (diagnosis_series == "Type 2").astype(int)
    
    # Convert to DataFrame - this might be the issue
    df_method2 = pd.DataFrame(prepared_features, columns=feature_cols)
    
    print(f"Method 2 DataFrame shape: {df_method2.shape}")
    print(f"Method 2 first few values:")
    for feat in feature_cols[:5]:
        if feat in prepared_features:
            values = prepared_features[feat]
            if hasattr(values, 'iloc'):
                print(f"  {feat}: {values.iloc[:3].tolist()}")
            else:
                print(f"  {feat}: {values}")
    
    # Check if all values are identical (constant bug)
    scaled_method2 = scaler.transform(df_method2)
    pred_method2 = model.predict(scaled_method2)
    
    print(f"Method 2 predictions (first 5): {pred_method2[:5]}")
    print(f"Method 2 unique predictions: {len(np.unique(pred_method2))}")
    
    if len(np.unique(pred_method2)) == 1:
        print(f"🐛 BUG CONFIRMED: Method 2 produces constant predictions ({pred_method2[0]:.1f})")
        
        # Debug: check if features are all identical
        print(f"\nDebugging constant feature values:")
        for feat in feature_cols:
            if feat in prepared_features:
                values = prepared_features[feat]
                if hasattr(values, 'nunique'):
                    unique_count = values.nunique()
                    print(f"  {feat}: {unique_count} unique values")
                    if unique_count == 1:
                        print(f"    -> All values are: {values.iloc[0]}")

def main():
    """Main debugging pipeline"""
    print("🐛 DEBUGGING FEATURE ASSEMBLY BUG")
    print("="*80)
    print("Investigating why stratified report produces constant 95.3 predictions")
    
    test_feature_assembly_methods()

if __name__ == "__main__":
    main()