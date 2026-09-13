"""
Deep investigation of feature preprocessing mismatch between training/calibration 
and predict.py serving path.

This is NOT a secondary issue - if features don't match, both point predictions 
and confidence intervals are potentially wrong.
"""

import sys
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict import GlucosePredictor

def load_training_preprocessing_artifacts():
    """Load the scalers and processors used during training"""
    print("="*80)
    print("LOADING TRAINING PREPROCESSING ARTIFACTS")
    print("="*80)
    
    artifacts = {}
    
    # Look for scaler files
    models_dir = BASE_DIR / "models"
    for scaler_file in models_dir.glob("*scaler*.pkl"):
        scaler_name = scaler_file.stem
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
        artifacts[scaler_name] = scaler
        print(f"Found scaler: {scaler_name}")
        print(f"  Type: {type(scaler)}")
        if hasattr(scaler, 'feature_names_in_'):
            print(f"  Features: {len(scaler.feature_names_in_)} features")
        if hasattr(scaler, 'scale_'):
            print(f"  Scale factors: {scaler.scale_[:3]}... (first 3)")
    
    # Load feature manifest
    manifest_path = BASE_DIR / "reports" / "features_manifest_full_sensor.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    artifacts["manifest"] = manifest
    print(f"\nFeature manifest loaded:")
    print(f"  Scaled numeric features: {len(manifest['scaled_numeric_features'])}")
    print(f"  Categorical/binary features: {len(manifest['categorical_and_binary_features'])}")
    
    return artifacts

def get_training_features_for_sample(artifacts, test_df, sample_idx=0):
    """Get the exact feature values from training data"""
    print(f"\n" + "="*80)
    print(f"TRAINING DATA FEATURES (SAMPLE {sample_idx})")
    print("="*80)
    
    sample_row = test_df.iloc[sample_idx]
    manifest = artifacts["manifest"]
    feature_cols = manifest["scaled_numeric_features"] + manifest["categorical_and_binary_features"]
    
    # This is the "ground truth" - exactly what the model was trained/calibrated on
    training_features = sample_row[feature_cols]
    
    print(f"Sample metadata:")
    print(f"  True BGL: {sample_row['bgl_mg_dl']:.1f} mg/dL")
    print(f"  Diagnosis: {sample_row.get('diabetes_diagnosis', 'Unknown')}")
    
    print(f"\nTraining feature values:")
    for i, (feature_name, value) in enumerate(training_features.items()):
        if i < 20:  # Show first 20 features
            print(f"  {feature_name:30s}: {value:10.6f}")
        elif i == 20:
            print(f"  ... ({len(training_features) - 20} more features)")
    
    return training_features, sample_row

def get_predict_py_features_for_sample(sample_row):
    """Get features as computed by predict.py for the same sample"""
    print(f"\n" + "="*80)
    print("PREDICT.PY FEATURES (SAME SAMPLE)")
    print("="*80)
    
    # Convert sample back to "raw" input format
    # This is the key challenge - we need to reverse-engineer what the raw inputs were
    input_dict = {
        # Direct mappings where possible
        "saliva_ph": float(sample_row.get("saliva_ph", 7.25)),
        "temperature_c": float(sample_row.get("temperature_c", 36.6)),
        "hr_bmp": float(sample_row.get("hr_bpm", 72.0)),
        "hrv_sdnn": float(sample_row.get("hrv_sdnn", 45.0)),
        "hrv_rmssd": float(sample_row.get("hrv_rmssd", 35.0)),
        "hrv_pnn50": float(sample_row.get("hrv_pnn50", 20.0)),
        "hrv_lf_hf_ratio": float(sample_row.get("hrv_lf_hf_ratio", 1.5)),
        "perfusion_index": float(sample_row.get("perfusion_index", 0.8)),
        "pulse_width_ms": float(sample_row.get("pulse_width_ms", 280.0)),
        
        # PPG raw values - these might be the source of the problem
        "ppg_raw_dc_baseline": float(sample_row.get("ppg_raw_dc_baseline", 175000.0)),
        "ppg_raw_ac_p2p": float(sample_row.get("ppg_raw_ac_p2p", 1200.0)),
        
        # Demographics
        "age": float(sample_row.get("age", 45.0)),
        "bmi": float(sample_row.get("bmi", 26.0)),
        "diabetes_diagnosis": str(sample_row.get("diabetes_diagnosis", "None")),
        "fasting": int(sample_row.get("fasting", 1)),
        "gender": "Male" if sample_row.get("gender_male", 1) == 1 else "Female",
        "family_history": int(sample_row.get("family_history", 0)),
        "smoking": int(sample_row.get("smoking", 0)),
        "med_taking_insulin": int(sample_row.get("med_taking_insulin", 0)),
        "med_taking_oral": int(sample_row.get("med_taking_oral", 0))
    }
    
    print("Input dict for predict.py:")
    for key, value in input_dict.items():
        print(f"  {key:25s}: {value}")
    
    # Get features via predict.py preprocessing
    predictor = GlucosePredictor()
    
    try:
        # Use internal method to get just the feature matrix
        predict_features_df = predictor._prepare_full_sensor_features(input_dict)
        predict_features = predict_features_df.iloc[0]  # First (only) row
        
        print(f"\nPredict.py feature values:")
        for i, (feature_name, value) in enumerate(predict_features.items()):
            if i < 20:  # Show first 20 features
                print(f"  {feature_name:30s}: {value:10.6f}")
            elif i == 20:
                print(f"  ... ({len(predict_features) - 20} more features)")
        
        return predict_features, input_dict
        
    except Exception as e:
        print(f"❌ Error getting predict.py features: {e}")
        import traceback
        traceback.print_exc()
        return None, input_dict

def compare_feature_values(training_features, predict_features):
    """Compare feature values side by side to find differences"""
    print(f"\n" + "="*80)
    print("SIDE-BY-SIDE FEATURE COMPARISON")
    print("="*80)
    
    if predict_features is None:
        print("❌ Cannot compare - predict.py features failed to generate")
        return {}
    
    # Find common features
    training_set = set(training_features.index)
    predict_set = set(predict_features.index)
    
    common_features = training_set & predict_set
    training_only = training_set - predict_set
    predict_only = predict_set - training_set
    
    print(f"Feature alignment:")
    print(f"  Common features: {len(common_features)}")
    print(f"  Training only: {len(training_only)}")
    print(f"  Predict.py only: {len(predict_only)}")
    
    if training_only:
        print(f"  Missing in predict.py: {list(training_only)[:5]}...")
    if predict_only:
        print(f"  Extra in predict.py: {list(predict_only)[:5]}...")
    
    # Compare common features
    differences = []
    
    print(f"\nFeature value comparison (showing differences >0.001):")
    print(f"{'Feature Name':35s} {'Training':>12s} {'Predict.py':>12s} {'Difference':>12s}")
    print("-" * 75)
    
    for feature in sorted(common_features):
        train_val = training_features[feature]
        pred_val = predict_features[feature]
        diff = abs(train_val - pred_val)
        
        if diff > 0.001:  # Significant difference
            differences.append({
                "feature": feature,
                "training": train_val,
                "predict": pred_val,
                "diff": diff
            })
            print(f"{feature:35s} {train_val:12.6f} {pred_val:12.6f} {diff:12.6f}")
    
    if not differences:
        print("✅ All common features match within 0.001 tolerance")
    else:
        print(f"\n⚠️  Found {len(differences)} features with significant differences")
        
        # Focus on the biggest differences
        differences.sort(key=lambda x: x["diff"], reverse=True)
        print(f"\nTop 10 largest differences:")
        for i, diff_info in enumerate(differences[:10]):
            print(f"  {i+1:2d}. {diff_info['feature']:30s}: diff={diff_info['diff']:8.6f}")
    
    return differences

def investigate_ppg_signal_energy_mismatch(training_features, predict_features, sample_row, input_dict):
    """Deep dive into ppg_signal_energy_scaled mismatch"""
    print(f"\n" + "="*80)
    print("PPG_SIGNAL_ENERGY MISMATCH INVESTIGATION")
    print("="*80)
    
    if predict_features is None:
        print("❌ Cannot investigate - predict.py features not available")
        return
    
    train_energy = training_features.get("ppg_signal_energy_scaled", None)
    pred_energy = predict_features.get("ppg_signal_energy_scaled", None)
    
    if train_energy is None or pred_energy is None:
        print("❌ ppg_signal_energy_scaled not found in one or both feature sets")
        return
    
    print(f"PPG Signal Energy comparison:")
    print(f"  Training data: {train_energy:10.6f}")
    print(f"  Predict.py:    {pred_energy:10.6f}")
    print(f"  Difference:    {abs(train_energy - pred_energy):10.6f}")
    
    # Try to reverse engineer what the raw values should be
    print(f"\nRaw PPG values from input:")
    print(f"  DC baseline: {input_dict.get('ppg_raw_dc_baseline', 'N/A')}")
    print(f"  AC p2p:     {input_dict.get('ppg_raw_ac_p2p', 'N/A')}")
    
    # Check if there are unscaled versions in the training data
    raw_features = [col for col in sample_row.index if "ppg" in col.lower() and "scaled" not in col]
    print(f"\nRaw PPG features in training data:")
    for feature in raw_features[:10]:  # Show first 10
        value = sample_row[feature]
        print(f"  {feature:35s}: {value:12.6f}")
    
    # Try to understand the scaling
    print(f"\nInvestigating scaling transformation...")
    
    # Look for the raw ppg_signal_energy in training data
    if "ppg_signal_energy" in sample_row.index:
        raw_energy_train = sample_row["ppg_signal_energy"]
        print(f"  Raw energy (training): {raw_energy_train:12.6f}")
        
        # If we can reverse the scaling...
        if raw_energy_train != 0:
            implied_scale_train = train_energy / raw_energy_train if raw_energy_train != 0 else 0
            implied_scale_pred = pred_energy / raw_energy_train if raw_energy_train != 0 else 0
            print(f"  Implied scale (training): {implied_scale_train:12.6f}")
            print(f"  Implied scale (predict):  {implied_scale_pred:12.6f}")
            print(f"  Scale difference: {abs(implied_scale_train - implied_scale_pred):12.6f}")

def test_point_prediction_impact(training_features, predict_features, artifacts):
    """Test if feature differences affect point predictions, not just intervals"""
    print(f"\n" + "="*80)
    print("POINT PREDICTION IMPACT ASSESSMENT")
    print("="*80)
    
    if predict_features is None:
        print("❌ Cannot test - predict.py features not available")
        return
    
    # Load the conformal model
    conformal_path = BASE_DIR / "models" / "quantile_regressor_conformal_calibrated.pkl"
    with open(conformal_path, "rb") as f:
        conformal_bundle = pickle.load(f)
    
    feature_cols = conformal_bundle["feature_list"]
    
    try:
        # Predict using training features
        X_train = pd.DataFrame([training_features[feature_cols]], columns=feature_cols)
        pred_mean_train = conformal_bundle["point_model"].predict(X_train)[0]
        q05_train = conformal_bundle["q05_model"].predict(X_train)[0]
        q95_train = conformal_bundle["q95_model"].predict(X_train)[0]
        
        print(f"Predictions using TRAINING features:")
        print(f"  Point prediction: {pred_mean_train:8.1f} mg/dL")
        print(f"  Raw interval: [{q05_train:6.1f}, {q95_train:6.1f}]")
        
        # Predict using predict.py features
        X_pred = pd.DataFrame([predict_features[feature_cols]], columns=feature_cols)
        pred_mean_pred = conformal_bundle["point_model"].predict(X_pred)[0]
        q05_pred = conformal_bundle["q05_model"].predict(X_pred)[0]
        q95_pred = conformal_bundle["q95_model"].predict(X_pred)[0]
        
        print(f"Predictions using PREDICT.PY features:")
        print(f"  Point prediction: {pred_mean_pred:8.1f} mg/dL")
        print(f"  Raw interval: [{q05_pred:6.1f}, {q95_pred:6.1f}]")
        
        # Calculate differences
        point_diff = abs(pred_mean_train - pred_mean_pred)
        q05_diff = abs(q05_train - q05_pred)
        q95_diff = abs(q95_train - q95_pred)
        
        print(f"\nPREDICTION DIFFERENCES:")
        print(f"  Point prediction difference: {point_diff:8.1f} mg/dL")
        print(f"  Q05 difference: {q05_diff:8.1f} mg/dL")
        print(f"  Q95 difference: {q95_diff:8.1f} mg/dL")
        
        if point_diff > 5.0:
            print(f"❌ SIGNIFICANT POINT PREDICTION DIFFERENCE!")
            print(f"   This confirms the feature preprocessing bug affects core predictions")
        elif point_diff > 1.0:
            print(f"⚠️  Moderate point prediction difference")
        else:
            print(f"✅ Point predictions match reasonably well")
        
        return {
            "point_diff": point_diff,
            "q05_diff": q05_diff,
            "q95_diff": q95_diff,
            "train_prediction": pred_mean_train,
            "predict_prediction": pred_mean_pred
        }
        
    except Exception as e:
        print(f"❌ Error testing point predictions: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main investigation pipeline"""
    print("FEATURE PREPROCESSING MISMATCH INVESTIGATION")
    print("="*80)
    print("GOAL: Find and fix the root cause of feature differences between")
    print("      training/calibration data and predict.py serving path")
    print("")
    print("SUCCESS CRITERIA:")
    print("  1. Identify exactly which features differ and by how much")
    print("  2. Find the root cause (scaler, formula, units, etc.)")
    print("  3. Confirm if point predictions are also affected")
    print("  4. Fix the preprocessing to achieve ~90% coverage")
    
    # Load necessary data
    test_df = pd.read_csv(BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv")
    artifacts = load_training_preprocessing_artifacts()
    
    # Analyze one sample in detail
    sample_idx = 0
    training_features, sample_row = get_training_features_for_sample(artifacts, test_df, sample_idx)
    predict_features, input_dict = get_predict_py_features_for_sample(sample_row)
    
    # Compare feature values
    differences = compare_feature_values(training_features, predict_features)
    
    # Deep dive into ppg_signal_energy mismatch
    investigate_ppg_signal_energy_mismatch(training_features, predict_features, sample_row, input_dict)
    
    # Test impact on predictions
    prediction_impact = test_point_prediction_impact(training_features, predict_features, artifacts)
    
    # Final assessment
    print(f"\n" + "="*80)
    print("MISMATCH ROOT CAUSE ASSESSMENT")
    print("="*80)
    
    if not differences:
        print("✅ NO FEATURE MISMATCH FOUND")
        print("   The 90% vs 76.4% gap must be due to other factors")
    elif len(differences) < 5 and all(d["diff"] < 1.0 for d in differences):
        print("⚠️  MINOR FEATURE DIFFERENCES")
        print(f"   {len(differences)} features differ by <1.0 units")
        print("   Likely due to rounding or minor preprocessing variations")
    else:
        print("❌ SIGNIFICANT FEATURE PREPROCESSING MISMATCH")
        print(f"   {len(differences)} features differ significantly")
        print(f"   Largest difference: {max(d['diff'] for d in differences):.3f} units")
        
        if prediction_impact and prediction_impact["point_diff"] > 5.0:
            print(f"   ❌ POINT PREDICTIONS ALSO AFFECTED: {prediction_impact['point_diff']:.1f} mg/dL difference")
            print("   🚫 BLOCKING ISSUE: Core model predictions are wrong via predict.py")
        else:
            print("   ⚠️  Point predictions less affected, mainly interval width issue")
    
    # Next steps
    print(f"\nNEXT STEPS:")
    if differences and len(differences) > 10:
        print("   1. 🔍 Fix feature preprocessing pipeline alignment")
        print("   2. 🧪 Re-test coverage after fix")
        print("   3. 🎯 Target: Achieve ~90% coverage with fixed preprocessing")
        print("   4. 📋 Only approve for production after reaching target")
    else:
        print("   1. 📊 Small differences acceptable - investigate other coverage factors")
        print("   2. 🤔 Consider if 76.4% coverage is due to model limitations vs preprocessing")
    
    print(f"\n⚠️  PRODUCTION STATUS: NOT APPROVED")
    print(f"   Current: 76.4% coverage vs 90% target")
    print(f"   Gap: {90 - 76.4:.1f} percentage points unacceptable for production")
    
    return differences, prediction_impact

if __name__ == "__main__":
    differences, impact = main()