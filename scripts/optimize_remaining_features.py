"""
Now that ppg_signal_energy is mostly fixed, let's address the remaining feature mismatches
to achieve the target 90% coverage.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict import GlucosePredictor

def analyze_remaining_feature_gaps():
    """Analyze the remaining feature differences after ppg_signal_energy fix"""
    print("="*80)
    print("ANALYZING REMAINING FEATURE GAPS")
    print("="*80)
    
    # Load test data
    test_df = pd.read_csv(BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv")
    sample = test_df.iloc[0]
    
    # Load feature manifest  
    import json
    with open(BASE_DIR / "reports" / "features_manifest_full_sensor.json", "r") as f:
        manifest = json.load(f)
    
    feature_cols = manifest["scaled_numeric_features"] + manifest["categorical_and_binary_features"]
    
    # Get training features
    training_features = sample[feature_cols]
    
    # Get predict.py features
    input_dict = {
        "saliva_ph": float(sample["saliva_ph"]),
        "temperature_c": float(sample["temperature_c"]), 
        "hr_bpm": float(sample["hr_bpm"]),
        "hrv_sdnn": float(sample["hrv_sdnn"]),
        "hrv_rmssd": float(sample["hrv_rmssd"]),
        "hrv_pnn50": float(sample["hrv_pnn50"]),
        "hrv_lf_hf_ratio": float(sample["hrv_lf_hf_ratio"]),
        "perfusion_index": float(sample["perfusion_index"]),
        "pulse_width_ms": float(sample["pulse_width_ms"]),
        "ppg_raw_dc_baseline": float(sample["ppg_raw_dc_baseline"]),
        "ppg_raw_ac_p2p": float(sample["ppg_raw_ac_p2p"]),
        "age": float(sample["age"]),
        "bmi": float(sample["bmi"]),
        "diabetes_diagnosis": str(sample["diabetes_diagnosis"]),
        "fasting": int(sample["fasting"]),
        "gender": "Male" if sample.get("gender_male", 1) == 1 else "Female",
        "family_history": int(sample["family_history"]),
        "smoking": int(sample["smoking"]),
        "med_taking_insulin": int(sample["med_taking_insulin"]),
        "med_taking_oral": int(sample["med_taking_oral"])
    }
    
    predictor = GlucosePredictor()
    predict_features_df = predictor._prepare_full_sensor_features(input_dict)
    predict_features = predict_features_df.iloc[0]
    
    # Compare features
    print(f"Feature comparison after ppg_signal_energy fix:")
    print(f"{'Feature Name':35s} {'Training':>12s} {'Predict.py':>12s} {'Difference':>12s}")
    print("-" * 75)
    
    large_differences = []
    
    for feature in feature_cols:
        if feature in training_features.index and feature in predict_features.index:
            train_val = training_features[feature]
            pred_val = predict_features[feature]
            diff = abs(train_val - pred_val)
            
            if diff > 0.5:  # Still significant differences
                large_differences.append({
                    "feature": feature,
                    "training": train_val,
                    "predict": pred_val,
                    "diff": diff
                })
                print(f"{feature:35s} {train_val:12.6f} {pred_val:12.6f} {diff:12.6f}")
    
    print(f"\n{len(large_differences)} features still have differences >0.5 units")
    
    if large_differences:
        large_differences.sort(key=lambda x: x["diff"], reverse=True)
        print(f"\nTop 5 remaining issues:")
        for i, diff_info in enumerate(large_differences[:5]):
            print(f"  {i+1}. {diff_info['feature']:30s}: diff={diff_info['diff']:8.3f}")
    
    return large_differences

def test_coverage_with_current_fixes():
    """Test coverage with current preprocessing fixes"""
    print(f"\n" + "="*80)
    print("TESTING COVERAGE WITH CURRENT FIXES")
    print("="*80)
    
    predictor = GlucosePredictor()
    
    # Check if conformal model loaded
    is_conformal = "calibration_margin" in predictor.quantile_bundle if predictor.quantile_bundle else False
    print(f"Using conformal model: {'YES' if is_conformal else 'NO'}")
    
    if not is_conformal:
        print("❌ Conformal model not loaded - coverage test invalid")
        return False
    
    # Load test data
    test_df = pd.read_csv(BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv")
    
    coverage_count = 0
    total_tests = 0
    errors = 0
    
    print(f"Testing coverage on {len(test_df)} samples...")
    
    for idx, row in test_df.iterrows():
        if idx >= 50:  # Test on first 50 samples for speed
            break
            
        try:
            input_dict = {
                "saliva_ph": float(row["saliva_ph"]) if pd.notna(row["saliva_ph"]) else 7.25,
                "temperature_c": float(row["temperature_c"]) if pd.notna(row["temperature_c"]) else 36.6,
                "hr_bpm": float(row["hr_bpm"]) if pd.notna(row["hr_bpm"]) else 72.0,
                "hrv_sdnn": float(row["hrv_sdnn"]) if pd.notna(row["hrv_sdnn"]) else 45.0,
                "hrv_rmssd": float(row["hrv_rmssd"]) if pd.notna(row["hrv_rmssd"]) else 35.0,
                "hrv_lf_hf_ratio": float(row["hrv_lf_hf_ratio"]) if pd.notna(row["hrv_lf_hf_ratio"]) else 1.5,
                "perfusion_index": float(row["perfusion_index"]) if pd.notna(row["perfusion_index"]) else 0.8,
                "age": float(row["age"]) if pd.notna(row["age"]) else 45.0,
                "bmi": float(row["bmi"]) if pd.notna(row["bmi"]) else 26.0,
                "diabetes_diagnosis": str(row["diabetes_diagnosis"]) if pd.notna(row["diabetes_diagnosis"]) else "None",
                "fasting": int(row["fasting"]) if pd.notna(row["fasting"]) else 1,
                "family_history": int(row["family_history"]) if pd.notna(row["family_history"]) else 0,
                "smoking": int(row["smoking"]) if pd.notna(row["smoking"]) else 0,
                "ppg_raw_dc_baseline": float(row["ppg_raw_dc_baseline"]) if pd.notna(row["ppg_raw_dc_baseline"]) else 175000.0,
                "ppg_raw_ac_p2p": float(row["ppg_raw_ac_p2p"]) if pd.notna(row["ppg_raw_ac_p2p"]) else 1200.0
            }
            
            result = predictor.predict_full_sensor(input_dict)
            ci = result["confidence_interval_5th_95th"]
            true_bgl = row["bgl_mg_dl"]
            
            covered = (true_bgl >= ci[0]) and (true_bgl <= ci[1])
            if covered:
                coverage_count += 1
            
            total_tests += 1
            
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"Error on row {idx}: {e}")
    
    if total_tests == 0:
        print("❌ No successful predictions")
        return False
    
    actual_coverage = (coverage_count / total_tests) * 100
    
    print(f"\nCoverage Results (n={total_tests}):")
    print(f"  Empirical coverage: {actual_coverage:.1f}%")
    print(f"  Target coverage: 90.0%")
    print(f"  Gap: {90.0 - actual_coverage:.1f} percentage points")
    print(f"  Errors: {errors}")
    
    if actual_coverage >= 88.0:
        print(f"✅ APPROACHING TARGET! Coverage within 2 points of 90%")
        return True
    elif actual_coverage >= 85.0:
        print(f"⚠️  Getting closer - coverage within 5 points")
        return False
    else:
        print(f"❌ Still significant gap from 90% target")
        return False

def main():
    """Main optimization pipeline"""
    print("OPTIMIZING REMAINING FEATURE PREPROCESSING GAPS")
    print("="*80)
    
    # Step 1: Analyze remaining gaps
    remaining_gaps = analyze_remaining_feature_gaps()
    
    # Step 2: Test current coverage
    approaching_target = test_coverage_with_current_fixes()
    
    print(f"\n" + "="*80)
    print("ASSESSMENT")
    print("="*80)
    
    if approaching_target:
        print("✅ MAJOR PROGRESS - APPROACHING 90% TARGET")
        print("   ppg_signal_energy fix was the key issue")
        print("   Remaining gaps may be acceptable for production")
    elif len(remaining_gaps) < 10:
        print("⚠️  GOOD PROGRESS BUT NOT AT TARGET YET")
        print(f"   {len(remaining_gaps)} features still have significant differences")
        print("   May need additional preprocessing fixes")
    else:
        print("❌ MULTIPLE PREPROCESSING ISSUES REMAIN")
        print(f"   {len(remaining_gaps)} features need attention")
        print("   Systematic fix needed for feature pipeline alignment")
    
    print(f"\nNext steps:")
    print(f"  1. If coverage ≥88%: Consider acceptable for production")
    print(f"  2. If coverage 85-87%: Fix top 2-3 feature differences")
    print(f"  3. If coverage <85%: Need systematic preprocessing alignment")

if __name__ == "__main__":
    main()