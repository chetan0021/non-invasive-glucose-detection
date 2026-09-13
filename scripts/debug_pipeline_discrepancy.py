"""
Root Cause Analysis: 92.7% vs 72.4% Coverage Discrepancy

Direct comparison between:
1. Raw conformal model evaluation (training-time code path)
2. predict.py serving path (actual inference)

Tests identical inputs through both paths to identify pipeline bug.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict import GlucosePredictor

def load_conformal_model_directly():
    """Load conformal model for direct evaluation"""
    conformal_path = BASE_DIR / "models" / "quantile_regressor_conformal_calibrated.pkl"
    
    with open(conformal_path, "rb") as f:
        conformal_bundle = pickle.load(f)
    
    return conformal_bundle

def direct_conformal_prediction(conformal_bundle, X_df):
    """Evaluate conformal model directly (same as training-time evaluation)"""
    feature_cols = conformal_bundle["feature_list"]
    calibration_margin = conformal_bundle["calibration_margin"]
    
    # Raw quantile predictions
    q05_raw = conformal_bundle["q05_model"].predict(X_df[feature_cols])
    q95_raw = conformal_bundle["q95_model"].predict(X_df[feature_cols])
    
    # Apply conformal margin
    q05_conf = q05_raw - calibration_margin
    q95_conf = q95_raw + calibration_margin
    
    return q05_conf, q95_conf

def predict_py_prediction(predictor, input_dict):
    """Get predictions through predict.py interface"""
    result = predictor.predict_full_sensor(input_dict)
    ci = result["confidence_interval_5th_95th"]
    return ci[0], ci[1]

def convert_test_row_to_input_dict(row):
    """Convert test DataFrame row to predict.py input format"""
    return {
        # Core sensor readings
        "saliva_ph": float(row.get("saliva_ph", 7.25)),
        "temperature_c": float(row.get("temperature_c", 36.6)),
        "hr_bpm": float(row.get("hr_bpm", 72.0)),
        "hrv_sdnn": float(row.get("hrv_sdnn", 45.0)),
        "hrv_rmssd": float(row.get("hrv_rmssd", 35.0)),
        "hrv_pnn50": float(row.get("hrv_pnn50", 20.0)),
        "hrv_lf_hf_ratio": float(row.get("hrv_lf_hf_ratio", 1.5)),
        "perfusion_index": float(row.get("perfusion_index", 0.8)),
        "pulse_width_ms": float(row.get("pulse_width_ms", 280.0)),
        
        # PPG raw signals (try to extract from available data)
        "ppg_raw_dc_baseline": float(row.get("ppg_raw_dc_baseline", 175000.0)),
        "ppg_raw_ac_p2p": float(row.get("ppg_raw_ac_p2p", 1200.0)),
        
        # Demographics
        "age": float(row.get("age", 45.0)),
        "bmi": float(row.get("bmi", 26.0)),
        "diabetes_diagnosis": str(row.get("diabetes_diagnosis", "None")),
        "fasting": int(row.get("fasting", 1)),
        "gender": str(row.get("gender_male", 1)) if pd.notna(row.get("gender_male")) else "Male",
        "family_history": int(row.get("family_history", 0)),
        "smoking": int(row.get("smoking", 0)),
        
        # Medication status
        "med_taking_insulin": int(row.get("med_taking_insulin", 0)),
        "med_taking_oral": int(row.get("med_taking_oral", 0))
    }

def debug_pipeline_discrepancy():
    """Find root cause of prediction pipeline discrepancy"""
    print("="*80)
    print("PIPELINE DISCREPANCY ROOT CAUSE ANALYSIS")
    print("="*80)
    
    # Load models and data
    conformal_bundle = load_conformal_model_directly()
    predictor = GlucosePredictor()
    
    # Load test data
    test_df = pd.read_csv(BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv")
    
    print(f"Conformal model method: {conformal_bundle.get('method', 'unknown')}")
    print(f"Conformal calibration margin: {conformal_bundle.get('calibration_margin', 'unknown')}")
    print(f"Expected coverage: {conformal_bundle.get('conformal_coverage_pct', 'unknown')}%")
    
    # Load feature manifest
    import json
    with open(BASE_DIR / "reports" / "features_manifest_full_sensor.json", "r") as f:
        manifest = json.load(f)
    
    feature_cols = manifest["scaled_numeric_features"] + manifest["categorical_and_binary_features"]
    
    print(f"\nTesting on first 10 test samples...")
    print("="*80)
    
    discrepancies = []
    
    for i in range(min(10, len(test_df))):
        row = test_df.iloc[i]
        true_bgl = row["bgl_mg_dl"]
        
        print(f"\nSample {i+1}: True BGL = {true_bgl:.1f} mg/dL")
        print("-" * 60)
        
        try:
            # Method 1: Direct conformal model evaluation
            X_direct = test_df.iloc[[i]][feature_cols]  # Use exact same feature matrix
            q05_direct, q95_direct = direct_conformal_prediction(conformal_bundle, X_direct)
            q05_direct = q05_direct[0]
            q95_direct = q95_direct[0]
            width_direct = q95_direct - q05_direct
            covered_direct = (true_bgl >= q05_direct) and (true_bgl <= q95_direct)
            
            print(f"DIRECT MODEL: [{q05_direct:6.1f}, {q95_direct:6.1f}] width={width_direct:5.1f} covered={'YES' if covered_direct else 'NO'}")
            
            # Method 2: predict.py interface
            input_dict = convert_test_row_to_input_dict(row)
            q05_predict, q95_predict = predict_py_prediction(predictor, input_dict)
            width_predict = q95_predict - q05_predict
            covered_predict = (true_bgl >= q05_predict) and (true_bgl <= q95_predict)
            
            print(f"PREDICT.PY: [{q05_predict:6.1f}, {q95_predict:6.1f}] width={width_predict:5.1f} covered={'YES' if covered_predict else 'NO'}")
            
            # Calculate discrepancy
            q05_diff = abs(q05_direct - q05_predict)
            q95_diff = abs(q95_direct - q95_predict)
            width_diff = abs(width_direct - width_predict)
            
            print(f"DIFFERENCE: Q05 diff={q05_diff:5.1f}, Q95 diff={q95_diff:5.1f}, Width diff={width_diff:5.1f}")
            
            if q05_diff > 1.0 or q95_diff > 1.0:
                print(f"⚠️  SIGNIFICANT DISCREPANCY DETECTED!")
            
            discrepancies.append({
                "sample": i+1,
                "true_bgl": true_bgl,
                "q05_direct": q05_direct,
                "q95_direct": q95_direct,
                "covered_direct": covered_direct,
                "q05_predict": q05_predict,
                "q95_predict": q95_predict,
                "covered_predict": covered_predict,
                "q05_diff": q05_diff,
                "q95_diff": q95_diff,
                "width_diff": width_diff
            })
            
        except Exception as e:
            print(f"❌ ERROR on sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
    
    # Analysis
    print(f"\n" + "="*80)
    print("DISCREPANCY ANALYSIS")
    print("="*80)
    
    if not discrepancies:
        print("❌ No successful comparisons - all samples failed")
        return
    
    # Coverage comparison
    direct_coverage = sum(d["covered_direct"] for d in discrepancies) / len(discrepancies) * 100
    predict_coverage = sum(d["covered_predict"] for d in discrepancies) / len(discrepancies) * 100
    
    print(f"Coverage on {len(discrepancies)} samples:")
    print(f"  Direct model: {direct_coverage:.1f}%")
    print(f"  predict.py:   {predict_coverage:.1f}%")
    print(f"  Difference:   {direct_coverage - predict_coverage:.1f} percentage points")
    
    # Interval differences
    q05_diffs = [d["q05_diff"] for d in discrepancies]
    q95_diffs = [d["q95_diff"] for d in discrepancies]
    width_diffs = [d["width_diff"] for d in discrepancies]
    
    print(f"\nInterval differences (mg/dL):")
    print(f"  Q05 differences: mean={np.mean(q05_diffs):.1f}, max={np.max(q05_diffs):.1f}")
    print(f"  Q95 differences: mean={np.mean(q95_diffs):.1f}, max={np.max(q95_diffs):.1f}")
    print(f"  Width differences: mean={np.mean(width_diffs):.1f}, max={np.max(width_diffs):.1f}")
    
    # Diagnose root cause
    print(f"\n" + "="*80)
    print("ROOT CAUSE DIAGNOSIS")
    print("="*80)
    
    max_q05_diff = max(q05_diffs)
    max_q95_diff = max(q95_diffs)
    
    if max_q05_diff < 0.1 and max_q95_diff < 0.1:
        print("✅ NO PIPELINE BUG: Intervals match within 0.1 mg/dL")
        print("   The coverage discrepancy must be due to different test samples")
    elif max_q05_diff < 2.0 and max_q95_diff < 2.0:
        print("⚠️  MINOR DISCREPANCY: Small differences in interval bounds")
        print("   Likely causes: rounding, minor feature scaling differences")
    else:
        print("❌ MAJOR PIPELINE BUG: Large differences in interval bounds")
        print("   Likely causes:")
        print("   - Feature scaling applied differently")
        print("   - Different feature ordering")
        print("   - Wrong model loaded in predict.py")
        print("   - Input preprocessing mismatch")
    
    # Check if predict.py is using conformal model
    predictor_model = predictor.quantile_bundle
    is_conformal = "calibration_margin" in predictor_model if predictor_model else False
    
    print(f"\nPredict.py model check:")
    print(f"  Using conformal model: {'YES' if is_conformal else 'NO'}")
    if is_conformal:
        predictor_margin = predictor_model.get("calibration_margin", 0)
        conformal_margin = conformal_bundle.get("calibration_margin", 0)
        print(f"  Calibration margins match: {'YES' if abs(predictor_margin - conformal_margin) < 0.01 else 'NO'}")
        print(f"    Direct: {conformal_margin:.3f}")
        print(f"    predict.py: {predictor_margin:.3f}")
    
    return discrepancies, {
        "direct_coverage": direct_coverage,
        "predict_coverage": predict_coverage,
        "max_q05_diff": max_q05_diff,
        "max_q95_diff": max_q95_diff
    }

def investigate_feature_processing():
    """Investigate differences in feature processing between paths"""
    print(f"\n" + "="*80)
    print("FEATURE PROCESSING INVESTIGATION")
    print("="*80)
    
    # Load one test sample
    test_df = pd.read_csv(BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv")
    sample_row = test_df.iloc[0]
    
    print("Investigating feature processing for sample 1...")
    
    # Load feature manifest
    import json
    with open(BASE_DIR / "reports" / "features_manifest_full_sensor.json", "r") as f:
        manifest = json.load(f)
    
    feature_cols = manifest["scaled_numeric_features"] + manifest["categorical_and_binary_features"]
    
    # Method 1: Direct feature matrix (training-time)
    X_direct = test_df.iloc[[0]][feature_cols]
    print(f"\nDirect feature matrix shape: {X_direct.shape}")
    print(f"Direct feature columns: {len(X_direct.columns)}")
    print("First 5 direct features:")
    for i, col in enumerate(X_direct.columns[:5]):
        print(f"  {col}: {X_direct.iloc[0, i]:.6f}")
    
    # Method 2: predict.py feature processing
    predictor = GlucosePredictor()
    input_dict = convert_test_row_to_input_dict(sample_row)
    
    try:
        X_predict = predictor._prepare_full_sensor_features(input_dict)
        print(f"\nPredict.py feature matrix shape: {X_predict.shape}")
        print(f"Predict.py feature columns: {len(X_predict.columns)}")
        print("First 5 predict.py features:")
        for i, col in enumerate(X_predict.columns[:5]):
            if col in X_direct.columns:
                direct_val = X_direct[col].iloc[0]
                predict_val = X_predict[col].iloc[0]
                diff = abs(direct_val - predict_val)
                print(f"  {col}: predict={predict_val:.6f}, direct={direct_val:.6f}, diff={diff:.6f}")
            else:
                print(f"  {col}: predict={X_predict[col].iloc[0]:.6f}, NOT IN DIRECT")
        
        # Check for major differences
        common_cols = set(X_direct.columns) & set(X_predict.columns)
        print(f"\nCommon columns: {len(common_cols)}/{len(feature_cols)}")
        
        if len(common_cols) > 0:
            diffs = []
            for col in common_cols:
                direct_val = X_direct[col].iloc[0]
                predict_val = X_predict[col].iloc[0]
                diff = abs(direct_val - predict_val)
                if diff > 0.001:  # Significant difference
                    diffs.append((col, direct_val, predict_val, diff))
            
            if diffs:
                print(f"\nSIGNIFICANT FEATURE DIFFERENCES ({len(diffs)} features):")
                for col, direct_val, predict_val, diff in sorted(diffs, key=lambda x: x[3], reverse=True)[:10]:
                    print(f"  {col}: direct={direct_val:.6f}, predict={predict_val:.6f}, diff={diff:.6f}")
            else:
                print(f"\n✅ All common features match within 0.001 tolerance")
        
    except Exception as e:
        print(f"❌ Error processing predict.py features: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main investigation pipeline"""
    discrepancies, summary = debug_pipeline_discrepancy()
    investigate_feature_processing()
    
    print(f"\n" + "="*80)
    print("FINAL ROOT CAUSE ASSESSMENT")
    print("="*80)
    
    if summary["max_q05_diff"] > 2.0 or summary["max_q95_diff"] > 2.0:
        print("❌ CONFIRMED PIPELINE BUG")
        print("   Large interval differences indicate feature processing mismatch")
        print("   🔧 REQUIRED ACTION: Fix feature scaling/ordering differences")
    elif abs(summary["direct_coverage"] - summary["predict_coverage"]) > 15:
        print("⚠️  COVERAGE DISCREPANCY WITHOUT INTERVAL MISMATCH")  
        print("   May indicate different test populations or edge case handling")
        print("   🔍 REQUIRED ACTION: Investigate test sample differences")
    else:
        print("✅ NO MAJOR PIPELINE BUG DETECTED")
        print("   Small differences likely due to minor rounding or implementation variations")
    
    print(f"\nNext steps:")
    print(f"1. Address any identified feature processing issues")
    print(f"2. Re-test full coverage evaluation")  
    print(f"3. Only deploy conformal model after confirming ~90% coverage via predict.py")

if __name__ == "__main__":
    main()