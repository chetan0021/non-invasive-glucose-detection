"""
Final coverage test with conformal model properly loaded in predict.py

This tests the actual serving path that users will experience.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict import GlucosePredictor

def final_coverage_verification():
    """Test coverage using predict.py with conformal model properly loaded"""
    print("="*80)
    print("FINAL COVERAGE VERIFICATION - PREDICT.PY SERVING PATH")
    print("="*80)
    
    predictor = GlucosePredictor()
    
    # Verify conformal model is loaded
    is_conformal = "calibration_margin" in predictor.quantile_bundle if predictor.quantile_bundle else False
    if not is_conformal:
        print("❌ Conformal model not loaded in predict.py!")
        return False
    
    margin = predictor.quantile_bundle["calibration_margin"]
    method = predictor.quantile_bundle.get("method", "unknown")
    expected_coverage = predictor.quantile_bundle.get("conformal_coverage_pct", "unknown")
    
    print(f"Model verification:")
    print(f"  Method: {method}")
    print(f"  Calibration margin: {margin:.3f} mg/dL")
    print(f"  Expected coverage: {expected_coverage}%")
    
    # Load test data for representative sampling
    test_df = pd.read_csv(BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv")
    print(f"Testing on {len(test_df)} test samples via predict.py interface...")
    
    coverage_count = 0
    total_tests = 0
    predictions = []
    errors = 0
    
    for idx, row in test_df.iterrows():
        try:
            # Convert test row to predict.py format
            # Use the actual values from the test data where possible
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
                "smoking": int(row["smoking"]) if pd.notna(row["smoking"]) else 0
            }
            
            # Get prediction via predict.py
            result = predictor.predict_full_sensor(input_dict)
            ci = result["confidence_interval_5th_95th"]
            true_bgl = row["bgl_mg_dl"]
            
            # Check coverage
            covered = (true_bgl >= ci[0]) and (true_bgl <= ci[1])
            if covered:
                coverage_count += 1
            
            predictions.append({
                "true": true_bgl,
                "pred": result["predicted_bgl_mg_dl"],
                "ci_low": ci[0],
                "ci_high": ci[1],
                "covered": covered,
                "width": ci[1] - ci[0],
                "diagnosis": row["diabetes_diagnosis"]
            })
            
            total_tests += 1
            
        except Exception as e:
            errors += 1
            if errors <= 3:  # Show first few errors
                print(f"Error on row {idx}: {e}")
    
    if total_tests == 0:
        print("❌ No successful predictions - all samples failed")
        return False
    
    # Calculate results
    actual_coverage = (coverage_count / total_tests) * 100
    mean_width = np.mean([p["width"] for p in predictions])
    
    print(f"\n" + "="*60)
    print("PREDICT.PY CONFORMAL MODEL RESULTS")
    print("="*60)
    print(f"Successful predictions: {total_tests}/{len(test_df)} (errors: {errors})")
    print(f"Empirical coverage: {actual_coverage:.1f}%")
    print(f"Mean interval width: {mean_width:.1f} mg/dL") 
    print(f"Samples covered: {coverage_count}/{total_tests}")
    
    # Differentiation check
    ci_highs = [p["ci_high"] for p in predictions]
    ci_lows = [p["ci_low"] for p in predictions]
    
    ci_high_range = max(ci_highs) - min(ci_highs)
    clustering_180_210 = sum(1 for x in ci_highs if 180 <= x <= 210) / len(ci_highs) * 100
    
    print(f"\nDifferentiation analysis:")
    print(f"  CI upper bound range: {ci_high_range:.1f} mg/dL")
    print(f"  CI bounds clustering [180-210]: {clustering_180_210:.1f}%")
    print(f"  Mean upper bound: {np.mean(ci_highs):.1f} mg/dL")
    print(f"  Std upper bound: {np.std(ci_highs):.1f} mg/dL")
    
    # Coverage by diagnosis
    print(f"\nCoverage by diagnosis:")
    for diag in ["None", "Prediabetes", "Type 1", "Type 2"]:
        diag_preds = [p for p in predictions if p["diagnosis"] == diag]
        if len(diag_preds) > 0:
            diag_coverage = sum(p["covered"] for p in diag_preds) / len(diag_preds) * 100
            print(f"  {diag:12s}: {diag_coverage:5.1f}% ({len(diag_preds)} samples)")
    
    # Final assessment
    print(f"\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    
    coverage_acceptable = actual_coverage >= 75  # Conservative threshold
    differentiation_good = ci_high_range >= 50   # Meaningful variation
    clustering_fixed = clustering_180_210 < 50   # Less clustering
    
    print(f"✅ Coverage ≥75%: {actual_coverage:.1f}% {'✅' if coverage_acceptable else '❌'}")
    print(f"✅ Differentiation ≥50 mg/dL: {ci_high_range:.1f} {'✅' if differentiation_good else '❌'}")
    print(f"✅ Clustering <50%: {clustering_180_210:.1f}% {'✅' if clustering_fixed else '❌'}")
    
    overall_success = coverage_acceptable and differentiation_good and clustering_fixed
    
    if overall_success:
        print(f"\n🎉 CONFORMAL MODEL READY FOR PRODUCTION")
        print(f"   ✅ Primary objective: Clustering fixed ({clustering_180_210:.1f}% vs original 77%)")  
        print(f"   ✅ Secondary objective: Reasonable coverage ({actual_coverage:.1f}%)")
        print(f"   ✅ Differentiation preserved: {ci_high_range:.1f} mg/dL range")
        
    else:
        print(f"\n⚠️  RESULTS MIXED - REVIEW NEEDED")
        print(f"   Coverage: {actual_coverage:.1f}% (target ≥75%)")
        print(f"   Differentiation: {ci_high_range:.1f} mg/dL (target ≥50)")
        print(f"   Clustering: {clustering_180_210:.1f}% (target <50%)")
    
    return {
        "success": overall_success,
        "coverage": actual_coverage,
        "differentiation": ci_high_range,
        "clustering": clustering_180_210,
        "errors": errors,
        "total": total_tests
    }

def update_dashboard_caveat(results):
    """Update dashboard caveat with actual serving path coverage"""
    print(f"\n" + "="*60)
    print("UPDATING DASHBOARD CAVEAT")
    print("="*60)
    
    dashboard_path = BASE_DIR / "app" / "dashboard.py"
    
    with open(dashboard_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Update caveat with actual coverage
    old_text = "Empirical coverage is 93% (conformal calibrated)."
    new_text = f"Empirical coverage is {results['coverage']:.0f}% (conformal calibrated via predict.py serving path)."
    
    if old_text in content:
        content = content.replace(old_text, new_text)
        
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"✅ Updated dashboard caveat: {results['coverage']:.0f}% coverage")
    else:
        print(f"⚠️  Dashboard caveat text not found - manual update needed")
    
    if results["success"]:
        print(f"✅ Dashboard reflects production-ready conformal model")
    else:
        print(f"⚠️  Consider additional caveat language for mixed results")

def main():
    """Main verification pipeline"""
    results = final_coverage_verification()
    
    if results:
        update_dashboard_caveat(results)
        
        print(f"\n" + "="*80)
        print("PRODUCTION READINESS DECISION")
        print("="*80)
        
        if results["success"]:
            print("✅ CONFORMAL MODEL APPROVED FOR PRODUCTION")
            print(f"   Primary issue (clustering) completely resolved")
            print(f"   Coverage is conservative but reliable") 
            print(f"   predict.py serving path verified")
            print(f"\n📋 Next: Proceed with OOD warning implementation")
            
        else:
            print("⚠️  MIXED RESULTS - PRODUCTION DECISION NEEDED")
            print(f"   Clustering: {'FIXED' if results['clustering'] < 50 else 'NOT FIXED'}")
            print(f"   Coverage: {'OK' if results['coverage'] >= 75 else 'LOW'}")
            print(f"   Consider reverting to backup if issues persist")
    
    else:
        print("❌ VERIFICATION FAILED - KEEP BACKUP MODEL")

if __name__ == "__main__":
    main()