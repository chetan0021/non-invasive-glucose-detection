"""
Final Verification of Conformal Calibrated Quantile Model

Tests both:
1. Proper calibration (~90% coverage)  
2. Meaningful differentiation across input types

Uses the production predict.py interface to ensure end-to-end functionality.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict import GlucosePredictor

def test_final_model():
    """Test the final conformal calibrated model"""
    print("="*80)
    print("FINAL VERIFICATION: CONFORMAL CALIBRATED MODEL")
    print("="*80)
    
    predictor = GlucosePredictor()
    
    # Load test data for coverage verification
    test_df = pd.read_csv(BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv")
    
    print(f"Testing on {len(test_df)} held-out test samples")
    print(f"Model method: {predictor.quantile_bundle.get('method', 'unknown')}")
    print(f"Reported coverage: {predictor.quantile_bundle.get('conformal_coverage_pct', 'unknown')}%")
    
    # Test 1: Empirical Coverage Verification
    print(f"\n" + "="*60)
    print("TEST 1: EMPIRICAL COVERAGE VERIFICATION")
    print("="*60)
    
    coverage_count = 0
    predictions = []
    
    for idx, row in test_df.iterrows():
        # Convert row to prediction input format
        input_dict = {
            "saliva_ph": row.get("saliva_ph", 7.25),
            "temperature_c": row.get("temperature_c", 36.6),
            "hr_bpm": row.get("hr_bpm", 72),
            "hrv_sdnn": row.get("hrv_sdnn", 45),
            "hrv_rmssd": row.get("hrv_rmssd", 35),
            "perfusion_index": row.get("perfusion_index", 0.8),
            "age": row.get("age", 45),
            "bmi": row.get("bmi", 26),
            "diabetes_diagnosis": row.get("diabetes_diagnosis", "None"),
            "fasting": row.get("fasting", 1)
        }
        
        try:
            pred_result = predictor.predict_full_sensor(input_dict)
            ci = pred_result["confidence_interval_5th_95th"]
            true_bgl = row["bgl_mg_dl"]
            
            covered = (true_bgl >= ci[0]) and (true_bgl <= ci[1])
            if covered:
                coverage_count += 1
                
            predictions.append({
                "true": true_bgl,
                "pred": pred_result["predicted_bgl_mg_dl"],
                "ci_low": ci[0],
                "ci_high": ci[1],
                "covered": covered,
                "width": ci[1] - ci[0],
                "diagnosis": row.get("diabetes_diagnosis", "None")
            })
            
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            continue
    
    actual_coverage = (coverage_count / len(predictions)) * 100
    mean_width = np.mean([p["width"] for p in predictions])
    
    print(f"Empirical coverage: {actual_coverage:.1f}% (target: 90%)")
    print(f"Mean interval width: {mean_width:.1f} mg/dL")
    print(f"Samples covered: {coverage_count}/{len(predictions)}")
    
    # Test 2: Differentiation Verification  
    print(f"\n" + "="*60)
    print("TEST 2: DIFFERENTIATION VERIFICATION")
    print("="*60)
    
    # Create diverse test cases
    test_cases = [
        {
            "name": "Healthy Fasting",
            "input": {
                "saliva_ph": 7.35, "temperature_c": 36.6, "hr_bpm": 68,
                "hrv_sdnn": 55, "hrv_rmssd": 45, "perfusion_index": 0.8,
                "age": 30, "bmi": 22, "diabetes_diagnosis": "None", "fasting": 1
            },
            "expected_range": "tight, low"
        },
        {
            "name": "Prediabetes",
            "input": {
                "saliva_ph": 7.0, "temperature_c": 36.7, "hr_bpm": 75,
                "hrv_sdnn": 35, "hrv_rmssd": 25, "perfusion_index": 0.7,
                "age": 50, "bmi": 28, "diabetes_diagnosis": "Prediabetes", "fasting": 1
            },
            "expected_range": "moderate"
        },
        {
            "name": "Type 1 Volatile",
            "input": {
                "saliva_ph": 6.8, "temperature_c": 36.3, "hr_bpm": 85,
                "hrv_sdnn": 22, "hrv_rmssd": 15, "perfusion_index": 0.65,
                "age": 25, "bmi": 21, "diabetes_diagnosis": "Type 1", "fasting": 0
            },
            "expected_range": "wide, volatile"
        },
        {
            "name": "Severe Hyperglycemia",
            "input": {
                "saliva_ph": 6.2, "temperature_c": 37.1, "hr_bpm": 95,
                "hrv_sdnn": 15, "hrv_rmssd": 8, "perfusion_index": 1.4,
                "age": 45, "bmi": 35, "diabetes_diagnosis": "Type 1", "fasting": 0
            },
            "expected_range": "very wide"
        }
    ]
    
    differentiation_results = []
    
    for case in test_cases:
        try:
            result = predictor.predict_full_sensor(case["input"])
            ci = result["confidence_interval_5th_95th"]
            
            case_result = {
                "name": case["name"],
                "predicted_bgl": result["predicted_bgl_mg_dl"],
                "ci_low": ci[0],
                "ci_high": ci[1],
                "width": ci[1] - ci[0],
                "expected": case["expected_range"]
            }
            
            differentiation_results.append(case_result)
            
            print(f"{case['name']:20s}: BGL={case_result['predicted_bgl']:5.1f}, "
                  f"CI=[{case_result['ci_low']:5.1f}, {case_result['ci_high']:5.1f}], "
                  f"Width={case_result['width']:5.1f} ({case['expected_range']})")
            
        except Exception as e:
            print(f"Error testing {case['name']}: {e}")
    
    # Analyze differentiation
    if len(differentiation_results) >= 2:
        ci_highs = [r["ci_high"] for r in differentiation_results]
        widths = [r["width"] for r in differentiation_results]
        
        ci_high_range = max(ci_highs) - min(ci_highs)
        width_range = max(widths) - min(widths)
        
        print(f"\nDifferentiation Analysis:")
        print(f"  CI upper bound range: {ci_high_range:.1f} mg/dL")
        print(f"  Width range: {width_range:.1f} mg/dL")
    
    # Test 3: Clustering Analysis
    print(f"\n" + "="*60)  
    print("TEST 3: CLUSTERING ANALYSIS")
    print("="*60)
    
    ci_highs = [p["ci_high"] for p in predictions]
    clustering_180_210 = sum(1 for x in ci_highs if 180 <= x <= 210) / len(ci_highs) * 100
    
    print(f"CI upper bounds clustering in [180-210]: {clustering_180_210:.1f}%")
    print(f"CI upper bound statistics:")
    print(f"  Mean: {np.mean(ci_highs):.1f} mg/dL")
    print(f"  Std:  {np.std(ci_highs):.1f} mg/dL")
    print(f"  Min:  {np.min(ci_highs):.1f} mg/dL")
    print(f"  Max:  {np.max(ci_highs):.1f} mg/dL")
    print(f"  Range: {np.max(ci_highs) - np.min(ci_highs):.1f} mg/dL")
    
    # Final Assessment
    print(f"\n" + "="*80)
    print("FINAL ASSESSMENT")
    print("="*80)
    
    coverage_good = actual_coverage >= 85
    differentiation_good = (max(ci_highs) - min(ci_highs)) >= 50
    clustering_acceptable = clustering_180_210 < 50
    
    print(f"✅ Coverage ≥85%: {actual_coverage:.1f}% {'✅' if coverage_good else '❌'}")
    print(f"✅ Differentiation ≥50 mg/dL: {max(ci_highs) - min(ci_highs):.1f} {'✅' if differentiation_good else '❌'}")
    print(f"✅ Clustering <50%: {clustering_180_210:.1f}% {'✅' if clustering_acceptable else '❌'}")
    
    overall_success = coverage_good and differentiation_good and clustering_acceptable
    
    if overall_success:
        print(f"\n🎉 OVERALL SUCCESS: Both calibration and differentiation objectives met!")
        print(f"   Model is ready for production deployment")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: Some objectives not fully met")
        print(f"   Review results and consider further improvements")
    
    return {
        "coverage": actual_coverage,
        "differentiation_range": max(ci_highs) - min(ci_highs),
        "clustering": clustering_180_210,
        "success": overall_success
    }

if __name__ == "__main__":
    results = test_final_model()