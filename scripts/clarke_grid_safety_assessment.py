"""
CRITICAL SAFETY ASSESSMENT: Clarke Error Grid Analysis

Assess whether extreme case predictions land in dangerous Clarke Error Grid zones.
Zone D failures are clinically unacceptable and constitute a safety regression.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict import GlucosePredictor

def clarke_error_grid_zone(reference_bg, predicted_bg):
    """
    Classify prediction according to Clarke Error Grid
    
    Zone A: Clinically accurate (within 20% or <70 mg/dL)
    Zone B: Benign error (would not affect treatment) 
    Zone C: Overcorrection (could lead to unnecessary treatment)
    Zone D: Dangerous failure to detect (could miss critical treatment)
    Zone E: Dangerous error (could lead to opposite treatment)
    
    Returns: 'A', 'B', 'C', 'D', or 'E'
    """
    ref = reference_bg
    pred = predicted_bg
    
    # Zone A: Clinically accurate
    if ref >= 70:
        if abs(pred - ref) <= 0.20 * ref:
            return 'A'
    else:  # ref < 70 (hypoglycemic)
        if pred <= 70:
            return 'A'
    
    # Zone B: Benign errors
    if ref >= 180:
        if pred >= 70:
            if pred <= ref + 110:
                return 'B'
    elif ref >= 70:
        if 70 <= pred <= 180:
            return 'B'
    else:  # ref < 70 (hypoglycemic range)
        if 70 < pred <= 180:
            return 'B'
    
    # Zone C: Overcorrection leading to acceptable outcome
    if ref >= 70 and ref <= 290:
        if pred > ref + 110:
            return 'C'
    elif ref <= 70:
        if pred > 180:
            return 'C'
    
    # Zone D: Dangerous failure to detect
    if ref >= 240:
        if pred < 70:
            return 'D'
    elif ref <= 70:
        if pred >= 180:
            return 'D'
    
    # Zone E: Dangerous opposite treatment
    if ref <= 70 and pred >= 180:
        return 'E'
    elif ref >= 180 and pred <= 70:
        return 'E'
    
    # Catch remaining cases that fall in dangerous ranges
    if ref >= 240 and pred < 110:
        return 'D'  # Severe hyperglycemia predicted as normal/low
    elif ref <= 70 and pred > 110:
        return 'D'  # Hypoglycemia predicted as normal/high
    
    # Default to most likely zone based on ranges
    return 'B'  # Most remaining cases are benign errors

def assess_extreme_case_safety():
    """Critical safety assessment of extreme case predictions"""
    print("="*80)
    print("🚨 CRITICAL SAFETY ASSESSMENT: CLARKE ERROR GRID ANALYSIS")
    print("="*80)
    
    predictor = GlucosePredictor()
    
    # Test the two extreme cases that showed concerning results
    extreme_cases = [
        {
            "name": "Severe Hyperglycemia",
            "reference": 265.0,
            "input": {
                "saliva_ph": 6.15, "temperature_c": 37.2, "hr_bpm": 98.0,
                "hrv_sdnn": 14.0, "hrv_rmssd": 8.0, "hrv_pnn50": 1.0, "hrv_lf_hf_ratio": 3.80,
                "perfusion_index": 1.57, "pulse_width_ms": 325.0,
                "ppg_raw_dc_baseline": 188000.0, "ppg_raw_ac_p2p": 2950.0,
                "vpg_max": 6100.0, "apg_a": 64.0, "apg_b": -38.0,
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
                "vpg_max": 4100.0, "apg_a": 88.0, "apg_b": -75.0,
                "age": 28.0, "bmi": 21.6, "diabetes_diagnosis": "Type 1", "fasting": 1,
                "family_history": 0, "smoking": 0, "gender": "Male"
            }
        }
    ]
    
    print("Testing extreme cases for Clarke Error Grid zone classification...")
    print(f"{'Case':25s} {'Reference':>10s} {'Predicted':>10s} {'Error':>8s} {'Zone':>6s} {'Safety':>15s}")
    print("-" * 80)
    
    safety_failures = []
    
    for case in extreme_cases:
        try:
            result = predictor.predict_full_sensor(case["input"])
            predicted = result["predicted_bgl_mg_dl"]
            reference = case["reference"]
            error = predicted - reference
            
            # Critical: Get Clarke zone classification
            zone = clarke_error_grid_zone(reference, predicted)
            
            # Assess safety
            if zone == 'D':
                safety_status = "🚨 DANGEROUS"
                safety_failures.append(f"{case['name']}: Zone D failure")
            elif zone == 'E':
                safety_status = "🚨 CRITICAL"
                safety_failures.append(f"{case['name']}: Zone E failure")
            elif zone == 'C':
                safety_status = "⚠️ OVERCORRECT"
            elif zone == 'B':
                safety_status = "✅ BENIGN"
            else:  # Zone A
                safety_status = "✅ ACCURATE"
            
            print(f"{case['name']:25s} {reference:10.1f} {predicted:10.1f} {error:8.1f} {zone:>6s} {safety_status:>15s}")
            
        except Exception as e:
            print(f"{case['name']:25s} ERROR: {e}")
            safety_failures.append(f"{case['name']}: Prediction failed")
    
    return safety_failures

def assess_all_benchmarks_clarke_zones():
    """Assess all benchmark presets for Clarke zone safety"""
    print(f"\n" + "="*80)
    print("FULL BENCHMARK CLARKE ZONE ASSESSMENT")
    print("="*80)
    
    predictor = GlucosePredictor()
    
    all_cases = [
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
            "name": "Type 2",
            "reference": 172.0,
            "input": {
                "saliva_ph": 6.60, "temperature_c": 36.9, "hr_bmp": 84.0,
                "hrv_sdnn": 26.0, "hrv_rmssd": 18.0, "hrv_pnn50": 6.0, "hrv_lf_hf_ratio": 2.20,
                "perfusion_index": 1.12, "pulse_width_ms": 298.0,
                "ppg_raw_dc_baseline": 164000.0, "ppg_raw_ac_p2p": 1850.0,
                "age": 59.0, "bmi": 29.8, "diabetes_diagnosis": "Type 2", "fasting": 0,
                "family_history": 1, "smoking": 1, "gender": "Male"
            }
        }
    ]
    
    # Add the extreme cases
    all_cases.extend([
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
                "saliva_ph": 7.42, "temperature_c": 36.1, "hr_bpm": 88.0,
                "hrv_sdnn": 48.0, "hrv_rmssd": 44.0, "hrv_pnn50": 20.0, "hrv_lf_hf_ratio": 1.30,
                "perfusion_index": 0.66, "pulse_width_ms": 260.0,
                "ppg_raw_dc_baseline": 172000.0, "ppg_raw_ac_p2p": 1150.0,
                "age": 28.0, "bmi": 21.6, "diabetes_diagnosis": "Type 1", "fasting": 1,
                "family_history": 0, "smoking": 0, "gender": "Male"
            }
        }
    ])
    
    print(f"{'Preset':25s} {'Reference':>10s} {'Predicted':>10s} {'Zone':>6s} {'Status':>15s}")
    print("-" * 70)
    
    zone_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    dangerous_zones = []
    
    for case in all_cases:
        try:
            result = predictor.predict_full_sensor(case["input"])
            predicted = result["predicted_bgl_mg_dl"]
            reference = case["reference"]
            
            zone = clarke_error_grid_zone(reference, predicted)
            zone_counts[zone] += 1
            
            if zone in ['D', 'E']:
                status = f"🚨 {zone} DANGER"
                dangerous_zones.append(f"{case['name']}: Zone {zone}")
            elif zone == 'C':
                status = f"⚠️ {zone} OVERCORR"
            else:
                status = f"✅ {zone} SAFE"
            
            print(f"{case['name']:25s} {reference:10.1f} {predicted:10.1f} {zone:>6s} {status:>15s}")
            
        except Exception as e:
            print(f"{case['name']:25s} ERROR: {e}")
            dangerous_zones.append(f"{case['name']}: Prediction failed")
    
    print(f"\n" + "="*60)
    print("CLARKE ZONE DISTRIBUTION")
    print("="*60)
    
    total_tests = sum(zone_counts.values())
    for zone in ['A', 'B', 'C', 'D', 'E']:
        count = zone_counts[zone]
        pct = (count / total_tests * 100) if total_tests > 0 else 0
        print(f"Zone {zone}: {count:2d} cases ({pct:5.1f}%)")
    
    return dangerous_zones

def compare_with_previous_model():
    """Check if previous model handled extreme cases better"""
    print(f"\n" + "="*80)
    print("COMPARING WITH PREVIOUS MODEL")
    print("="*80)
    
    # Check if backup model exists
    backup_path = BASE_DIR / "models" / "quantile_regressor_full_sensor_backup.pkl"
    
    if backup_path.exists():
        print("✅ Backup model found - comparison possible")
        print("📋 TODO: Load backup model and test same extreme cases")
        print("   This would require temporarily switching models")
    else:
        print("❌ No backup model found for comparison")
    
    # For now, note that we need to determine if reverting is necessary
    print(f"\n⚠️  CRITICAL DECISION NEEDED:")
    print(f"   If extreme cases show Zone D/E failures, must revert to backup")
    print(f"   Safety regression is unacceptable regardless of CI improvements")

def main():
    """Main safety assessment pipeline"""
    print("🚨 CRITICAL SAFETY REGRESSION ASSESSMENT")
    print("="*80)
    print("Assessing Clarke Error Grid zones for extreme case predictions")
    print("Zone D/E failures constitute unacceptable safety regressions")
    
    # Critical assessment of extreme cases
    safety_failures = assess_extreme_case_safety()
    
    # Full benchmark assessment
    dangerous_zones = assess_all_benchmarks_clarke_zones()
    
    # Compare with previous model capability
    compare_with_previous_model()
    
    print(f"\n" + "="*80)
    print("🚨 SAFETY VERDICT")
    print("="*80)
    
    if safety_failures or dangerous_zones:
        print("❌ CRITICAL SAFETY FAILURES DETECTED")
        print("🚫 IMMEDIATE ACTION REQUIRED")
        
        if safety_failures:
            print(f"\nExtreme case failures:")
            for failure in safety_failures:
                print(f"   • {failure}")
        
        if dangerous_zones:
            print(f"\nAll dangerous zones:")
            for danger in dangerous_zones:
                print(f"   • {danger}")
        
        print(f"\n📋 REQUIRED ACTIONS:")
        print(f"   1. 🔄 REVERT: Git revert the problematic commit")
        print(f"   2. 🛡️ RESTORE: Deploy previous safer model version")
        print(f"   3. 🔧 FIX: Rebalance training data extreme case representation")
        print(f"   4. 🧪 RETEST: Full pipeline regeneration and safety verification")
        print(f"   5. 📝 REPORT: Update safety findings in reports/")
        
        print(f"\n🚨 DO NOT DEPLOY UNTIL ZONE D/E FAILURES RESOLVED")
        
    else:
        print("✅ NO CRITICAL SAFETY FAILURES")
        print("   All predictions in acceptable Clarke zones")
        print("   Model approved for continued deployment")
    
    return len(safety_failures) > 0 or len(dangerous_zones) > 0

if __name__ == "__main__":
    has_safety_failures = main()
    sys.exit(1 if has_safety_failures else 0)