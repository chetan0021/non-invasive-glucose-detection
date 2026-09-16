"""
Verify that the reverted model does NOT have the same Zone D failures.
Use the verified Clarke Error Grid function from train_models.py.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict import GlucosePredictor
from scripts.train_models import clarke_error_grid_zone  # Use verified function

def test_reverted_model_safety():
    """Test reverted model against the same benchmark presets"""
    print("="*80)
    print("VERIFYING REVERTED MODEL SAFETY")
    print("="*80)
    print("Using verified Clarke Error Grid function from train_models.py")
    
    predictor = GlucosePredictor()
    
    # Test the same 5 benchmark presets that showed Zone D failures
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
            "name": "Type 2",
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
                "saliva_ph": 6.15, "temperature_c": 37.2, "hr_bmp": 98.0,
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
    
    print(f"Testing {len(benchmark_cases)} benchmark presets with reverted model...")
    print(f"{'Preset':25s} {'Reference':>10s} {'Predicted':>10s} {'Zone':>6s} {'Safety Status':>15s}")
    print("-" * 75)
    
    zone_d_failures = []
    zone_e_failures = []
    
    for case in benchmark_cases:
        try:
            result = predictor.predict_full_sensor(case["input"])
            predicted = result["predicted_bgl_mg_dl"]
            reference = case["reference"]
            
            # Use the verified Clarke grid function
            zone = clarke_error_grid_zone(reference, predicted)
            
            # Assess safety
            if zone == 'D':
                safety_status = "🚨 ZONE D"
                zone_d_failures.append(case["name"])
            elif zone == 'E':
                safety_status = "🚨 ZONE E" 
                zone_e_failures.append(case["name"])
            elif zone == 'C':
                safety_status = "⚠️ ZONE C"
            elif zone == 'B':
                safety_status = "✅ ZONE B"
            else:  # Zone A
                safety_status = "✅ ZONE A"
            
            print(f"{case['name']:25s} {reference:10.1f} {predicted:10.1f} {zone:>6s} {safety_status:>15s}")
            
        except Exception as e:
            print(f"{case['name']:25s} ERROR: {e}")
            zone_d_failures.append(f"{case['name']} (prediction failed)")
    
    return zone_d_failures, zone_e_failures

def main():
    """Main verification pipeline"""
    print("🔄 VERIFYING REVERTED MODEL SAFETY")
    print("="*80)
    print("Testing if previous model version avoids Zone D/E failures")
    
    zone_d_failures, zone_e_failures = test_reverted_model_safety()
    
    print(f"\n" + "="*60)
    print("REVERTED MODEL SAFETY VERDICT")
    print("="*60)
    
    critical_failures = zone_d_failures + zone_e_failures
    
    if critical_failures:
        print("❌ REVERTED MODEL ALSO HAS SAFETY FAILURES")
        print(f"   Zone D failures: {len(zone_d_failures)}")
        print(f"   Zone E failures: {len(zone_e_failures)}")
        
        for failure in critical_failures:
            print(f"   • {failure}")
        
        print(f"\n🔍 IMPLICATION:")
        print(f"   Safety issues predate the recent fixes")
        print(f"   Need to go back further in git history")
        print(f"   Or accept that extreme case prediction is a fundamental limitation")
        
        print(f"\n📋 NEXT ACTIONS:")
        print(f"   1. Check earlier commits for safer model versions")
        print(f"   2. OR proceed with training data rebalancing as primary fix")
        print(f"   3. Document that extreme case accuracy is a known limitation")
        
    else:
        print("✅ REVERTED MODEL IS SAFE")
        print(f"   No Zone D or Zone E failures detected")
        print(f"   Previous model version handled extreme cases acceptably")
        print(f"   Recent preprocessing fixes caused the regression")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. ✅ Safe model restored") 
        print(f"   2. 🔧 Proceed with training data rebalancing")
        print(f"   3. 🧪 Reapply preprocessing fixes after rebalancing")
        print(f"   4. 🔍 Re-verify safety before next deployment")
    
    print(f"\n⏰ Streamlit deployment status:")
    print(f"   The revert should trigger automatic redeployment")
    print(f"   Check Streamlit Cloud dashboard to confirm deployment")
    
    return len(critical_failures) == 0

if __name__ == "__main__":
    reverted_model_is_safe = main()
    sys.exit(0 if reverted_model_is_safe else 1)