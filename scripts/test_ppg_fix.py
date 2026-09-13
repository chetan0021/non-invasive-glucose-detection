"""
Test the ppg_signal_energy fix to see if it reduces the massive 28-unit difference
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from predict import GlucosePredictor

def test_ppg_energy_fix():
    """Test if the ppg_signal_energy fix reduces the feature mismatch"""
    print("="*80)
    print("TESTING PPG_SIGNAL_ENERGY FIX")
    print("="*80)
    
    # Load test data for comparison
    test_df = pd.read_csv(BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv")
    sample = test_df.iloc[0]
    
    print("Sample 0 training data values:")
    print(f"  Raw AC P2P: {sample['ppg_raw_ac_p2p']:15.6f}")
    print(f"  Expected ppg_signal_energy: {sample['ppg_signal_energy']:15.6f}")
    print(f"  Expected ppg_signal_energy_scaled: {sample['ppg_signal_energy_scaled']:15.6f}")
    
    # Calculate what the corrected formula should produce
    raw_ac = sample['ppg_raw_ac_p2p']
    corrected_energy = 0.5 * (0.5 * raw_ac) ** 2
    print(f"\nCorrected calculation:")
    print(f"  AC P2P: {raw_ac:15.6f}")
    print(f"  AC half: {0.5 * raw_ac:15.6f}") 
    print(f"  0.5 * (AC half)^2: {corrected_energy:15.6f}")
    print(f"  Training energy: {sample['ppg_signal_energy']:15.6f}")
    print(f"  Difference: {abs(corrected_energy - sample['ppg_signal_energy']):15.6f}")
    
    # Test predict.py with the fix
    predictor = GlucosePredictor()
    
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
    
    try:
        # Get predict.py features
        predict_features_df = predictor._prepare_full_sensor_features(input_dict)
        predict_energy_scaled = predict_features_df["ppg_signal_energy_scaled"].iloc[0]
        training_energy_scaled = sample["ppg_signal_energy_scaled"]
        
        print(f"\nFeature comparison after fix:")
        print(f"  Training ppg_signal_energy_scaled: {training_energy_scaled:15.6f}")
        print(f"  Predict.py ppg_signal_energy_scaled: {predict_energy_scaled:15.6f}")
        print(f"  Difference: {abs(training_energy_scaled - predict_energy_scaled):15.6f}")
        
        improvement = 28.030072 - abs(training_energy_scaled - predict_energy_scaled)
        print(f"  Improvement: {improvement:15.6f} units (was 28.030072)")
        
        if abs(training_energy_scaled - predict_energy_scaled) < 1.0:
            print("✅ MAJOR IMPROVEMENT! Feature mismatch <1.0 unit")
        elif improvement > 20.0:
            print("✅ SIGNIFICANT IMPROVEMENT! Reduced by >20 units")
        else:
            print("⚠️  Some improvement but still significant mismatch")
            
        return abs(training_energy_scaled - predict_energy_scaled) < 1.0
        
    except Exception as e:
        print(f"❌ Error testing fix: {e}")
        return False

def main():
    success = test_ppg_energy_fix()
    
    if success:
        print(f"\n🎉 PPG_SIGNAL_ENERGY FIX SUCCESSFUL!")
        print(f"   Ready to test full coverage with fixed preprocessing")
    else:
        print(f"\n⚠️  Fix incomplete - additional debugging needed")

if __name__ == "__main__":
    main()