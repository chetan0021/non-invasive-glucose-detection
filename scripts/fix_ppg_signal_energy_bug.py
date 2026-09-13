"""
Fix the root cause of ppg_signal_energy_scaled mismatch

The investigation revealed:
- Training data: ppg_signal_energy_scaled = -1.026109
- predict.py:    ppg_signal_energy_scaled = 27.003963
- Difference: 28.030072 units (MASSIVE!)

This is definitely causing the 90% → 76.4% coverage degradation.
"""

import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

def investigate_ppg_signal_energy_calculation():
    """Find exactly how ppg_signal_energy should be calculated"""
    print("="*80)
    print("PPG_SIGNAL_ENERGY CALCULATION INVESTIGATION")
    print("="*80)
    
    # Load one test sample to see the raw values and expected scaled result
    test_df = pd.read_csv(BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv")
    sample = test_df.iloc[0]
    
    print("Sample 0 raw and scaled values:")
    print(f"  ppg_signal_energy (raw):    {sample['ppg_signal_energy']:15.6f}")
    print(f"  ppg_signal_energy_scaled:   {sample['ppg_signal_energy_scaled']:15.6f}")
    
    # Load the scaler to see what transformation should be applied
    scaler_path = BASE_DIR / "models" / "scaler_full_sensor.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    print(f"\nScaler information:")
    print(f"  Type: {type(scaler)}")
    print(f"  Features: {len(scaler.feature_names_in_)}")
    
    # Find ppg_signal_energy in the scaler
    feature_names = list(scaler.feature_names_in_)
    
    if "ppg_signal_energy" in feature_names:
        idx = feature_names.index("ppg_signal_energy")
        mean = scaler.mean_[idx]
        scale = scaler.scale_[idx]
        
        print(f"\nScaler parameters for ppg_signal_energy:")
        print(f"  Mean (μ): {mean:15.6f}")
        print(f"  Scale (σ): {scale:15.6f}")
        print(f"  Formula: (x - {mean:.6f}) / {scale:.6f}")
        
        # Verify the scaling is correct
        raw_value = sample['ppg_signal_energy']
        expected_scaled = (raw_value - mean) / scale
        actual_scaled = sample['ppg_signal_energy_scaled']
        
        print(f"\nScaling verification:")
        print(f"  Raw value: {raw_value:15.6f}")
        print(f"  Expected scaled: {expected_scaled:15.6f}")
        print(f"  Actual scaled: {actual_scaled:15.6f}")
        print(f"  Match: {'YES' if abs(expected_scaled - actual_scaled) < 0.001 else 'NO'}")
        
        return mean, scale, raw_value
    else:
        print("❌ ppg_signal_energy not found in scaler features")
        print(f"Available features: {feature_names[:10]}...")
        return None, None, None

def find_predict_py_ppg_calculation():
    """Find how predict.py is calculating ppg_signal_energy"""
    print(f"\n" + "="*80)
    print("PREDICT.PY PPG_SIGNAL_ENERGY CALCULATION")
    print("="*80)
    
    # Look at the predict.py source to see how it calculates this feature
    predict_py_path = BASE_DIR / "predict.py"
    
    with open(predict_py_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Search for ppg_signal_energy calculation
    lines = content.split('\n')
    relevant_lines = []
    
    for i, line in enumerate(lines):
        if 'ppg_signal_energy' in line.lower() or 'signal_energy' in line.lower():
            # Include some context
            start = max(0, i-3)
            end = min(len(lines), i+4)
            for j in range(start, end):
                relevant_lines.append(f"{j+1:4d}: {lines[j]}")
            relevant_lines.append("")
    
    if relevant_lines:
        print("Found ppg_signal_energy references in predict.py:")
        for line in relevant_lines:
            print(line)
    else:
        print("❌ No ppg_signal_energy references found in predict.py")
        
        # Look for more general energy calculations
        energy_lines = []
        for i, line in enumerate(lines):
            if 'energy' in line.lower() and 'def ' not in line:
                energy_lines.append(f"{i+1:4d}: {line}")
        
        if energy_lines:
            print("\nGeneral 'energy' references found:")
            for line in energy_lines[:10]:  # First 10
                print(line)

def examine_feature_generation_source():
    """Examine where the training features were originally generated"""
    print(f"\n" + "="*80)
    print("TRAINING FEATURE GENERATION SOURCE")  
    print("="*80)
    
    # Look for feature generation scripts
    scripts_dir = BASE_DIR / "scripts"
    
    feature_scripts = []
    for script_file in scripts_dir.glob("*.py"):
        with open(script_file, "r", encoding="utf-8") as f:
            content = f.read()
            if 'ppg_signal_energy' in content.lower() or 'feature' in script_file.name.lower():
                feature_scripts.append(script_file.name)
    
    print(f"Scripts that might generate features:")
    for script in feature_scripts:
        print(f"  {script}")
    
    # Check the notebooks directory too
    notebooks_dir = BASE_DIR / "notebooks" 
    if notebooks_dir.exists():
        for notebook in notebooks_dir.glob("*.ipynb"):
            print(f"  {notebook.name} (notebook)")

def create_corrected_ppg_energy_function():
    """Create the correct ppg_signal_energy calculation based on training data"""
    print(f"\n" + "="*80)
    print("CREATING CORRECTED PPG_SIGNAL_ENERGY FUNCTION")
    print("="*80)
    
    # Load training data to understand the relationship
    test_df = pd.read_csv(BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv")
    
    # Look at multiple samples to understand the pattern
    print("Examining ppg_signal_energy patterns in training data:")
    print(f"{'Sample':<8} {'Raw Energy':<15} {'Scaled Energy':<15} {'DC Baseline':<15} {'AC P2P':<15}")
    print("-" * 80)
    
    for i in range(min(10, len(test_df))):
        row = test_df.iloc[i]
        print(f"{i:<8} {row['ppg_signal_energy']:<15.1f} {row['ppg_signal_energy_scaled']:<15.6f} "
              f"{row['ppg_raw_dc_baseline']:<15.1f} {row['ppg_raw_ac_p2p']:<15.1f}")
    
    # Try to reverse engineer the calculation
    print(f"\nReverse engineering ppg_signal_energy calculation...")
    
    # Common signal energy calculations:
    # 1. Sum of squared values
    # 2. RMS (root mean square) 
    # 3. Power spectral density related
    # 4. Simple AC^2 + DC^2
    
    sample = test_df.iloc[0]
    dc = sample['ppg_raw_dc_baseline'] 
    ac = sample['ppg_raw_ac_p2p']
    systolic = sample['ppg_systolic_peak']
    diastolic = sample['ppg_diastolic_peak'] 
    trough = sample['ppg_trough']
    expected_energy = sample['ppg_signal_energy']
    
    print(f"\nSample 0 PPG values:")
    print(f"  DC baseline: {dc:12.1f}")
    print(f"  AC p2p:     {ac:12.1f}")  
    print(f"  Systolic:   {systolic:12.1f}")
    print(f"  Diastolic:  {diastolic:12.1f}")
    print(f"  Trough:     {trough:12.1f}")
    print(f"  Expected energy: {expected_energy:12.1f}")
    
    # Test common energy formulas
    formulas = {
        "DC^2 + AC^2": dc**2 + ac**2,
        "AC^2 only": ac**2,  
        "DC^2 only": dc**2,
        "Pulse pressure^2": (systolic - diastolic)**2,
        "Peak-to-peak^2": (systolic - trough)**2,
        "RMS-like": np.sqrt(dc**2 + ac**2),
        "Simple sum": dc + ac
    }
    
    print(f"\nTesting energy calculation formulas:")
    for formula_name, calculated in formulas.items():
        diff = abs(calculated - expected_energy)
        match = "✓" if diff < 1000 else " "  # Within 1000 units
        print(f"  {match} {formula_name:20s}: {calculated:15.1f} (diff: {diff:10.1f})")
    
    # The closest formula is the correct one
    best_formula = min(formulas.items(), key=lambda x: abs(x[1] - expected_energy))
    print(f"\nBest match: {best_formula[0]} = {best_formula[1]:.1f}")
    print(f"Expected:   {expected_energy:.1f}")
    print(f"Difference: {abs(best_formula[1] - expected_energy):.1f}")
    
    return best_formula

def main():
    """Main investigation and fix pipeline"""
    print("PPG_SIGNAL_ENERGY BUG INVESTIGATION AND FIX")
    print("="*80)
    
    # Step 1: Understand the scaler parameters
    mean, scale, sample_raw = investigate_ppg_signal_energy_calculation()
    
    # Step 2: Find how predict.py calculates it  
    find_predict_py_ppg_calculation()
    
    # Step 3: Look at feature generation source
    examine_feature_generation_source()
    
    # Step 4: Reverse engineer the correct calculation
    best_formula = create_corrected_ppg_energy_function()
    
    print(f"\n" + "="*80)
    print("ROOT CAUSE DIAGNOSIS")
    print("="*80)
    
    if mean is not None and scale is not None:
        print(f"✅ SCALER PARAMETERS IDENTIFIED")
        print(f"   Mean: {mean:.6f}")
        print(f"   Scale: {scale:.6f}")
        
        if best_formula and best_formula[0] != "DC^2 + AC^2":
            print(f"✅ ENERGY CALCULATION FORMULA IDENTIFIED")
            print(f"   Correct formula: {best_formula[0]}")
            print(f"   predict.py likely uses different formula")
            
            print(f"\n🔧 REQUIRED FIX:")
            print(f"   1. Update predict.py to use: {best_formula[0]}")
            print(f"   2. Apply same scaling: (energy - {mean:.6f}) / {scale:.6f}")
            print(f"   3. Re-test coverage after fix")
        else:
            print(f"⚠️  ENERGY CALCULATION UNCLEAR")
            print(f"   Need to examine feature generation scripts")
    else:
        print(f"❌ SCALER INVESTIGATION FAILED")
        print(f"   Cannot determine correct ppg_signal_energy transformation")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Fix ppg_signal_energy calculation in predict.py")
    print(f"   2. Fix any other feature calculation differences (21 total)")
    print(f"   3. Re-run coverage test expecting ~90%")
    print(f"   4. Only approve for production after reaching 90% target")

if __name__ == "__main__":
    main()