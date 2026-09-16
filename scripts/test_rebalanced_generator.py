"""
Test the rebalanced synth_generator.py to verify extreme value distribution.
Target: 8% hypoglycemic (<70), 10% severe hyperglycemic (>250), 82% normal (70-250).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.synth_generator import generate_synthetic_dataset

def test_extreme_value_distribution():
    """Test that rebalanced generator produces target extreme distributions"""
    print("="*80)
    print("TESTING REBALANCED SYNTH_GENERATOR")
    print("="*80)
    
    # Generate test dataset with same parameters as production
    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset(
        n_participants=160,  # Same as production
        readings_per_participant_range=(2, 5),
        random_seed=42
    )
    
    print(f"Generated {len(df)} samples")
    
    # Analyze glucose distribution 
    glucose_values = df["bgl_mg_dl"]
    
    # Count extreme values
    hypoglycemic = (glucose_values < 70).sum()
    normal_range = ((glucose_values >= 70) & (glucose_values <= 250)).sum()
    severe_hyperglycemic = (glucose_values > 250).sum()
    
    total = len(glucose_values)
    hypo_pct = (hypoglycemic / total) * 100
    normal_pct = (normal_range / total) * 100
    severe_pct = (severe_hyperglycemic / total) * 100
    
    print(f"\nREBALANCED DISTRIBUTION:")
    print(f"  Hypoglycemic (<70):      {hypoglycemic:3d} samples ({hypo_pct:5.1f}%)")
    print(f"  Normal (70-250):         {normal_range:3d} samples ({normal_pct:5.1f}%)")
    print(f"  Severe hyperglycemic (>250): {severe_hyperglycemic:3d} samples ({severe_pct:5.1f}%)")
    
    # Compare against targets
    target_hypo = 8.0
    target_severe = 10.0
    target_normal = 82.0
    
    print(f"\nTARGET vs ACTUAL:")
    print(f"  Hypoglycemic:     {target_hypo:5.1f}% → {hypo_pct:5.1f}% (Δ{hypo_pct-target_hypo:+5.1f}%)")
    print(f"  Severe hyper:     {target_severe:5.1f}% → {severe_pct:5.1f}% (Δ{severe_pct-target_severe:+5.1f}%)")
    print(f"  Normal range:     {target_normal:5.1f}% → {normal_pct:5.1f}% (Δ{normal_pct-target_normal:+5.1f}%)")
    
    # Check if within acceptable tolerance (±2%)
    hypo_ok = abs(hypo_pct - target_hypo) <= 2.0
    severe_ok = abs(severe_pct - target_severe) <= 2.0
    normal_ok = abs(normal_pct - target_normal) <= 2.0
    
    print(f"\nTOLERANCE CHECK (±2%):")
    print(f"  Hypoglycemic:     {'✅ PASS' if hypo_ok else '❌ FAIL'}")
    print(f"  Severe hyper:     {'✅ PASS' if severe_ok else '❌ FAIL'}")
    print(f"  Normal range:     {'✅ PASS' if normal_ok else '❌ FAIL'}")
    
    # Show extreme value statistics
    if hypoglycemic > 0:
        hypo_values = glucose_values[glucose_values < 70]
        print(f"\nHYPOGLYCEMIC STATISTICS:")
        print(f"  Min: {hypo_values.min():.1f} mg/dL")
        print(f"  Max: {hypo_values.max():.1f} mg/dL")
        print(f"  Mean: {hypo_values.mean():.1f} mg/dL")
        print(f"  Std: {hypo_values.std():.1f} mg/dL")
    
    if severe_hyperglycemic > 0:
        severe_values = glucose_values[glucose_values > 250]
        print(f"\nSEVERE HYPERGLYCEMIC STATISTICS:")
        print(f"  Min: {severe_values.min():.1f} mg/dL")
        print(f"  Max: {severe_values.max():.1f} mg/dL")
        print(f"  Mean: {severe_values.mean():.1f} mg/dL")
        print(f"  Std: {severe_values.std():.1f} mg/dL")
    
    # Diabetes state breakdown
    print(f"\nGENERATED COLUMNS:")
    print(f"  {list(df.columns)}")
    
    if 'diabetes_diagnosis' in df.columns:
        print(f"\nDIABETES DIAGNOSIS BREAKDOWN:")
        diag_counts = df.groupby('diabetes_diagnosis').size()
        for diag, count in diag_counts.items():
            pct = (count / total) * 100
            print(f"  {diag:20s}: {count:3d} samples ({pct:5.1f}%)")
    
    if 'diabetes_status' in df.columns:
        print(f"\nDIABETES STATUS BREAKDOWN:")
        status_counts = df.groupby('diabetes_status').size()
        for status, count in status_counts.items():
            pct = (count / total) * 100
            print(f"  {status:25s}: {count:3d} samples ({pct:5.1f}%)")
    
    all_ok = hypo_ok and severe_ok and normal_ok
    
    print(f"\n" + "="*60)
    print("REBALANCING TEST RESULT")
    print("="*60)
    
    if all_ok:
        print("✅ REBALANCING SUCCESSFUL")
        print("   Target extreme value distribution achieved")
        print("   Generator ready for full pipeline regeneration")
    else:
        print("❌ REBALANCING FAILED")
        print("   Distribution targets not met within tolerance")
        print("   Review stratification logic in type1_extreme generation")
    
    return all_ok, {
        "hypoglycemic_pct": hypo_pct,
        "severe_pct": severe_pct,
        "normal_pct": normal_pct,
        "total_samples": total
    }

def main():
    """Main testing pipeline"""
    print("🧪 TESTING REBALANCED SYNTHETIC DATA GENERATOR")
    print("="*80)
    print("Verifying 8% hypoglycemic + 10% severe hyperglycemic targets")
    
    success, stats = test_extreme_value_distribution()
    
    if success:
        print(f"\n📋 NEXT STEPS:")
        print(f"1. ✅ Generator rebalancing verified")
        print(f"2. 🔄 Run full pipeline regeneration")
        print(f"3. 🤖 Retrain models on rebalanced data")
        print(f"4. 🛡️  Verify Clarke Grid zones")
    else:
        print(f"\n📋 REQUIRED FIXES:")
        print(f"1. ❌ Adjust type1_extreme stratification probabilities")
        print(f"2. ❌ Review glucose distribution parameters")
        print(f"3. ❌ Re-test before pipeline regeneration")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)