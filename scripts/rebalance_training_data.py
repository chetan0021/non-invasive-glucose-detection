"""
Rebalance synth_generator.py to include meaningful representation of extreme values.

Target: 8-10% each of hypoglycemic (<70) and severe hyperglycemic (>250) samples
Current: 0.4% hypoglycemic, 5.7% severe hyperglycemic (insufficient)

This addresses the root cause of Zone D failures in extreme benchmark cases.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

def analyze_current_distribution():
    """Analyze current glucose distribution in training data"""
    print("="*80)
    print("ANALYZING CURRENT TRAINING DATA DISTRIBUTION")
    print("="*80)
    
    # Load current training data
    train_df = pd.read_csv(BASE_DIR / "data" / "processed" / "full_sensor_train_features.csv")
    
    glucose_values = train_df["bgl_mg_dl"]
    
    print(f"Current training data (N={len(glucose_values)}):")
    print(f"  Min: {glucose_values.min():.1f} mg/dL")
    print(f"  Max: {glucose_values.max():.1f} mg/dL") 
    print(f"  Mean: {glucose_values.mean():.1f} mg/dL")
    print(f"  Median: {glucose_values.median():.1f} mg/dL")
    
    # Current extreme value counts
    hypoglycemic = (glucose_values < 70).sum()
    severe_hyperglycemic = (glucose_values > 250).sum()
    normal_range = ((glucose_values >= 70) & (glucose_values <= 250)).sum()
    
    total = len(glucose_values)
    hypo_pct = (hypoglycemic / total) * 100
    severe_pct = (severe_hyperglycemic / total) * 100
    normal_pct = (normal_range / total) * 100
    
    print(f"\nCurrent distribution:")
    print(f"  Hypoglycemic (<70):     {hypoglycemic:3d} samples ({hypo_pct:5.1f}%)")
    print(f"  Normal (70-250):        {normal_range:3d} samples ({normal_pct:5.1f}%)")
    print(f"  Severe hyperglycemic (>250): {severe_hyperglycemic:3d} samples ({severe_pct:5.1f}%)")
    
    return {
        "total": total,
        "hypoglycemic": hypoglycemic,
        "severe_hyperglycemic": severe_hyperglycemic,
        "normal_range": normal_range,
        "hypo_pct": hypo_pct,
        "severe_pct": severe_pct
    }

def calculate_rebalancing_targets():
    """Calculate target sample counts for rebalanced data"""
    print(f"\n" + "="*80)
    print("CALCULATING REBALANCING TARGETS")
    print("="*80)
    
    # Target percentages
    target_hypo_pct = 8.0  # 8% hypoglycemic
    target_severe_pct = 10.0  # 10% severe hyperglycemic
    target_normal_pct = 82.0  # 82% normal range
    
    # Assume similar total sample count as current (~600 samples)
    target_total = 600
    
    target_hypo = int(target_total * target_hypo_pct / 100)
    target_severe = int(target_total * target_severe_pct / 100)
    target_normal = target_total - target_hypo - target_severe
    
    print(f"Rebalancing targets (N={target_total}):")
    print(f"  Hypoglycemic (<70):     {target_hypo:3d} samples ({target_hypo_pct:5.1f}%)")
    print(f"  Normal (70-250):        {target_normal:3d} samples ({target_normal_pct:5.1f}%)")
    print(f"  Severe hyperglycemic (>250): {target_severe:3d} samples ({target_severe_pct:5.1f}%)")
    
    print(f"\n📊 Impact on extreme case representation:")
    print(f"  Hypoglycemic: 0.4% → {target_hypo_pct}% ({target_hypo_pct/0.4:.1f}x increase)")
    print(f"  Severe hyperglycemic: 5.7% → {target_severe_pct}% ({target_severe_pct/5.7:.1f}x increase)")
    
    return {
        "total": target_total,
        "hypo": target_hypo,
        "severe": target_severe, 
        "normal": target_normal
    }

def modify_synth_generator():
    """Modify synth_generator.py to target the rebalanced distribution"""
    print(f"\n" + "="*80)
    print("MODIFYING SYNTH_GENERATOR.PY FOR REBALANCING")
    print("="*80)
    
    synth_path = BASE_DIR / "scripts" / "synth_generator.py"
    
    print(f"📋 Required modifications to synth_generator.py:")
    print(f"1. Increase extreme value sampling probability")
    print(f"2. Adjust glucose distribution parameters")
    print(f"3. Add explicit extreme case generation logic")
    print(f"4. Verify physiological coherence for extreme values")
    
    # Read current synth_generator
    with open(synth_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Look for glucose generation logic
    lines = content.split('\n')
    glucose_lines = []
    
    for i, line in enumerate(lines):
        if 'glucose' in line.lower() and ('bgl' in line or 'mg_dl' in line or 'reading_glucose' in line):
            glucose_lines.append((i+1, line.strip()))
    
    if glucose_lines:
        print(f"\nFound glucose generation logic:")
        for line_num, line in glucose_lines[:10]:  # Show first 10
            print(f"  Line {line_num}: {line}")
    
    print(f"\n🔧 MODIFICATION STRATEGY:")
    print(f"Option 1: Stratified sampling approach")
    print(f"  - Generate samples in 3 strata: hypo, normal, severe")
    print(f"  - Ensure proper physiological correlation for each stratum")
    
    print(f"\nOption 2: Weighted distribution approach")
    print(f"  - Modify base glucose distribution to be multi-modal")
    print(f"  - Add peaks at ~60 mg/dL (hypoglycemic) and ~300 mg/dL (severe)")
    
    print(f"\nOption 3: Explicit extreme case insertion")
    print(f"  - Generate normal population, then replace subset with extreme cases")
    print(f"  - Ensure extreme cases have realistic physiological profiles")

def plan_full_pipeline_regeneration():
    """Plan the full pipeline regeneration after rebalancing"""
    print(f"\n" + "="*80)
    print("PLANNING FULL PIPELINE REGENERATION")
    print("="*80)
    
    steps = [
        "1. 🔧 Modify synth_generator.py for extreme value rebalancing",
        "2. 🔄 Regenerate synthetic dataset with new distribution", 
        "3. 🏗️  Re-run feature engineering pipeline",
        "4. 🤖 Retrain full-sensor regression model",
        "5. 📊 Retrain quantile regression models", 
        "6. 🔬 Re-calibrate conformal prediction",
        "7. 🧪 Reapply APG/VPG preprocessing fixes",
        "8. 🛡️  Verify Clarke Grid zones on all benchmarks",
        "9. 📈 Confirm no Zone D/E failures",
        "10. 🚀 Deploy if safety verified"
    ]
    
    print("Full pipeline regeneration steps:")
    for step in steps:
        print(f"  {step}")
    
    print(f"\n⏱️  Estimated timeline:")
    print(f"  Data generation: ~30 minutes")
    print(f"  Feature engineering: ~15 minutes") 
    print(f"  Model training: ~45 minutes")
    print(f"  Safety verification: ~15 minutes")
    print(f"  Total: ~1.5-2 hours")

def main():
    """Main rebalancing planning pipeline"""
    print("🔧 TRAINING DATA REBALANCING PLAN")
    print("="*80)
    print("Addressing root cause of Zone D failures through extreme value representation")
    
    # Analyze current state
    current_dist = analyze_current_distribution()
    
    # Calculate targets  
    targets = calculate_rebalancing_targets()
    
    # Plan modifications
    modify_synth_generator()
    
    # Plan full regeneration
    plan_full_pipeline_regeneration()
    
    print(f"\n" + "="*80)
    print("REBALANCING IMPLEMENTATION PLAN")
    print("="*80)
    
    print("✅ ANALYSIS COMPLETE")
    print(f"   Current extreme representation insufficient for safety")
    print(f"   Hypoglycemic: {current_dist['hypo_pct']:.1f}% (need 8%)")
    print(f"   Severe hyperglycemic: {current_dist['severe_pct']:.1f}% (need 10%)")
    
    print(f"\n📋 NEXT IMMEDIATE ACTIONS:")
    print(f"1. 🔧 Implement synth_generator.py modifications")
    print(f"2. 🧪 Test extreme case generation logic")  
    print(f"3. 🔄 Execute full pipeline regeneration")
    print(f"4. 🛡️  Verify Zone D failures eliminated")
    
    print(f"\n⚠️  CURRENT STATUS:")
    print(f"   🔄 Safety revert deployed (emergency measure)")
    print(f"   🚫 Production blocked until rebalancing complete") 
    print(f"   🎯 Target: Zero Zone D failures on benchmark presets")

if __name__ == "__main__":
    main()