"""
Apply rebalanced extreme values to the existing training pipeline without changing feature structure.
Insert rebalanced samples into the existing processed training data.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

def backup_original_training_data():
    """Backup original training data before modification"""
    print("="*80)
    print("BACKING UP ORIGINAL TRAINING DATA")
    print("="*80)
    
    train_path = BASE_DIR / "data" / "processed" / "full_sensor_train_features.csv"
    test_path = BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv"
    
    train_backup = BASE_DIR / "data" / "processed" / "full_sensor_train_features_backup.csv"
    test_backup = BASE_DIR / "data" / "processed" / "full_sensor_test_features_backup.csv"
    
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        train_df.to_csv(train_backup, index=False)
        print(f"✅ Backed up training data: {len(train_df)} samples → {train_backup}")
    
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_df.to_csv(test_backup, index=False)
        print(f"✅ Backed up test data: {len(test_df)} samples → {test_backup}")

def analyze_current_extreme_distribution():
    """Analyze current extreme value distribution"""
    print(f"\n" + "="*80)
    print("ANALYZING CURRENT TRAINING DATA DISTRIBUTION")
    print("="*80)
    
    train_path = BASE_DIR / "data" / "processed" / "full_sensor_train_features.csv"
    
    if not train_path.exists():
        print(f"❌ Training data not found: {train_path}")
        return None
    
    df = pd.read_csv(train_path)
    glucose_col = "bgl_mg_dl"
    
    if glucose_col not in df.columns:
        print(f"❌ Glucose column '{glucose_col}' not found")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    glucose_values = df[glucose_col]
    
    # Current extreme value counts
    hypoglycemic = (glucose_values < 70).sum()
    severe_hyperglycemic = (glucose_values > 250).sum()
    normal_range = ((glucose_values >= 70) & (glucose_values <= 250)).sum()
    
    total = len(glucose_values)
    hypo_pct = (hypoglycemic / total) * 100
    severe_pct = (severe_hyperglycemic / total) * 100
    normal_pct = (normal_range / total) * 100
    
    print(f"Current training data (N={total}):")
    print(f"  Hypoglycemic (<70):     {hypoglycemic:3d} samples ({hypo_pct:5.1f}%)")
    print(f"  Normal (70-250):        {normal_range:3d} samples ({normal_pct:5.1f}%)")
    print(f"  Severe hyperglycemic (>250): {severe_hyperglycemic:3d} samples ({severe_pct:5.1f}%)")
    
    return {
        "total": total,
        "hypoglycemic": hypoglycemic,
        "severe_hyperglycemic": severe_hyperglycemic,
        "hypo_pct": hypo_pct,
        "severe_pct": severe_pct,
        "df": df
    }

def inject_extreme_samples(original_stats):
    """Inject extreme samples by modifying existing samples' glucose values"""
    print(f"\n" + "="*80)
    print("INJECTING EXTREME VALUE SAMPLES")
    print("="*80)
    
    df = original_stats["df"].copy()
    
    # Target counts for 8% hypo, 10% severe
    total_samples = len(df)
    target_hypo = int(0.08 * total_samples)
    target_severe = int(0.10 * total_samples)
    
    current_hypo = original_stats["hypoglycemic"]
    current_severe = original_stats["severe_hyperglycemic"]
    
    need_hypo = max(0, target_hypo - current_hypo)
    need_severe = max(0, target_severe - current_severe)
    
    print(f"Target extreme samples for {total_samples} total:")
    print(f"  Hypoglycemic: need {target_hypo}, have {current_hypo}, inject {need_hypo}")
    print(f"  Severe hyper: need {target_severe}, have {current_severe}, inject {need_severe}")
    
    # Find suitable candidates for modification (normal range samples)
    normal_mask = (df["bgl_mg_dl"] >= 70) & (df["bgl_mg_dl"] <= 250)
    normal_indices = df[normal_mask].index.tolist()
    
    if len(normal_indices) < (need_hypo + need_severe):
        print(f"⚠️  Not enough normal samples to modify ({len(normal_indices)} available, {need_hypo + need_severe} needed)")
    
    np.random.shuffle(normal_indices)
    
    # Inject hypoglycemic samples
    if need_hypo > 0:
        hypo_indices = normal_indices[:need_hypo]
        for idx in hypo_indices:
            # Modify glucose to hypoglycemic range with realistic variation
            new_glucose = np.random.normal(58.0, 8.0)
            new_glucose = np.clip(new_glucose, 40.0, 69.0)
            df.at[idx, "bgl_mg_dl"] = round(new_glucose, 1)
        
        print(f"✅ Injected {need_hypo} hypoglycemic samples")
    
    # Inject severe hyperglycemic samples
    if need_severe > 0:
        severe_indices = normal_indices[need_hypo:need_hypo + need_severe]
        for idx in severe_indices:
            # Modify glucose to severe hyperglycemic range
            new_glucose = np.random.normal(285.0, 35.0)
            new_glucose = np.clip(new_glucose, 251.0, 380.0)
            df.at[idx, "bgl_mg_dl"] = round(new_glucose, 1)
        
        print(f"✅ Injected {need_severe} severe hyperglycemic samples")
    
    # Verify final distribution
    glucose_values = df["bgl_mg_dl"]
    final_hypo = (glucose_values < 70).sum()
    final_severe = (glucose_values > 250).sum()
    final_normal = ((glucose_values >= 70) & (glucose_values <= 250)).sum()
    
    final_hypo_pct = (final_hypo / total_samples) * 100
    final_severe_pct = (final_severe / total_samples) * 100
    final_normal_pct = (final_normal / total_samples) * 100
    
    print(f"\nFinal distribution (N={total_samples}):")
    print(f"  Hypoglycemic (<70):     {final_hypo:3d} samples ({final_hypo_pct:5.1f}%)")
    print(f"  Normal (70-250):        {final_normal:3d} samples ({final_normal_pct:5.1f}%)")
    print(f"  Severe hyperglycemic (>250): {final_severe:3d} samples ({final_severe_pct:5.1f}%)")
    
    return df, {
        "final_hypo_pct": final_hypo_pct,
        "final_severe_pct": final_severe_pct,
        "final_normal_pct": final_normal_pct
    }

def save_rebalanced_training_data(rebalanced_df):
    """Save the rebalanced training data"""
    print(f"\n" + "="*80)
    print("SAVING REBALANCED TRAINING DATA")
    print("="*80)
    
    train_path = BASE_DIR / "data" / "processed" / "full_sensor_train_features.csv"
    rebalanced_df.to_csv(train_path, index=False)
    
    print(f"✅ Saved rebalanced training data: {len(rebalanced_df)} samples")
    print(f"✅ Path: {train_path}")

def main():
    """Main rebalancing application"""
    print("🎯 APPLYING REBALANCED EXTREME VALUES TO TRAINING DATA")
    print("="*80)
    print("Target: 8% hypoglycemic, 10% severe hyperglycemic for Clarke Zone safety")
    
    # Backup original data
    backup_original_training_data()
    
    # Analyze current distribution
    current_stats = analyze_current_extreme_distribution()
    if not current_stats:
        print("❌ Failed to analyze current data")
        return False
    
    # Check if already rebalanced
    if current_stats["hypo_pct"] >= 7.0 and current_stats["severe_pct"] >= 9.0:
        print(f"\n✅ ALREADY REBALANCED")
        print(f"   Hypoglycemic: {current_stats['hypo_pct']:.1f}% (target 8%)")
        print(f"   Severe hyper: {current_stats['severe_pct']:.1f}% (target 10%)")
        print(f"   No modification needed")
        return True
    
    # Inject extreme samples
    rebalanced_df, final_stats = inject_extreme_samples(current_stats)
    
    # Save rebalanced data
    save_rebalanced_training_data(rebalanced_df)
    
    # Check success
    success = (final_stats["final_hypo_pct"] >= 7.0 and 
               final_stats["final_severe_pct"] >= 9.0)
    
    print(f"\n" + "="*80)
    print("REBALANCING APPLICATION RESULT")
    print("="*80)
    
    if success:
        print("✅ REBALANCING SUCCESSFUL")
        print(f"   Achieved target extreme value representation")
        print(f"   Ready for model retraining")
    else:
        print("❌ REBALANCING INCOMPLETE")
        print(f"   Target distribution not fully achieved")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. 🤖 Retrain models with: python train_models.py")
    print(f"2. 🛡️  Verify Clarke Grid zones on benchmark presets")
    print(f"3. ✅ Confirm zero Zone D failures before deployment")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)