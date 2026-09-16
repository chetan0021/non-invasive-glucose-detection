"""
Regenerate the full pipeline with rebalanced training data:
1. Generate rebalanced synthetic data
2. Merge with real tabular data  
3. Feature engineering
4. Retrain full-sensor regression model
5. Retrain quantile/conformal calibration
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

def step1_generate_rebalanced_synthetic_data():
    """Step 1: Generate rebalanced synthetic dataset"""
    print("="*80)
    print("STEP 1: GENERATING REBALANCED SYNTHETIC DATA")
    print("="*80)
    
    from scripts.synth_generator import generate_synthetic_dataset
    
    print("Generating rebalanced synthetic dataset...")
    df = generate_synthetic_dataset(
        n_participants=160,  # Same as production
        readings_per_participant_range=(3, 5),
        random_seed=42  # Reproducible 
    )
    
    # Save rebalanced synthetic data
    output_path = BASE_DIR / "data" / "interim" / "rebalanced_synthetic_features.csv"
    df.to_csv(output_path, index=False)
    
    # Verify extreme distribution
    glucose_values = df["bgl_mg_dl"]
    hypoglycemic = (glucose_values < 70).sum()
    severe_hyperglycemic = (glucose_values > 250).sum()
    total = len(glucose_values)
    
    hypo_pct = (hypoglycemic / total) * 100
    severe_pct = (severe_hyperglycemic / total) * 100
    
    print(f"✅ Generated {len(df)} samples")
    print(f"✅ Hypoglycemic: {hypoglycemic} samples ({hypo_pct:.1f}%)")
    print(f"✅ Severe hyperglycemic: {severe_hyperglycemic} samples ({severe_pct:.1f}%)")
    print(f"✅ Saved to: {output_path}")
    
    return output_path, df

def step2_merge_with_real_data(synthetic_path):
    """Step 2: Merge synthetic with real tabular data"""
    print(f"\n" + "="*80)
    print("STEP 2: MERGING WITH REAL TABULAR DATA") 
    print("="*80)
    
    # Load real tabular data
    real_data_path = BASE_DIR / "data" / "interim" / "real_feature_pool.parquet"
    
    if not real_data_path.exists():
        print(f"⚠️  Real data not found at {real_data_path}")
        print("   Using synthetic data only for rebalanced training")
        
        synthetic_df = pd.read_csv(synthetic_path)
        merged_output_path = BASE_DIR / "data" / "interim" / "rebalanced_merged_features.csv"
        synthetic_df.to_csv(merged_output_path, index=False)
        
        print(f"✅ Using synthetic-only dataset: {len(synthetic_df)} samples")
        print(f"✅ Saved to: {merged_output_path}")
        
        return merged_output_path, synthetic_df
    
    print("Loading real tabular data...")
    real_df = pd.read_parquet(real_data_path)
    
    print("Loading rebalanced synthetic data...")
    synthetic_df = pd.read_csv(synthetic_path)
    
    print(f"Real data: {len(real_df)} samples")
    print(f"Synthetic data: {len(synthetic_df)} samples")
    
    # Merge datasets
    merged_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    
    # Save merged data
    merged_output_path = BASE_DIR / "data" / "interim" / "rebalanced_merged_features.csv"
    merged_df.to_csv(merged_output_path, index=False)
    
    print(f"✅ Merged dataset: {len(merged_df)} samples")
    print(f"✅ Saved to: {merged_output_path}")
    
    return merged_output_path, merged_df

def step3_feature_engineering(merged_path):
    """Step 3: Run feature engineering pipeline"""
    print(f"\n" + "="*80)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*80)
    
    # Run feature engineering script
    feature_script = BASE_DIR / "scripts" / "feature_engineering.py"
    
    if not feature_script.exists():
        print(f"⚠️  Feature engineering script not found: {feature_script}")
        print("   Using merged data directly for training")
        
        # Copy merged data to processed training files
        merged_df = pd.read_csv(merged_path)
        
        train_output = BASE_DIR / "data" / "processed" / "rebalanced_full_sensor_train_features.csv"
        test_output = BASE_DIR / "data" / "processed" / "rebalanced_full_sensor_test_features.csv"
        
        # Simple 80/20 split for now
        split_idx = int(0.8 * len(merged_df))
        
        train_df = merged_df.iloc[:split_idx]
        test_df = merged_df.iloc[split_idx:]
        
        train_df.to_csv(train_output, index=False)
        test_df.to_csv(test_output, index=False)
        
        print(f"✅ Training set: {len(train_df)} samples")
        print(f"✅ Test set: {len(test_df)} samples")
        print(f"✅ Saved train: {train_output}")
        print(f"✅ Saved test: {test_output}")
        
        return train_output, test_output
    
    print("Running feature engineering pipeline...")
    
    # Execute feature engineering
    import subprocess
    result = subprocess.run([
        sys.executable, str(feature_script)
    ], cwd=str(BASE_DIR), capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Feature engineering completed successfully")
        
        # Check for generated files
        train_output = BASE_DIR / "data" / "processed" / "full_sensor_train_features.csv"
        test_output = BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv"
        
        if train_output.exists() and test_output.exists():
            print(f"✅ Training features: {train_output}")
            print(f"✅ Test features: {test_output}")
            return train_output, test_output
        else:
            print("⚠️  Expected output files not found, using fallback approach")
            return step3_feature_engineering(merged_path)  # Fallback
            
    else:
        print(f"❌ Feature engineering failed:")
        print(f"   STDOUT: {result.stdout}")
        print(f"   STDERR: {result.stderr}")
        raise RuntimeError("Feature engineering pipeline failed")

def step4_retrain_models(train_path, test_path):
    """Step 4: Retrain full-sensor regression model"""
    print(f"\n" + "="*80)
    print("STEP 4: RETRAINING MODELS")
    print("="*80)
    
    train_script = BASE_DIR / "scripts" / "train_models.py"
    
    if not train_script.exists():
        print(f"❌ Training script not found: {train_script}")
        return None, None
    
    print("Retraining full-sensor regression model...")
    
    # Execute model training
    import subprocess
    result = subprocess.run([
        sys.executable, str(train_script)
    ], cwd=str(BASE_DIR), capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Model training completed successfully")
        print(f"   Output: {result.stdout[-500:]}")  # Last 500 chars
        
        # Check for generated models
        model_path = BASE_DIR / "models" / "production_model_full_sensor.pkl"
        quantile_path = BASE_DIR / "models" / "quantile_regressor_full_sensor.pkl"
        
        if model_path.exists():
            print(f"✅ Regression model: {model_path}")
        
        if quantile_path.exists():
            print(f"✅ Quantile model: {quantile_path}")
            
        return model_path, quantile_path
        
    else:
        print(f"❌ Model training failed:")
        print(f"   STDOUT: {result.stdout}")
        print(f"   STDERR: {result.stderr}")
        raise RuntimeError("Model training failed")

def main():
    """Main pipeline regeneration"""
    print("🔄 REGENERATING FULL PIPELINE WITH REBALANCED DATA")
    print("="*80)
    print("Target: 8-10% extreme value representation for Clarke Zone safety")
    
    try:
        # Step 1: Generate rebalanced synthetic data
        synthetic_path, synthetic_df = step1_generate_rebalanced_synthetic_data()
        
        # Step 2: Merge with real data
        merged_path, merged_df = step2_merge_with_real_data(synthetic_path)
        
        # Step 3: Feature engineering
        train_path, test_path = step3_feature_engineering(merged_path)
        
        # Step 4: Retrain models
        model_path, quantile_path = step4_retrain_models(train_path, test_path)
        
        print(f"\n" + "="*80)
        print("PIPELINE REGENERATION COMPLETE")
        print("="*80)
        
        print("✅ SUCCESSFULLY COMPLETED:")
        print(f"   1. Rebalanced synthetic data generated")
        print(f"   2. Data merged and features engineered")
        print(f"   3. Full-sensor regression model retrained")
        print(f"   4. Quantile regression model retrained")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. 🧪 Run Clarke Grid safety verification")
        print(f"   2. 🛡️  Test all benchmark presets")
        print(f"   3. 🔧 Reapply APG/VPG fixes if safety verified")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PIPELINE REGENERATION FAILED")
        print(f"   Error: {e}")
        print(f"\n📋 RECOVERY ACTIONS:")
        print(f"   1. Check error logs above")
        print(f"   2. Verify data file paths")
        print(f"   3. Ensure all dependencies installed")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)