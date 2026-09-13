"""
Retrain quantile regressors with improved hyperparameters to fix CI clustering
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

def retrain_quantile_models():
    """Retrain with improved hyperparameters"""
    print("="*80)
    print("RETRAINING QUANTILE MODELS WITH IMPROVED HYPERPARAMETERS")
    print("="*80)
    
    # Load training data
    train_df = pd.read_csv(DATA_DIR / "full_sensor_train_features.csv")
    test_df = pd.read_csv(DATA_DIR / "full_sensor_test_features.csv")
    
    # Load feature manifest
    import json
    with open(BASE_DIR / "reports" / "features_manifest_full_sensor.json", "r") as f:
        manifest = json.load(f)
    
    feature_cols = manifest["scaled_numeric_features"] + manifest["categorical_and_binary_features"]
    
    X_train = train_df[feature_cols]
    y_train = train_df["bgl_mg_dl"]
    X_test = test_df[feature_cols]
    y_test = test_df["bgl_mg_dl"]
    
    print(f"Training data: {len(X_train)} samples")
    print(f"Test data: {len(X_test)} samples")
    
    # Original hyperparameters
    original_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05
    }
    
    # Improved hyperparameters (doubled n_estimators and max_depth)
    improved_params = {
        "n_estimators": 200,  # Doubled from 100
        "max_depth": 8,       # Doubled from 4
        "learning_rate": 0.05, # Keep same
        "random_state": 42
    }
    
    print(f"\nOriginal params: {original_params}")
    print(f"Improved params: {improved_params}")
    
    # Train improved quantile models
    alphas = [0.05, 0.50, 0.95]
    q_models_improved = {}
    
    for a in alphas:
        print(f"\n  Training improved quantile model (alpha={a})...")
        gbr = GradientBoostingRegressor(loss="quantile", alpha=a, **improved_params)
        gbr.fit(X_train, y_train)
        q_models_improved[a] = gbr
        print(f"    Training complete - {gbr.n_estimators} estimators")
    
    # Test improved models
    q05_preds = q_models_improved[0.05].predict(X_test)
    q50_preds = q_models_improved[0.50].predict(X_test)
    q95_preds = q_models_improved[0.95].predict(X_test)
    
    # Compute metrics
    y_true = y_test.values
    covered = (y_true >= q05_preds) & (y_true <= q95_preds)
    new_coverage = round(float(np.mean(covered) * 100), 2)
    new_width = round(float(np.mean(q95_preds - q05_preds)), 2)
    
    print(f"\n" + "="*60)
    print("IMPROVED MODEL PERFORMANCE")
    print("="*60)
    print(f"Empirical coverage: {new_coverage}% (target: 90%)")
    print(f"Mean interval width: {new_width} mg/dL")
    
    # Analyze clustering
    print(f"\nImproved Q95 Statistics:")
    print(f"  Mean: {np.mean(q95_preds):.1f} mg/dL")
    print(f"  Std:  {np.std(q95_preds):.1f} mg/dL")
    print(f"  Min:  {np.min(q95_preds):.1f} mg/dL") 
    print(f"  Max:  {np.max(q95_preds):.1f} mg/dL")
    print(f"  Range: {np.max(q95_preds) - np.min(q95_preds):.1f} mg/dL")
    
    # Check clustering
    near_190_200 = np.sum((q95_preds >= 180) & (q95_preds <= 210))
    total_samples = len(q95_preds)
    clustering_pct = (near_190_200 / total_samples) * 100
    
    print(f"\nClustering Analysis (Improved):")
    print(f"  Samples with Q95 in [180-210]: {near_190_200}/{total_samples} ({clustering_pct:.1f}%)")
    
    if clustering_pct < 50:
        print("  ✅ SUCCESS: Clustering issue resolved!")
        success = True
    else:
        print("  ⚠️  Still some clustering - may need conformal prediction")
        success = False
    
    # Test on diverse cases from before
    print(f"\n" + "="*60)
    print("TESTING ON DIVERSE BENCHMARK CASES")
    print("="*60)
    
    # Get same test cases as before
    cases = {}
    prediab_mask = (test_df["diabetes_diagnosis"] == "Prediabetes")
    if prediab_mask.any():
        idx = prediab_mask.idxmax()
        cases["Prediabetes"] = (idx, test_df.loc[idx, "bgl_mg_dl"])
    
    t2_mask = (test_df["diabetes_diagnosis"] == "Type 2")
    if t2_mask.any():
        idx = t2_mask.idxmax()
        cases["Type 2"] = (idx, test_df.loc[idx, "bgl_mg_dl"])
        
    t1_mask = (test_df["diabetes_diagnosis"] == "Type 1")
    if t1_mask.any():
        idx = t1_mask.idxmax()
        cases["Type 1"] = (idx, test_df.loc[idx, "bgl_mg_dl"])
    
    high_mask = test_df["bgl_mg_dl"] > 250
    if high_mask.any():
        idx = test_df.loc[high_mask, "bgl_mg_dl"].idxmax()
        cases["Severe Hyperglycemia"] = (idx, test_df.loc[idx, "bgl_mg_dl"])
    
    # Test improved models on these cases
    for case_name, (idx, true_bgl) in cases.items():
        case_x = X_test.iloc[[idx]]
        q05_val = q_models_improved[0.05].predict(case_x)[0]
        q95_val = q_models_improved[0.95].predict(case_x)[0]
        width = q95_val - q05_val
        
        print(f"{case_name} (True BGL={true_bgl:.1f}):")
        print(f"  Q05: {q05_val:.1f}   Q95: {q95_val:.1f}   Width: {width:.1f}")
    
    # Create improved bundle
    improved_bundle = {
        "q05_model": q_models_improved[0.05],
        "q50_model": q_models_improved[0.50],
        "q95_model": q_models_improved[0.95],
        "feature_list": feature_cols,
        "empirical_test_coverage_pct": new_coverage,
        "mean_interval_width_mg_dl": new_width,
        "validation_status": "synthetic_self_consistency_only",
        "hyperparameters": {
            "original": original_params,
            "improved": improved_params
        },
        "improvement_note": "Doubled n_estimators (100->200) and max_depth (4->8) to fix CI clustering issue",
        "clustering_test_passed": success
    }
    
    return improved_bundle, success

def save_improved_models(improved_bundle, success):
    """Save improved models and backup original"""
    print(f"\n" + "="*60)
    print("SAVING IMPROVED MODELS")
    print("="*60)
    
    if success:
        # Save improved version
        improved_path = MODELS_DIR / "quantile_regressor_full_sensor_improved.pkl"
        with open(improved_path, "wb") as f:
            pickle.dump(improved_bundle, f)
        print(f"✅ Saved improved quantile bundle: {improved_path}")
        
        # Backup original
        import shutil
        original_path = MODELS_DIR / "quantile_regressor_full_sensor.pkl"
        backup_path = MODELS_DIR / "quantile_regressor_full_sensor_backup.pkl"
        
        if original_path.exists():
            shutil.copy(original_path, backup_path)
            print(f"📁 Backed up original to: {backup_path}")
        
        # Replace with improved version
        shutil.copy(improved_path, original_path)
        print(f"🔄 Replaced production model with improved version")
        
    else:
        # Save as alternative but don't replace production
        alt_path = MODELS_DIR / "quantile_regressor_alternative.pkl"
        with open(alt_path, "wb") as f:
            pickle.dump(improved_bundle, f)
        print(f"⚠️  Saved alternative model: {alt_path}")
        print("   (Not replacing production due to persistent clustering)")
        
    return success

if __name__ == "__main__":
    improved_bundle, success = retrain_quantile_models()
    save_improved_models(improved_bundle, success)
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if success:
        print("✅ Quantile regressor improvement SUCCESSFUL")
        print("   - CI clustering issue resolved")
        print("   - Production model updated")
        print("   - Ready for dashboard calibration caveat update")
    else:
        print("⚠️  Quantile regressor improvement PARTIAL")
        print("   - Some clustering still present")
        print("   - Recommend conformal prediction for future phase")
        print("   - Dashboard calibration caveat still needed")