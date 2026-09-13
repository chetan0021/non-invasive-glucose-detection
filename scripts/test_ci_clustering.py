"""
More focused test to check for CI clustering near 190-200 mg/dL specifically
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

def test_ci_clustering():
    """Test if CI bounds cluster near 190-200 uniformly across diverse inputs"""
    
    print("="*80)
    print("FOCUSED CI CLUSTERING TEST")
    print("="*80)
    
    # Load quantile models
    with open(MODELS_DIR / "quantile_regressor_full_sensor.pkl", "rb") as f:
        q_bundle = pickle.load(f)
    
    # Load test data
    test_df = pd.read_csv(DATA_DIR / "full_sensor_test_features.csv")
    
    # Load feature manifest
    import json
    with open(BASE_DIR / "reports" / "features_manifest_full_sensor.json", "r") as f:
        manifest = json.load(f)
    
    feature_cols = manifest["scaled_numeric_features"] + manifest["categorical_and_binary_features"]
    X_test = test_df[feature_cols]
    y_test = test_df["bgl_mg_dl"]
    
    print(f"Testing on {len(test_df)} samples")
    
    # Get all quantile predictions
    q05_preds = q_bundle["q05_model"].predict(X_test)
    q95_preds = q_bundle["q95_model"].predict(X_test)
    
    # Analyze clustering behavior
    print(f"\nQ95 Statistics:")
    print(f"  Mean: {np.mean(q95_preds):.1f} mg/dL")
    print(f"  Std:  {np.std(q95_preds):.1f} mg/dL") 
    print(f"  Min:  {np.min(q95_preds):.1f} mg/dL")
    print(f"  Max:  {np.max(q95_preds):.1f} mg/dL")
    print(f"  Range: {np.max(q95_preds) - np.min(q95_preds):.1f} mg/dL")
    
    # Check if clustering near 190-200
    near_190_200 = np.sum((q95_preds >= 180) & (q95_preds <= 210))
    total_samples = len(q95_preds)
    clustering_pct = (near_190_200 / total_samples) * 100
    
    print(f"\nClustering Analysis:")
    print(f"  Samples with Q95 in [180-210] range: {near_190_200}/{total_samples} ({clustering_pct:.1f}%)")
    
    if clustering_pct > 70:
        print("  ⚠️  CLUSTERING DETECTED: >70% of upper bounds fall in narrow 180-210 range!")
        clustering_problem = True
    else:
        print(f"  ✓ No clustering issue: Only {clustering_pct:.1f}% in narrow range")
        clustering_problem = False
    
    # Analyze by different glucose ranges
    print(f"\nQ95 Bounds by True Glucose Range:")
    ranges = [
        ("Low <100", y_test < 100),
        ("Normal 100-140", (y_test >= 100) & (y_test < 140)),
        ("High 140-200", (y_test >= 140) & (y_test < 200)), 
        ("Very High >=200", y_test >= 200)
    ]
    
    for range_name, mask in ranges:
        if mask.any():
            range_q95 = q95_preds[mask]
            print(f"  {range_name:15s}: mean={np.mean(range_q95):.1f}, std={np.std(range_q95):.1f}, range={np.max(range_q95)-np.min(range_q95):.1f}")
    
    # Check interval widths
    widths = q95_preds - q05_preds
    print(f"\nInterval Width Statistics:")
    print(f"  Mean: {np.mean(widths):.1f} mg/dL")
    print(f"  Std:  {np.std(widths):.1f} mg/dL")
    print(f"  Range: {np.max(widths) - np.min(widths):.1f} mg/dL")
    
    # Check coverage
    covered = (y_test >= q05_preds) & (y_test <= q95_preds)
    coverage = np.mean(covered) * 100
    print(f"\nEmpirical Coverage: {coverage:.1f}% (Target: 90%)")
    
    return clustering_problem

def recommend_action(clustering_problem):
    """Recommend next steps based on clustering analysis"""
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if clustering_problem:
        print("❌ CI CLUSTERING ISSUE CONFIRMED")
        print("\nRecommended actions:")
        print("1. Increase n_estimators from 100 to 200")
        print("2. Increase max_depth from 4 to 8") 
        print("3. If that doesn't help, consider conformal prediction")
        return True
    else:
        print("✅ NO CLUSTERING ISSUE DETECTED")
        print("\nCurrent quantile regressor appears well-calibrated.")
        print("Only action needed: Add calibration caveat to dashboard")
        return False

if __name__ == "__main__":
    clustering_detected = test_ci_clustering()
    recommend_action(clustering_detected)