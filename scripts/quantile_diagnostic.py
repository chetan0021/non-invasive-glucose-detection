"""
Quantile Regressor Diagnostic & Hyperparameter Tuning

1. Load current quantile models and report hyperparameters
2. Test current models on benchmark cases to see if CI bounds vary meaningfully 
3. Retrain with improved hyperparameters (increased n_estimators, max_depth)
4. Compare performance and CI behavior
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

def load_current_quantile_models():
    """Load and inspect current quantile regressor bundle"""
    print("="*80)
    print("QUANTILE REGRESSOR DIAGNOSTIC - CURRENT MODELS")
    print("="*80)
    
    q_path = MODELS_DIR / "quantile_regressor_full_sensor.pkl"
    
    with open(q_path, "rb") as f:
        q_bundle = pickle.load(f)
    
    print(f"Loaded quantile bundle with keys: {list(q_bundle.keys())}")
    print(f"Current empirical coverage: {q_bundle.get('empirical_test_coverage_pct', 'N/A')}%")
    print(f"Current mean interval width: {q_bundle.get('mean_interval_width_mg_dl', 'N/A')} mg/dL")
    print(f"Validation status: {q_bundle.get('validation_status', 'N/A')}")
    
    # Report hyperparameters for each quantile
    for quantile in [0.05, 0.50, 0.95]:
        if quantile in q_bundle:
            model = q_bundle[quantile]
        else:
            # Try alternative key names
            key_map = {0.05: "q05_model", 0.50: "q50_model", 0.95: "q95_model"}
            model = q_bundle.get(key_map[quantile])
        
        if model is not None:
            print(f"\nQuantile {quantile} (q{int(quantile*100):02d}) Model Hyperparameters:")
            print(f"  - n_estimators: {model.n_estimators}")
            print(f"  - max_depth: {model.max_depth}")
            print(f"  - learning_rate: {model.learning_rate}")
            print(f"  - loss: {model.loss}")
            print(f"  - alpha: {model.alpha}")
        else:
            print(f"\nQuantile {quantile}: Model not found!")
    
    return q_bundle

def create_benchmark_test_cases():
    """Use existing test data samples to evaluate CI behavior"""
    print("\n" + "="*80)
    print("LOADING TEST DATA FOR CI EVALUATION")
    print("="*80)
    
    # Load existing test data
    test_df = pd.read_csv(DATA_DIR / "full_sensor_test_features.csv")
    
    # Load feature manifest
    import json
    with open(BASE_DIR / "reports" / "features_manifest_full_sensor.json", "r") as f:
        manifest = json.load(f)
    
    feature_cols = manifest["scaled_numeric_features"] + manifest["categorical_and_binary_features"]
    
    # Select diverse cases from test data
    cases = {}
    
    # Find a healthy case (None diagnosis, normal BGL)
    healthy_mask = (test_df["diabetes_diagnosis"] == "None") & (test_df["bgl_mg_dl"] < 100)
    if healthy_mask.any():
        idx = healthy_mask.idxmax()
        cases[f"Healthy (BGL={test_df.loc[idx, 'bgl_mg_dl']:.1f})"] = test_df.loc[[idx]][feature_cols]
    
    # Find prediabetes case
    prediab_mask = (test_df["diabetes_diagnosis"] == "Prediabetes")
    if prediab_mask.any():
        idx = prediab_mask.idxmax() 
        cases[f"Prediabetes (BGL={test_df.loc[idx, 'bgl_mg_dl']:.1f})"] = test_df.loc[[idx]][feature_cols]
    
    # Find Type 2 case
    t2_mask = (test_df["diabetes_diagnosis"] == "Type 2")
    if t2_mask.any():
        idx = t2_mask.idxmax()
        cases[f"Type 2 (BGL={test_df.loc[idx, 'bgl_mg_dl']:.1f})"] = test_df.loc[[idx]][feature_cols]
        
    # Find Type 1 case
    t1_mask = (test_df["diabetes_diagnosis"] == "Type 1")
    if t1_mask.any():
        idx = t1_mask.idxmax()
        cases[f"Type 1 (BGL={test_df.loc[idx, 'bgl_mg_dl']:.1f})"] = test_df.loc[[idx]][feature_cols]
    
    # Find high glucose case
    high_mask = test_df["bgl_mg_dl"] > 200
    if high_mask.any():
        idx = test_df.loc[high_mask, "bgl_mg_dl"].idxmax()  # Highest BGL
        cases[f"Severe Hyperglycemia (BGL={test_df.loc[idx, 'bgl_mg_dl']:.1f})"] = test_df.loc[[idx]][feature_cols]
    
    print(f"Selected {len(cases)} diverse test cases from existing test data")
    for case_name in cases.keys():
        print(f"  - {case_name}")
    
    return cases, feature_cols


def evaluate_ci_behavior(q_bundle, test_cases, feature_cols):
    """Test current models on benchmark cases"""
    print("\n" + "="*80)
    print("EVALUATING CURRENT CI BEHAVIOR ON BENCHMARK CASES")
    print("="*80)
    
    results = {}
    
    for case_name, case_df in test_cases.items():
        # Get predictions from all quantiles
        q05_pred = q_bundle["q05_model"].predict(case_df)[0]
        q50_pred = q_bundle["q50_model"].predict(case_df)[0]  
        q95_pred = q_bundle["q95_model"].predict(case_df)[0]
        
        width = q95_pred - q05_pred
        
        results[case_name] = {
            "q05": round(q05_pred, 1),
            "q50": round(q50_pred, 1), 
            "q95": round(q95_pred, 1),
            "width": round(width, 1)
        }
        
        print(f"{case_name}:")
        print(f"  q05: {q05_pred:.1f}   q50: {q50_pred:.1f}   q95: {q95_pred:.1f}   width: {width:.1f}")
    
    # Check if CI varies meaningfully
    widths = [r["width"] for r in results.values()]
    q95_vals = [r["q95"] for r in results.values()]
    
    width_range = max(widths) - min(widths)
    q95_range = max(q95_vals) - min(q95_vals)
    
    print(f"\nCI Behavior Analysis:")
    print(f"  Width range: {width_range:.1f} mg/dL (min: {min(widths):.1f}, max: {max(widths):.1f})")
    print(f"  Q95 range: {q95_range:.1f} mg/dL (min: {min(q95_vals):.1f}, max: {max(q95_vals):.1f})")
    
    # Diagnose if clustering near 190-200
    if q95_range < 30 and min(q95_vals) > 170 and max(q95_vals) < 220:
        print("  ⚠️  PROBLEM DETECTED: Q95 bounds clustering near 190-200 uniformly!")
        return False, results
    else:
        print("  ✓ CI bounds appear to vary appropriately with input")
        return True, results

def retrain_improved_quantile_models(feature_cols):
    """Retrain quantile models with improved hyperparameters"""
    print("\n" + "="*80)
    print("RETRAINING QUANTILE MODELS WITH IMPROVED HYPERPARAMETERS")
    print("="*80)
    
    # Load training data
    train_df = pd.read_csv(DATA_DIR / "full_sensor_train_features.csv")
    test_df = pd.read_csv(DATA_DIR / "full_sensor_test_features.csv")
    
    X_train = train_df[feature_cols]
    y_train = train_df["bgl_mg_dl"]
    X_test = test_df[feature_cols]
    y_test = test_df["bgl_mg_dl"]
    
    print(f"Training data: {len(X_train)} samples")
    print(f"Test data: {len(X_test)} samples")
    
    # Improved hyperparameters (doubled n_estimators and max_depth)
    improved_params = {
        "n_estimators": 200,  # Was 100
        "max_depth": 8,       # Was 4  
        "learning_rate": 0.05, # Keep same
        "random_state": 42
    }
    
    print(f"New hyperparameters: {improved_params}")
    
    # Train improved quantile models
    q_models_improved = {}
    alphas = [0.05, 0.50, 0.95]
    
    for a in alphas:
        print(f"  Training improved quantile model (alpha={a})...")
        gbr = GradientBoostingRegressor(loss="quantile", alpha=a, **improved_params)
        gbr.fit(X_train, y_train)
        q_models_improved[a] = gbr
    
    # Test improved models
    q05_preds = q_models_improved[0.05].predict(X_test)
    q50_preds = q_models_improved[0.50].predict(X_test) 
    q95_preds = q_models_improved[0.95].predict(X_test)
    
    # Compute improved coverage
    y_true = y_test.values
    covered = (y_true >= q05_preds) & (y_true <= q95_preds)
    new_coverage = round(float(np.mean(covered) * 100), 2)
    new_width = round(float(np.mean(q95_preds - q05_preds)), 2)
    
    print(f"\nImproved Model Performance:")
    print(f"  Empirical coverage: {new_coverage}% (target: 90%)")
    print(f"  Mean interval width: {new_width} mg/dL")
    
    # Create improved bundle
    improved_bundle = {
        "q05_model": q_models_improved[0.05],
        "q50_model": q_models_improved[0.50], 
        "q95_model": q_models_improved[0.95],
        "feature_list": feature_cols,
        "empirical_test_coverage_pct": new_coverage,
        "mean_interval_width_mg_dl": new_width,
        "validation_status": "synthetic_self_consistency_only",
        "hyperparameters": improved_params,
        "improvement_note": "Doubled n_estimators (100->200) and max_depth (4->8) to fix CI clustering"
    }
    
    return improved_bundle

def main():
    """Main diagnostic and improvement pipeline"""
    # 1. Load and inspect current models  
    current_bundle = load_current_quantile_models()
    
    # 2. Create benchmark test cases
    test_cases, feature_cols = create_benchmark_test_cases()
    
    # 3. Evaluate current CI behavior
    ci_varies_properly, current_results = evaluate_ci_behavior(current_bundle, test_cases, feature_cols)
    
    if ci_varies_properly:
        print("\n✓ Current quantile regressor appears to be working properly")
        print("No retraining needed - CI bounds vary appropriately with input")
        return
    
    # 4. Retrain with improved hyperparameters
    print("\n⚠️  Current quantile regressor has clustering issue - retraining...")
    improved_bundle = retrain_improved_quantile_models(feature_cols)
    
    # 5. Test improved models on same benchmark cases
    print(f"\n" + "="*80)
    print("TESTING IMPROVED MODELS ON BENCHMARK CASES")
    print("="*80)
    
    improved_results = {}
    for case_name, case_df in test_cases.items():
        q05_pred = improved_bundle["q05_model"].predict(case_df)[0]
        q50_pred = improved_bundle["q50_model"].predict(case_df)[0]
        q95_pred = improved_bundle["q95_model"].predict(case_df)[0]
        width = q95_pred - q05_pred
        
        improved_results[case_name] = {
            "q05": round(q05_pred, 1),
            "q50": round(q50_pred, 1),
            "q95": round(q95_pred, 1), 
            "width": round(width, 1)
        }
        
        # Compare to original
        orig = current_results[case_name]
        print(f"{case_name}:")
        print(f"  Original: q05: {orig['q05']}   q95: {orig['q95']}   width: {orig['width']}")
        print(f"  Improved: q05: {improved_results[case_name]['q05']}   q95: {improved_results[case_name]['q95']}   width: {improved_results[case_name]['width']}")
        print()
    
    # 6. Check if improvement was successful
    improved_widths = [r["width"] for r in improved_results.values()]
    improved_q95_vals = [r["q95"] for r in improved_results.values()]
    
    width_range = max(improved_widths) - min(improved_widths) 
    q95_range = max(improved_q95_vals) - min(improved_q95_vals)
    
    print(f"Improved CI Behavior Analysis:")
    print(f"  Width range: {width_range:.1f} mg/dL")
    print(f"  Q95 range: {q95_range:.1f} mg/dL")
    
    if q95_range > 40 and width_range > 30:
        print("  ✅ SUCCESS: Improved models show proper CI variation!")
        
        # Save improved models
        improved_path = MODELS_DIR / "quantile_regressor_full_sensor_improved.pkl"
        with open(improved_path, "wb") as f:
            pickle.dump(improved_bundle, f)
        print(f"  💾 Saved improved quantile bundle to: {improved_path}")
        
        # Backup original and replace
        import shutil
        original_path = MODELS_DIR / "quantile_regressor_full_sensor.pkl"
        backup_path = MODELS_DIR / "quantile_regressor_full_sensor_backup.pkl"
        shutil.copy(original_path, backup_path)
        shutil.copy(improved_path, original_path)
        print(f"  📁 Backed up original to: {backup_path}")
        print(f"  🔄 Replaced production model with improved version")
        
    else:
        print("  ❌ Improved models still show clustering - deeper fix needed")
        print("  Recommend switching to conformal prediction in future phase")

if __name__ == "__main__":
    main()