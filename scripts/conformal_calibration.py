"""
Split Conformal Prediction for Quantile Regressor Calibration

Uses the improved (n_estimators=200, max_depth=8) quantile model that shows good differentiation
but applies conformal calibration to restore ~90% empirical coverage while preserving
the meaningful variation between healthy vs severe cases.

Algorithm:
1. Load the differentiated quantile model (from quantile_regressor_full_sensor_improved.pkl)
2. Split training data into proper_train (60%) and calibration (40%) 
3. Retrain quantile models on proper_train only
4. Compute residuals on calibration set
5. Find calibration margin that achieves ~90% coverage
6. Apply this margin to test predictions
7. Verify both differentiation AND coverage are achieved
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

def split_conformal_calibration():
    """Implement split conformal prediction on improved quantile model"""
    print("="*80)
    print("SPLIT CONFORMAL PREDICTION CALIBRATION")
    print("="*80)
    
    # Load original training data
    train_df = pd.read_csv(DATA_DIR / "full_sensor_train_features.csv")
    test_df = pd.read_csv(DATA_DIR / "full_sensor_test_features.csv")
    
    # Load feature manifest
    import json
    with open(BASE_DIR / "reports" / "features_manifest_full_sensor.json", "r") as f:
        manifest = json.load(f)
    
    feature_cols = manifest["scaled_numeric_features"] + manifest["categorical_and_binary_features"]
    
    X_train_full = train_df[feature_cols]
    y_train_full = train_df["bgl_mg_dl"]
    X_test = test_df[feature_cols]
    y_test = test_df["bgl_mg_dl"]
    
    print(f"Original training data: {len(X_train_full)} samples")
    print(f"Test data: {len(X_test)} samples")
    
    # Step 1: Split training data for conformal prediction
    # 60% for proper training, 40% for calibration
    # Clean stratification column first
    stratify_col = train_df["diabetes_diagnosis"].fillna("Unknown")
    
    X_proper_train, X_calibration, y_proper_train, y_calibration = train_test_split(
        X_train_full, y_train_full, test_size=0.4, random_state=42, 
        stratify=stratify_col  # Use cleaned stratification
    )
    
    print(f"\nSplit conformal setup:")
    print(f"  Proper training: {len(X_proper_train)} samples")
    print(f"  Calibration: {len(X_calibration)} samples")
    print(f"  Test (final): {len(X_test)} samples")
    
    # Step 2: Train improved quantile models on proper_train only
    print(f"\nTraining improved quantile models on proper training set...")
    
    improved_params = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.05,
        "random_state": 42
    }
    
    alphas = [0.05, 0.50, 0.95]
    q_models = {}
    
    for alpha in alphas:
        print(f"  Training quantile model (alpha={alpha})...")
        gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, **improved_params)
        gbr.fit(X_proper_train, y_proper_train)
        q_models[alpha] = gbr
    
    # Step 3: Get predictions on calibration set
    print(f"\nComputing predictions on calibration set...")
    cal_q05 = q_models[0.05].predict(X_calibration)
    cal_q50 = q_models[0.50].predict(X_calibration)
    cal_q95 = q_models[0.95].predict(X_calibration)
    
    # Step 4: Compute calibration residuals (non-conformity scores)
    # Use interval width as the non-conformity score
    cal_widths = cal_q95 - cal_q05
    cal_actual = y_calibration.values
    
    # Check if actual values fall within predicted intervals
    cal_covered = (cal_actual >= cal_q05) & (cal_actual <= cal_q95)
    raw_cal_coverage = np.mean(cal_covered) * 100
    
    print(f"Raw calibration coverage: {raw_cal_coverage:.1f}%")
    
    # Compute residuals for conformal calibration
    # Distance from interval boundaries when actual falls outside
    lower_residuals = np.maximum(0, cal_q05 - cal_actual)  # How much below lower bound
    upper_residuals = np.maximum(0, cal_actual - cal_q95)  # How much above upper bound
    total_residuals = lower_residuals + upper_residuals
    
    # Step 5: Find conformal calibration margin
    # For 90% coverage, use 90th percentile of residuals
    target_coverage = 0.90
    calibration_margin = np.quantile(total_residuals, target_coverage)
    
    print(f"Conformal calibration margin: {calibration_margin:.1f} mg/dL")
    
    # Step 6: Apply conformal calibration to test predictions
    print(f"\nApplying conformal calibration to test set...")
    
    test_q05_raw = q_models[0.05].predict(X_test)
    test_q50_raw = q_models[0.50].predict(X_test)
    test_q95_raw = q_models[0.95].predict(X_test)
    
    # Apply symmetric margin expansion
    test_q05_calibrated = test_q05_raw - calibration_margin
    test_q95_calibrated = test_q95_raw + calibration_margin
    
    # Step 7: Evaluate conformal-calibrated results
    y_test_actual = y_test.values
    conformally_covered = (y_test_actual >= test_q05_calibrated) & (y_test_actual <= test_q95_calibrated)
    conformal_coverage = np.mean(conformally_covered) * 100
    conformal_width = np.mean(test_q95_calibrated - test_q05_calibrated)
    
    print(f"\n" + "="*60)
    print("CONFORMAL CALIBRATION RESULTS")
    print("="*60)
    print(f"Conformal coverage: {conformal_coverage:.1f}% (target: 90%)")
    print(f"Conformal mean width: {conformal_width:.1f} mg/dL")
    
    # Step 8: Test differentiation is preserved
    print(f"\nTesting differentiation preservation...")
    
    # Use same diverse test cases as before
    cases = {}
    
    # Find diverse cases
    prediab_mask = (test_df["diabetes_diagnosis"] == "Prediabetes")
    if prediab_mask.any():
        idx = prediab_mask.idxmax()
        cases["Prediabetes"] = idx
    
    high_mask = test_df["bgl_mg_dl"] > 250
    if high_mask.any():
        idx = test_df.loc[high_mask, "bgl_mg_dl"].idxmax()
        cases["Severe Hyperglycemia"] = idx
    
    t1_mask = (test_df["diabetes_diagnosis"] == "Type 1")
    if t1_mask.any():
        idx = t1_mask.idxmax()
        cases["Type 1"] = idx
    
    healthy_mask = (test_df["diabetes_diagnosis"] == "None") & (test_df["bgl_mg_dl"] < 100)
    if healthy_mask.any():
        idx = healthy_mask.idxmax()
        cases["Healthy"] = idx
    
    differentiation_results = {}
    for case_name, idx in cases.items():
        true_bgl = test_df.loc[idx, "bgl_mg_dl"]
        q05_conf = test_q05_calibrated[idx]
        q95_conf = test_q95_calibrated[idx]
        width_conf = q95_conf - q05_conf
        
        differentiation_results[case_name] = {
            "true_bgl": true_bgl,
            "q05": q05_conf,
            "q95": q95_conf,
            "width": width_conf
        }
        
        print(f"{case_name} (True={true_bgl:.1f}): Q05={q05_conf:.1f}, Q95={q95_conf:.1f}, Width={width_conf:.1f}")
    
    # Check if differentiation is meaningful
    q95_values = [r["q95"] for r in differentiation_results.values()]
    width_values = [r["width"] for r in differentiation_results.values()]
    
    q95_range = max(q95_values) - min(q95_values)
    width_range = max(width_values) - min(width_values)
    
    print(f"\nDifferentiation Analysis:")
    print(f"  Q95 range: {q95_range:.1f} mg/dL")
    print(f"  Width range: {width_range:.1f} mg/dL")
    
    # Step 9: Check clustering
    clustering_180_210 = np.sum((test_q95_calibrated >= 180) & (test_q95_calibrated <= 210)) / len(test_q95_calibrated) * 100
    print(f"  Clustering in [180-210]: {clustering_180_210:.1f}%")
    
    # Success criteria
    coverage_good = conformal_coverage >= 85  # Close to 90%
    differentiation_good = q95_range >= 50    # Meaningful variation
    clustering_fixed = clustering_180_210 < 40  # Less clustering
    
    success = coverage_good and differentiation_good and clustering_fixed
    
    print(f"\n" + "="*60)
    print("SUCCESS CRITERIA CHECK")
    print("="*60)
    print(f"✅ Coverage ≥85%: {conformal_coverage:.1f}% {'✅' if coverage_good else '❌'}")
    print(f"✅ Differentiation ≥50 mg/dL range: {q95_range:.1f} {'✅' if differentiation_good else '❌'}")
    print(f"✅ Clustering <40%: {clustering_180_210:.1f}% {'✅' if clustering_fixed else '❌'}")
    print(f"\nOverall Success: {'✅ YES' if success else '❌ NO'}")
    
    # Step 10: Create conformal-calibrated model bundle
    conformal_bundle = {
        "q05_model": q_models[0.05],
        "q50_model": q_models[0.50],
        "q95_model": q_models[0.95],
        "calibration_margin": calibration_margin,
        "feature_list": feature_cols,
        "conformal_coverage_pct": round(conformal_coverage, 2),
        "mean_interval_width_mg_dl": round(conformal_width, 2),
        "validation_status": "synthetic_self_consistency_only",
        "method": "split_conformal_prediction",
        "hyperparameters": improved_params,
        "calibration_data_size": len(X_calibration),
        "differentiation_preserved": differentiation_good,
        "clustering_resolved": clustering_fixed,
        "success_criteria_met": success
    }
    
    return conformal_bundle, success, {
        "coverage": conformal_coverage,
        "width": conformal_width,
        "q95_range": q95_range,
        "clustering": clustering_180_210,
        "cases": differentiation_results
    }

def save_conformal_model(conformal_bundle, success, metrics):
    """Save conformal model and update production if successful"""
    print(f"\n" + "="*80)
    print("SAVING CONFORMAL CALIBRATED MODEL")
    print("="*80)
    
    # Always save conformal version
    conformal_path = MODELS_DIR / "quantile_regressor_conformal_calibrated.pkl"
    with open(conformal_path, "wb") as f:
        pickle.dump(conformal_bundle, f)
    print(f"💾 Saved conformal model: {conformal_path}")
    
    if success:
        # Replace production model with conformal version
        production_path = MODELS_DIR / "quantile_regressor_full_sensor.pkl"
        with open(production_path, "wb") as f:
            pickle.dump(conformal_bundle, f)
        print(f"🔄 Updated production model with conformal calibrated version")
        
        # Update dashboard caveat
        update_dashboard_caveat_conformal(metrics["coverage"])
        
        print(f"\n✅ CONFORMAL CALIBRATION SUCCESSFUL")
        print(f"   - Coverage: {metrics['coverage']:.1f}% (target: 90%)")
        print(f"   - Differentiation: {metrics['q95_range']:.1f} mg/dL range preserved")
        print(f"   - Clustering: {metrics['clustering']:.1f}% (reduced)")
        
    else:
        print(f"\n⚠️  CONFORMAL CALIBRATION INCOMPLETE")
        print(f"   - Saved alternative model but did not replace production")
        print(f"   - Coverage: {metrics['coverage']:.1f}%")
        print(f"   - May need further tuning or different approach")

def update_dashboard_caveat_conformal(actual_coverage):
    """Update dashboard caveat with actual conformal coverage"""
    dashboard_path = BASE_DIR / "app" / "dashboard.py"
    
    with open(dashboard_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find and replace the calibration caveat
    old_caveat = "Empirical coverage is 63% vs target 90%."
    new_caveat = f"Empirical coverage is {actual_coverage:.0f}% (conformal calibrated)."
    
    if old_caveat in content:
        content = content.replace(old_caveat, new_caveat)
        
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"✅ Updated dashboard caveat with actual coverage: {actual_coverage:.0f}%")
    else:
        print(f"⚠️  Dashboard caveat not found - manual update needed")

def main():
    """Main conformal calibration pipeline"""
    conformal_bundle, success, metrics = split_conformal_calibration()
    save_conformal_model(conformal_bundle, success, metrics)
    
    print(f"\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    
    if success:
        print("🎯 BOTH OBJECTIVES ACHIEVED:")
        print(f"   ✅ Proper Calibration: {metrics['coverage']:.1f}% coverage")
        print(f"   ✅ Meaningful Differentiation: {metrics['q95_range']:.1f} mg/dL Q95 range")
        print(f"   ✅ Reduced Clustering: {metrics['clustering']:.1f}% in narrow band")
        print(f"\n📊 SAMPLE DIFFERENTIATION:")
        for case_name, case_data in metrics["cases"].items():
            print(f"   {case_name}: Q95={case_data['q95']:.0f} mg/dL (True={case_data['true_bgl']:.0f})")
        
        print(f"\n🚀 Production model updated - ready for deployment")
    else:
        print("❌ OBJECTIVES NOT FULLY MET:")
        print(f"   Coverage: {metrics['coverage']:.1f}% (need ≥85%)")
        print(f"   Differentiation: {metrics['q95_range']:.1f} mg/dL (need ≥50)")
        print(f"   Clustering: {metrics['clustering']:.1f}% (need <40%)")
        print(f"\n🔄 Production model remains on backup - further work needed")

if __name__ == "__main__":
    main()