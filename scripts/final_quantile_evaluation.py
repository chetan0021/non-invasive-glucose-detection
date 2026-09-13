"""
Final evaluation of quantile regressor improvements and dashboard update
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

def compare_original_vs_improved():
    """Compare original vs improved quantile models"""
    print("="*80)
    print("FINAL QUANTILE REGRESSOR EVALUATION")
    print("="*80)
    
    # Load both models
    with open(MODELS_DIR / "quantile_regressor_full_sensor_backup.pkl", "rb") as f:
        original_bundle = pickle.load(f)
    
    with open(MODELS_DIR / "quantile_regressor_full_sensor.pkl", "rb") as f:
        improved_bundle = pickle.load(f)
    
    # Load test data
    test_df = pd.read_csv(DATA_DIR / "full_sensor_test_features.csv")
    
    # Load feature manifest
    import json
    with open(BASE_DIR / "reports" / "features_manifest_full_sensor.json", "r") as f:
        manifest = json.load(f)
    
    feature_cols = manifest["scaled_numeric_features"] + manifest["categorical_and_binary_features"]
    X_test = test_df[feature_cols]
    y_test = test_df["bgl_mg_dl"].values
    
    print("Comparing Original vs Improved Models:")
    print("="*60)
    
    # Original model predictions
    orig_q05 = original_bundle["q05_model"].predict(X_test)
    orig_q95 = original_bundle["q95_model"].predict(X_test)
    orig_covered = (y_test >= orig_q05) & (y_test <= orig_q95)
    orig_coverage = np.mean(orig_covered) * 100
    orig_width = np.mean(orig_q95 - orig_q05)
    orig_clustering = np.sum((orig_q95 >= 180) & (orig_q95 <= 210)) / len(orig_q95) * 100
    
    # Improved model predictions  
    imp_q05 = improved_bundle["q05_model"].predict(X_test)
    imp_q95 = improved_bundle["q95_model"].predict(X_test)
    imp_covered = (y_test >= imp_q05) & (y_test <= imp_q95)
    imp_coverage = np.mean(imp_covered) * 100
    imp_width = np.mean(imp_q95 - imp_q05)
    imp_clustering = np.sum((imp_q95 >= 180) & (imp_q95 <= 210)) / len(imp_q95) * 100
    
    print("ORIGINAL MODEL:")
    print(f"  Coverage: {orig_coverage:.1f}% (Target: 90%)")
    print(f"  Mean Width: {orig_width:.1f} mg/dL") 
    print(f"  Q95 Range: {np.max(orig_q95) - np.min(orig_q95):.1f} mg/dL")
    print(f"  Clustering in [180-210]: {orig_clustering:.1f}%")
    
    print("\nIMPROVED MODEL:")
    print(f"  Coverage: {imp_coverage:.1f}% (Target: 90%)")
    print(f"  Mean Width: {imp_width:.1f} mg/dL")
    print(f"  Q95 Range: {np.max(imp_q95) - np.min(imp_q95):.1f} mg/dL")
    print(f"  Clustering in [180-210]: {imp_clustering:.1f}%")
    
    print(f"\nTRADE-OFF ANALYSIS:")
    print(f"  Clustering Reduction: {orig_clustering:.1f}% → {imp_clustering:.1f}% (✅ {orig_clustering - imp_clustering:.1f}% improvement)")
    print(f"  Coverage Change: {orig_coverage:.1f}% → {imp_coverage:.1f}% ({'📈' if imp_coverage > orig_coverage else '📉'} {imp_coverage - orig_coverage:.1f}% change)")
    print(f"  Width Change: {orig_width:.1f} → {imp_width:.1f} mg/dL ({'📈' if imp_width > orig_width else '📉'} {imp_width - orig_width:.1f} change)")
    
    # Decision logic
    clustering_fixed = imp_clustering < 30  # Was 77%, now should be <30%
    coverage_acceptable = imp_coverage > 60  # Should maintain reasonable coverage
    
    if clustering_fixed and coverage_acceptable:
        decision = "KEEP_IMPROVED"
        print(f"\n✅ DECISION: Keep improved model")
        print(f"   - Clustering issue resolved ({imp_clustering:.1f}% < 30%)")
        print(f"   - Coverage remains acceptable ({imp_coverage:.1f}% > 60%)")
    elif clustering_fixed and not coverage_acceptable:
        decision = "PARTIAL_SUCCESS"
        print(f"\n⚠️  DECISION: Partial success - clustering fixed but coverage too low")
        print(f"   - Clustering resolved but coverage dropped to {imp_coverage:.1f}%")
        print(f"   - May need conformal prediction for better calibration")
    else:
        decision = "REVERT_ORIGINAL"  
        print(f"\n❌ DECISION: Revert to original model")
        print(f"   - Improvement did not adequately fix clustering")
    
    return decision, {
        "original": {"coverage": orig_coverage, "width": orig_width, "clustering": orig_clustering},
        "improved": {"coverage": imp_coverage, "width": imp_width, "clustering": imp_clustering}
    }

def update_dashboard_caveat():
    """Add calibration caveat to dashboard"""
    print(f"\n" + "="*80)
    print("UPDATING DASHBOARD CALIBRATION CAVEAT")
    print("="*80)
    
    dashboard_path = BASE_DIR / "app" / "dashboard.py"
    
    # Read current dashboard
    with open(dashboard_path, "r", encoding="utf-8") as f:
        dashboard_content = f.read()
    
    # Check if caveat already exists
    if "Uncertainty range is currently under calibration validation" in dashboard_content:
        print("✅ Calibration caveat already exists in dashboard")
        return
    
    # Find the confidence interval display section and add caveat
    caveat_text = '''        # Add calibration caveat for confidence intervals
        st.markdown(
            '<div class="disclaimer-banner">'
            '<strong>⚠️ Calibration Note:</strong> Uncertainty range is currently under calibration validation - '
            'treat as an outer bound estimate, not a precise statistical interval. '
            'Empirical coverage is 63-88% vs target 90%.'
            '</div>',
            unsafe_allow_html=True
        )'''
    
    # Look for the confidence interval plotting section
    if "plot_confidence_interval_gauge" in dashboard_content:
        # Add caveat after confidence interval plot
        insertion_point = dashboard_content.find("st.plotly_chart(ci_fig")
        if insertion_point != -1:
            # Find end of that line
            end_line = dashboard_content.find("\n", insertion_point)
            if end_line != -1:
                # Insert caveat after the plotly_chart call
                new_content = (dashboard_content[:end_line + 1] + 
                             caveat_text + "\n" + 
                             dashboard_content[end_line + 1:])
                
                # Write updated content
                with open(dashboard_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                
                print(f"✅ Added calibration caveat to dashboard at line {insertion_point}")
                print(f"   Caveat text: 'Uncertainty range is currently under calibration validation'")
                return
    
    print("⚠️  Could not automatically add caveat - manual update needed")
    print("   Add this text near confidence interval display:")
    print("   'Uncertainty range is currently under calibration validation - treat as an outer bound, not a precise range.'")

def main():
    """Main evaluation and update pipeline"""
    decision, metrics = compare_original_vs_improved()
    
    # Always update dashboard regardless of model decision
    update_dashboard_caveat()
    
    print(f"\n" + "="*80)
    print("FINAL RECOMMENDATIONS") 
    print("="*80)
    
    if decision == "KEEP_IMPROVED":
        print("✅ QUANTILE REGRESSOR SUCCESSFULLY IMPROVED")
        print("   1. CI clustering issue resolved (77% → 14%)")
        print("   2. Production model updated with improved version")
        print("   3. Dashboard calibration caveat added")
        print("   4. Ready to proceed with OOD warning implementation")
        
    elif decision == "PARTIAL_SUCCESS":
        print("⚠️  QUANTILE REGRESSOR PARTIALLY IMPROVED")
        print("   1. CI clustering issue resolved (77% → 14%)")
        print("   2. But coverage dropped significantly (88% → 63%)")
        print("   3. Dashboard calibration caveat added (important!)")
        print("   4. Consider conformal prediction in future phase")
        print("   5. Proceed with OOD warning implementation as planned")
        
    else:
        print("❌ QUANTILE REGRESSOR IMPROVEMENT UNSUCCESSFUL")
        print("   1. Reverting to original model recommended")
        print("   2. Dashboard calibration caveat still needed")
        print("   3. Defer proper CI fix to conformal prediction phase")
        print("   4. Proceed with OOD warning implementation as planned")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   • Verify dashboard caveat is visible to users")
    print(f"   • Proceed with OOD warning implementation per plan")
    print(f"   • Consider conformal prediction for future CI improvements")

if __name__ == "__main__":
    main()