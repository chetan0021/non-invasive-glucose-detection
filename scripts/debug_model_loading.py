"""
Debug what model is actually loaded in production
"""

import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

def debug_production_model():
    """Check what's actually in the production model file"""
    print("="*80)
    print("DEBUGGING PRODUCTION MODEL LOADING")
    print("="*80)
    
    prod_path = MODELS_DIR / "quantile_regressor_full_sensor.pkl"
    
    print(f"Loading: {prod_path}")
    
    with open(prod_path, "rb") as f:
        model = pickle.load(f)
    
    print(f"Model keys: {list(model.keys())}")
    
    if "method" in model:
        print(f"Method: {model['method']}")
    
    if "conformal_coverage_pct" in model:
        print(f"Expected coverage: {model['conformal_coverage_pct']}%")
    
    if "calibration_margin" in model:
        print(f"Calibration margin: {model['calibration_margin']}")
    
    if "hyperparameters" in model:
        print(f"Hyperparameters: {model['hyperparameters']}")
    
    # Check if this is the conformal model or the backup
    is_conformal = "calibration_margin" in model
    print(f"\nIs conformal model: {'YES' if is_conformal else 'NO'}")
    
    # Also check backup
    backup_path = MODELS_DIR / "quantile_regressor_full_sensor_backup.pkl"
    if backup_path.exists():
        print(f"\nBackup model exists: {backup_path}")
        with open(backup_path, "rb") as f:
            backup_model = pickle.load(f)
        backup_coverage = backup_model.get("empirical_test_coverage_pct", "unknown")
        print(f"Backup coverage: {backup_coverage}%")
    
    # Check conformal model
    conformal_path = MODELS_DIR / "quantile_regressor_conformal_calibrated.pkl"  
    if conformal_path.exists():
        print(f"\nConformal model exists: {conformal_path}")
        with open(conformal_path, "rb") as f:
            conformal_model = pickle.load(f)
        conformal_coverage = conformal_model.get("conformal_coverage_pct", "unknown")
        print(f"Conformal coverage: {conformal_coverage}%")
        print(f"Conformal margin: {conformal_model.get('calibration_margin', 'unknown')}")

if __name__ == "__main__":
    debug_production_model()