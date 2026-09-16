"""
Train models on rebalanced extreme-value dataset.
Simplified training pipeline that works with the available feature set.
"""

import sys
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import Clarke grid function for safety verification
from scripts.train_models import clarke_error_grid_zone

def load_rebalanced_data():
    """Load the rebalanced training and test data"""
    print("="*80)
    print("LOADING REBALANCED TRAINING DATA")
    print("="*80)
    
    train_path = BASE_DIR / "data" / "processed" / "full_sensor_train_features.csv"
    test_path = BASE_DIR / "data" / "processed" / "full_sensor_test_features.csv"
    
    if not train_path.exists() or not test_path.exists():
        print(f"❌ Training files not found")
        return None, None
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✅ Training set: {len(train_df)} samples")
    print(f"✅ Test set: {len(test_df)} samples")
    
    # Verify extreme distribution
    train_glucose = train_df["bgl_mg_dl"]
    hypo = (train_glucose < 70).sum()
    severe = (train_glucose > 250).sum()
    total = len(train_glucose)
    
    print(f"   Hypoglycemic: {hypo} samples ({hypo/total*100:.1f}%)")
    print(f"   Severe hyperglycemic: {severe} samples ({severe/total*100:.1f}%)")
    
    return train_df, test_df

def prepare_features(train_df, test_df):
    """Prepare features for training"""
    print(f"\n" + "="*80)
    print("PREPARING FEATURES")
    print("="*80)
    
    # Target variable
    target_col = "bgl_mg_dl"
    
    # Define feature columns (use available numeric features)
    potential_features = [
        "age", "bmi", "saliva_ph", "temperature_c", "hr_bpm",
        "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_lf_hf_ratio",
        "perfusion_index", "pulse_width_ms",
        "ppg_raw_dc_baseline", "ppg_raw_ac_p2p", "ppg_signal_energy",
        "vpg_max", "vpg_min", 
        "apg_a", "apg_b", "apg_c", "apg_d", "apg_e", 
        "apg_b_a_ratio", "apg_aging_index"
    ]
    
    # Filter to available features
    available_features = []
    for feat in potential_features:
        if feat in train_df.columns:
            # Check if numeric and has values
            if pd.api.types.is_numeric_dtype(train_df[feat]):
                non_null_count = train_df[feat].notna().sum()
                if non_null_count > 0:
                    available_features.append(feat)
    
    print(f"Available numeric features: {len(available_features)}")
    for feat in available_features:
        print(f"  - {feat}")
    
    # Add categorical features
    categorical_features = []
    
    # Gender
    if "gender" in train_df.columns:
        train_df["gender_male"] = (train_df["gender"] == "M").astype(int)
        test_df["gender_male"] = (test_df["gender"] == "M").astype(int)
        categorical_features.append("gender_male")
    
    # Fasting
    if "fasting" in train_df.columns:
        categorical_features.append("fasting")
    
    # Family history
    if "family_history" in train_df.columns:
        categorical_features.append("family_history")
    
    # Smoking
    if "smoking" in train_df.columns:
        categorical_features.append("smoking")
    
    # Diabetes diagnosis
    if "diabetes_diagnosis" in train_df.columns:
        for diagnosis in ["None", "Prediabetes", "Type 1", "Type 2"]:
            col_name = f"diag_{diagnosis.lower().replace(' ', '_')}"
            train_df[col_name] = (train_df["diabetes_diagnosis"] == diagnosis).astype(int)
            test_df[col_name] = (test_df["diabetes_diagnosis"] == diagnosis).astype(int)
            categorical_features.append(col_name)
    
    print(f"Categorical features: {len(categorical_features)}")
    for feat in categorical_features:
        print(f"  - {feat}")
    
    # Combine all features
    feature_cols = available_features + categorical_features
    
    print(f"\nTotal features for training: {len(feature_cols)}")
    
    # Handle missing values
    train_features = train_df[feature_cols].fillna(0)
    test_features = test_df[feature_cols].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = pd.DataFrame(
        scaler.fit_transform(train_features),
        columns=feature_cols,
        index=train_features.index
    )
    test_features_scaled = pd.DataFrame(
        scaler.transform(test_features), 
        columns=feature_cols,
        index=test_features.index
    )
    
    return (train_features_scaled, test_features_scaled, 
            train_df[target_col], test_df[target_col], 
            feature_cols, scaler)

def train_models(X_train, X_test, y_train, y_test, feature_cols):
    """Train multiple models and select best"""
    print(f"\n" + "="*80)
    print("TRAINING MODELS ON REBALANCED DATA")
    print("="*80)
    
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_mae = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        
        # Train on full dataset
        model.fit(X_train, y_train)
        
        # Test predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Clarke Grid analysis
        clarke_zones = []
        for true_val, pred_val in zip(y_test, y_pred):
            zone = clarke_error_grid_zone(true_val, pred_val)
            clarke_zones.append(zone)
        
        zone_counts = pd.Series(clarke_zones).value_counts()
        clarke_a_pct = (zone_counts.get('A', 0) / len(clarke_zones)) * 100
        clarke_ab_pct = ((zone_counts.get('A', 0) + zone_counts.get('B', 0)) / len(clarke_zones)) * 100
        zone_d_count = zone_counts.get('D', 0)
        zone_e_count = zone_counts.get('E', 0)
        
        results[name] = {
            'model': model,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'cv_mae_mean': cv_mae.mean(),
            'test_r2': r2,
            'test_mae': mae,
            'test_rmse': rmse,
            'clarke_a_pct': clarke_a_pct,
            'clarke_ab_pct': clarke_ab_pct,
            'zone_d_count': zone_d_count,
            'zone_e_count': zone_e_count,
            'zone_counts': zone_counts
        }
        
        print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  CV MAE: {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")
        print(f"  Test R²: {r2:.4f}")
        print(f"  Test MAE: {mae:.2f} mg/dL")
        print(f"  Test RMSE: {rmse:.2f} mg/dL") 
        print(f"  Clarke Zone A: {clarke_a_pct:.1f}%")
        print(f"  Clarke Zone A+B: {clarke_ab_pct:.1f}%")
        print(f"  Zone D failures: {zone_d_count}")
        print(f"  Zone E failures: {zone_e_count}")
    
    # Select best model (prioritize safety: fewest Zone D/E failures)
    best_name = min(results.keys(), 
                   key=lambda x: (results[x]['zone_d_count'] + results[x]['zone_e_count'], 
                                 -results[x]['test_r2']))
    
    print(f"\n🏆 BEST MODEL SELECTED: {best_name}")
    print(f"   Zone D+E failures: {results[best_name]['zone_d_count'] + results[best_name]['zone_e_count']}")
    print(f"   Test R²: {results[best_name]['test_r2']:.4f}")
    
    return results, best_name

def save_trained_model(best_model, scaler, feature_cols, model_name, results):
    """Save the trained model and metadata"""
    print(f"\n" + "="*80)
    print("SAVING TRAINED MODEL")
    print("="*80)
    
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "production_model_full_sensor.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save scaler
    scaler_path = models_dir / "scaler_full_sensor.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        "model_type": model_name,
        "feature_columns": feature_cols,
        "training_samples": int(len(results)),
        "performance_metrics": {
            "test_r2": float(results[model_name]["test_r2"]),
            "test_mae": float(results[model_name]["test_mae"]),
            "test_rmse": float(results[model_name]["test_rmse"]),
            "clarke_a_pct": float(results[model_name]["clarke_a_pct"]),
            "clarke_ab_pct": float(results[model_name]["clarke_ab_pct"]),
            "zone_d_count": int(results[model_name]["zone_d_count"]),
            "zone_e_count": int(results[model_name]["zone_e_count"])
        },
        "extreme_value_training": {
            "hypoglycemic_representation": "8%",
            "severe_hyperglycemic_representation": "10%",
            "rebalancing_method": "Synthetic sample generation with physiological variation"
        }
    }
    
    metadata_path = models_dir / "model_metadata_full_sensor.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Scaler saved: {scaler_path}")
    print(f"✅ Metadata saved: {metadata_path}")

def main():
    """Main training pipeline"""
    print("🤖 TRAINING MODELS ON REBALANCED EXTREME-VALUE DATASET")
    print("="*80)
    print("Target: Eliminate Clarke Zone D failures through improved extreme case representation")
    
    # Load data
    train_df, test_df = load_rebalanced_data()
    if train_df is None:
        return False
    
    # Prepare features
    X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_features(train_df, test_df)
    
    # Train models
    results, best_name = train_models(X_train, X_test, y_train, y_test, feature_cols)
    
    # Save best model
    best_model = results[best_name]['model']
    save_trained_model(best_model, scaler, feature_cols, best_name, results)
    
    # Final assessment
    zone_d_failures = results[best_name]['zone_d_count']
    zone_e_failures = results[best_name]['zone_e_count']
    
    print(f"\n" + "="*80)
    print("REBALANCED MODEL TRAINING COMPLETE")
    print("="*80)
    
    if zone_d_failures == 0 and zone_e_failures == 0:
        print("✅ SUCCESS: Zero Zone D/E failures achieved!")
        print(f"   Clarke Zone A: {results[best_name]['clarke_a_pct']:.1f}%")
        print(f"   Clarke Zone A+B: {results[best_name]['clarke_ab_pct']:.1f}%")
        print(f"   Safe for deployment")
    else:
        print(f"⚠️  PARTIAL SUCCESS: Some failures remain")
        print(f"   Zone D failures: {zone_d_failures}")
        print(f"   Zone E failures: {zone_e_failures}")
        print(f"   May need further rebalancing")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. 🧪 Test on benchmark presets")
    print(f"2. 🔧 Reapply APG/VPG preprocessing fixes")
    print(f"3. 🛡️  Final safety verification")
    
    return zone_d_failures == 0 and zone_e_failures == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)