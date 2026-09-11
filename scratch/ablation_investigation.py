import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from scripts.train_models import compute_metrics, clarke_error_grid_zone

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

df_tr = pd.read_csv(DATA_DIR / "full_sensor_train_features.csv")
df_te = pd.read_csv(DATA_DIR / "full_sensor_test_features.csv")
y_tr = df_tr["bgl_mg_dl"]
y_te = df_te["bgl_mg_dl"]

with open(REPORTS_DIR / "features_manifest_full_sensor.json") as f:
    manifest = json.load(f)

all_feats = manifest["scaled_numeric_features"] + manifest["categorical_and_binary_features"]

ppg_morphology_cols = [
    "ppg_raw_dc_baseline_scaled", "ppg_raw_ac_p2p_scaled", "ppg_systolic_peak_scaled",
    "ppg_diastolic_peak_scaled", "ppg_trough_scaled", "perfusion_index_scaled",
    "ppg_signal_energy_scaled", "pulse_pressure_scaled", "pulse_width_ms_scaled",
    "trough_to_trough_ms_scaled", "dicrotic_notch_amp_scaled", "dicrotic_ratio_scaled",
    "vpg_max_scaled", "vpg_min_scaled", "apg_a_scaled", "apg_b_scaled", "apg_c_scaled",
    "apg_d_scaled", "apg_e_scaled", "apg_b_a_ratio_scaled", "apg_aging_index_scaled",
    "ppg_hr_bpm_scaled"
]
no_ppg_feats = [c for c in all_feats if c not in ppg_morphology_cols]

print("="*80)
print("1. PPG-ALONE ISOLATION BENCHMARK (Pure PPG Waveform / Derivative Signal)")
print("="*80)
for seed in [42, 123, 999]:
    rf_ppg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1)
    rf_ppg.fit(df_tr[ppg_morphology_cols], y_tr)
    preds = rf_ppg.predict(df_te[ppg_morphology_cols])
    m = compute_metrics(y_te.values, preds)
    print(f"  PPG-Only RF (seed={seed}) -> R2: {m['R2']:.4f} | MAE: {m['MAE']} mg/dL | RMSE: {m['RMSE']} mg/dL | MARD: {m['MARD']}% | Clarke A: {m['Zone_A']}% | Clarke A+B: {m['Zone_AB']}%")

# XGBoost PPG alone
xgb_ppg = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
xgb_ppg.fit(df_tr[ppg_morphology_cols], y_tr)
m_xgb_ppg = compute_metrics(y_te.values, xgb_ppg.predict(df_te[ppg_morphology_cols]))
print(f"  PPG-Only XGBoost (seed=42) -> R2: {m_xgb_ppg['R2']:.4f} | MAE: {m_xgb_ppg['MAE']} mg/dL | RMSE: {m_xgb_ppg['RMSE']} mg/dL | MARD: {m_xgb_ppg['MARD']}% | Clarke A: {m_xgb_ppg['Zone_A']}% | Clarke A+B: {m_xgb_ppg['Zone_AB']}%")

print("\n" + "="*80)
print("2. MULTI-SEED STABILITY AUDIT: Full Sensor Baseline vs. No PPG Morphology")
print("="*80)
for seed in [42, 123, 999]:
    rf_full = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1)
    rf_full.fit(df_tr[all_feats], y_tr)
    m_full = compute_metrics(y_te.values, rf_full.predict(df_te[all_feats]))

    rf_no = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1)
    rf_no.fit(df_tr[no_ppg_feats], y_tr)
    m_no = compute_metrics(y_te.values, rf_no.predict(df_te[no_ppg_feats]))

    print(f"  Seed {seed:3d}: Full Baseline R2 = {m_full['R2']:.4f}, MAE = {m_full['MAE']:.2f} mg/dL | No PPG Morphology R2 = {m_no['R2']:.4f}, MAE = {m_no['MAE']:.2f} mg/dL (Delta R2: {m_no['R2']-m_full['R2']:+.4f})")

print("\n" + "="*80)
print("3. TYPE 1 CLARKE ERROR GRID OUTLIER AUDIT (95.83% = 23/24)")
print("="*80)
with open(MODELS_DIR / "model_metadata_full_sensor.json") as f:
    meta = json.load(f)

with open(MODELS_DIR / "production_model_full_sensor.pkl", "rb") as f:
    prod_model = pickle.load(f)

# Predict on test set
t1_mask = (df_te["diabetes_diagnosis"] == "Type 1")
t1_df = df_te[t1_mask].copy()
t1_preds = prod_model.predict(t1_df[all_feats])
t1_true = t1_df["bgl_mg_dl"].values

for idx, (r_id, p_id, ref, est) in enumerate(zip(t1_df["reading_id"], t1_df["participant_id"], t1_true, t1_preds)):
    zone = clarke_error_grid_zone(ref, est)
    err = est - ref
    pct_err = (err / ref) * 100.0
    is_outlier = (zone not in ["A", "B"])
    flag = ">>> OUTLIER <<<" if is_outlier else ""
    print(f"  T1 Sample {idx+1:02d} [{r_id} | {p_id}]: Ref={ref:5.1f} mg/dL, Est={est:5.1f} mg/dL, Err={err:+5.1f} mg/dL ({pct_err:+5.1f}%) -> Zone {zone} {flag}")
