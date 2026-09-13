"""
Feature Engineering and Model-Ready Feature Matrix Construction.

Builds two distinct, non-leaking feature matrices and scalers:
1. Full-Sensor Feature Pipeline (PPG, HRV, pH, Temp, Demographics, Clinical)
   -> full_sensor_train_features.csv / full_sensor_test_features.csv
   -> models/scaler_full_sensor.pkl
   -> reports/features_manifest_full_sensor.json

2. Tabular-Only Feature Pipeline (Demographics, Clinical, Diagnosis, BMI)
   -> tabular_train_features.csv / tabular_test_features.csv
   -> models/scaler_tabular.pkl
   -> reports/features_manifest_tabular.json

Categorical Encoding Rationale:
- `gender`: Binary encoding (1 for M, 0 for F) since cardinality is 2 and no arbitrary scale is introduced.
- `diabetes_diagnosis`: One-hot encoded ('diag_None', 'diag_Prediabetes', 'diag_Type1', 'diag_Type2').
  Rationale: Diagnosis categories represent distinct pathophysiological etiologies (autoimmune destruction
  vs insulin resistance vs non-diabetic). An ordinal assignment would impose a false 1D distance metric.
- `glycemic_state_at_reading`: STRICTLY EXCLUDED from model feature inputs. Directly derived from bgl_mg_dl (target leakage).
  Preserved only as a metadata/stratification column.
- `diabetes_status`: STRICTLY EXCLUDED from model feature inputs. Preserved only as a metadata/stratification column.

Derived Features:
- `pulse_pressure`: ppg_systolic_peak - ppg_diastolic_peak (full_sensor branch).
- `bmi_category`: Standard clinical buckets (underweight <18.5, normal 18.5-24.9, overweight 25-29.9, obese >=30.0, unknown).
- `ph_deviation_from_mean`: saliva_ph - mean(train saliva_ph) (full_sensor branch, fit only on train split).
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"


# ==============================================================================
# Helper Encoders & Categorical Transformers
# ==============================================================================

def encode_bmi_category(bmi_series: pd.Series) -> pd.Series:
    """Categorizes BMI into standard WHO/CDC clinical buckets."""
    def categorize(val):
        if pd.isna(val):
            return "missing"
        val = float(val)
        if val < 18.5:
            return "underweight"
        elif val < 25.0:
            return "normal"
        elif val < 30.0:
            return "overweight"
        else:
            return "obese"
    return bmi_series.apply(categorize)


def encode_medication_features(med_series: pd.Series) -> pd.DataFrame:
    """Extracts standardized binary medication indicators."""
    taking_insulin = med_series.fillna("").apply(lambda s: 1 if "insulin" in str(s).lower() else 0)
    taking_oral = med_series.fillna("").apply(
        lambda s: 1 if any(w in str(s).lower() for w in ["pill", "metformin", "glipizide", "glyburide", "oral", "glimepiride", "pioglitazone", "rosiglitazone"]) else 0
    )
    taking_any_med = med_series.fillna("").apply(lambda s: 0 if str(s).strip() in ["", "None", "none", "nan"] else 1)
    
    return pd.DataFrame({
        "med_taking_insulin": taking_insulin,
        "med_taking_oral": taking_oral,
        "med_taking_any": taking_any_med
    }, index=med_series.index)


# ==============================================================================
# 1. Full-Sensor Feature Pipeline
# ==============================================================================

FULL_SENSOR_NUMERIC_RAW = [
    # MAX30102 Raw Signals & Morphology
    "ppg_raw_dc_baseline",
    "ppg_raw_ac_p2p",
    "ppg_systolic_peak",
    "ppg_diastolic_peak",
    "ppg_trough",
    "perfusion_index",
    "ppg_signal_energy",
    "pulse_pressure",  # Derived
    "hr_bpm",
    "ppg_hr_bpm",
    "pulse_width_ms",
    "trough_to_trough_ms",
    "dicrotic_notch_amp",
    "dicrotic_ratio",
    # VPG & APG Derivatives
    "vpg_max",
    "vpg_min",
    "apg_a",
    "apg_b",
    "apg_c",
    "apg_d",
    "apg_e",
    "apg_b_a_ratio",
    "apg_aging_index",
    # ECG-HRV
    "hrv_sdnn",
    "hrv_rmssd",
    "hrv_pnn50",
    "hrv_lf",
    "hrv_hf",
    "hrv_lf_hf_ratio",
    # Bio-Chemical & Thermal
    "saliva_ph",
    "ph_deviation_from_mean",  # Derived
    "temperature_c",
    # Personal Parameters
    "age",
    "bmi",
]

DIAGNOSIS_CATEGORIES = ["None", "Prediabetes", "Type 1", "Type 2"]
BMI_CATEGORIES = ["underweight", "normal", "overweight", "obese", "missing"]


def process_full_sensor_features(
    df_train_raw: pd.DataFrame, df_test_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, Dict[str, Any]]:
    """Constructs model-ready full-sensor training and testing matrices."""
    print("\n" + "="*80)
    print("BUILDING FULL-SENSOR FEATURE MATRICES (Branch: full_sensor)")
    print("="*80)

    train_fs = df_train_raw[df_train_raw["training_branch"] == "full_sensor"].copy()
    test_fs = df_test_raw[df_test_raw["training_branch"] == "full_sensor"].copy()

    print(f"  Full-Sensor Train Rows: {len(train_fs)} | Test Rows: {len(test_fs)}")

    # 1. Derived Features: Pulse Pressure
    train_fs["pulse_pressure"] = train_fs["ppg_systolic_peak"] - train_fs["ppg_diastolic_peak"]
    test_fs["pulse_pressure"] = test_fs["ppg_systolic_peak"] - test_fs["ppg_diastolic_peak"]

    # 2. Derived Features: Saliva pH Deviation from Train Mean (avoiding test leakage)
    train_ph_mean = float(train_fs["saliva_ph"].dropna().mean())
    train_fs["ph_deviation_from_mean"] = train_fs["saliva_ph"] - train_ph_mean
    test_fs["ph_deviation_from_mean"] = test_fs["saliva_ph"] - train_ph_mean
    print(f"  Saliva pH Train Mean Baseline: {train_ph_mean:.4f}")

    # 3. Categorical Encodings
    # Gender (Binary M=1, F=0)
    for df in [train_fs, test_fs]:
        df["gender_male"] = (df["gender"] == "M").astype(int)

    # Diabetes Diagnosis (One-Hot)
    for df in [train_fs, test_fs]:
        for diag in DIAGNOSIS_CATEGORIES:
            col_name = f"diag_{diag.replace(' ', '_').lower()}"
            df[col_name] = (df["diabetes_diagnosis"] == diag).astype(int)

    # BMI Category (One-Hot)
    train_fs["bmi_category"] = encode_bmi_category(train_fs["bmi"])
    test_fs["bmi_category"] = encode_bmi_category(test_fs["bmi"])
    for df in [train_fs, test_fs]:
        for b_cat in BMI_CATEGORIES:
            col_name = f"bmi_cat_{b_cat}"
            df[col_name] = (df["bmi_category"] == b_cat).astype(int)

    # Clinical Binary Indicators
    for df in [train_fs, test_fs]:
        df["family_history"] = df["family_history"].fillna(0).astype(int)
        df["smoking"] = df["smoking"].fillna(0).astype(int)
        df["fasting"] = df["fasting"].fillna(0).astype(int)
        med_df = encode_medication_features(df["medication"])
        for m_col in med_df.columns:
            df[m_col] = med_df[m_col]

    # Explicit Model Feature List (Ensuring diabetes_status and glycemic_state are strictly EXCLUDED)
    diag_onehot_cols = [f"diag_{diag.replace(' ', '_').lower()}" for diag in DIAGNOSIS_CATEGORIES]
    bmi_onehot_cols = [f"bmi_cat_{b_cat}" for b_cat in BMI_CATEGORIES]
    med_cols = ["med_taking_insulin", "med_taking_oral", "med_taking_any"]
    clinical_binary_cols = ["gender_male", "family_history", "smoking", "fasting"]

    feature_cols = (
        FULL_SENSOR_NUMERIC_RAW
        + diag_onehot_cols
        + bmi_onehot_cols
        + med_cols
        + clinical_binary_cols
    )

    # Confirm exclusion of diabetes_status and glycemic_state leakage
    assert "diabetes_status" not in feature_cols, "CRITICAL ERROR: diabetes_status leaked into feature set!"
    assert not any("glycemic_state" in c for c in feature_cols), "CRITICAL ERROR: glycemic_state leaked into feature set!"
    print(f"  >>> LEAKAGE CHECK CONFIRMED: diabetes_status & glycemic_state excluded from model features ({len(feature_cols)} total features).")

    # Impute train medians for any missing numericals in train & test
    for col in FULL_SENSOR_NUMERIC_RAW:
        med_val = float(train_fs[col].median())
        train_fs[col] = train_fs[col].fillna(med_val)
        test_fs[col] = test_fs[col].fillna(med_val)

    # Fit StandardScaler strictly on Train Split
    scaler = StandardScaler()
    train_scaled_arr = scaler.fit_transform(train_fs[FULL_SENSOR_NUMERIC_RAW])
    test_scaled_arr = scaler.transform(test_fs[FULL_SENSOR_NUMERIC_RAW])

    scaled_cols = [f"{col}_scaled" for col in FULL_SENSOR_NUMERIC_RAW]
    train_scaled_df = pd.DataFrame(train_scaled_arr, columns=scaled_cols, index=train_fs.index)
    test_scaled_df = pd.DataFrame(test_scaled_arr, columns=scaled_cols, index=test_fs.index)

    # Metadata & Target Columns (preserved for downstream evaluation & identification)
    meta_cols = [
        "reading_id",
        "participant_id",
        "data_source",
        "training_branch",
        "diabetes_diagnosis",
        "glycemic_state_at_reading",
        "diabetes_status",  # Stratification/metadata only
        "bgl_mg_dl"         # Target label
    ]

    # Combine into model-ready dataframe
    train_out = pd.concat([train_fs[meta_cols], train_fs[feature_cols], train_scaled_df], axis=1)
    test_out = pd.concat([test_fs[meta_cols], test_fs[feature_cols], test_scaled_df], axis=1)

    manifest = {
        "training_branch": "full_sensor",
        "num_train_samples": len(train_out),
        "num_test_samples": len(test_out),
        "target_column": "bgl_mg_dl",
        "metadata_columns": meta_cols,
        "raw_numeric_features": FULL_SENSOR_NUMERIC_RAW,
        "scaled_numeric_features": scaled_cols,
        "categorical_and_binary_features": diag_onehot_cols + bmi_onehot_cols + med_cols + clinical_binary_cols,
        "all_feature_columns": feature_cols,
        "ph_mean_train_baseline": train_ph_mean
    }

    return train_out, test_out, scaler, manifest


# ==============================================================================
# 2. Tabular-Only Feature Pipeline
# ==============================================================================

TABULAR_NUMERIC_RAW = [
    "age",
    "bmi",
    "waist_circumference_cm",
]

def process_tabular_features(
    df_train_raw: pd.DataFrame, df_test_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, Dict[str, Any]]:
    """Constructs model-ready tabular-only training and testing matrices."""
    print("\n" + "="*80)
    print("BUILDING TABULAR-ONLY FEATURE MATRICES (Branch: tabular_only)")
    print("="*80)

    train_tab = df_train_raw[df_train_raw["training_branch"] == "tabular_only"].copy()
    test_tab = df_test_raw[df_test_raw["training_branch"] == "tabular_only"].copy()

    print(f"  Tabular Train Rows: {len(train_tab)} | Test Rows: {len(test_tab)}")

    # 1. Categorical & Clinical Encodings
    for df in [train_tab, test_tab]:
        # Gender (Binary M=1, F=0)
        df["gender_male"] = (df["gender"] == "M").astype(int)

        # Race / Ethnicity (ADA Risk Test Groupings)
        r = df["race_ethnicity"].fillna("unknown").astype(str).str.lower()
        df["race_white"] = (r == "non_hispanic_white").astype(int)
        df["race_black"] = (r == "non_hispanic_black").astype(int)
        df["race_hispanic"] = (r.isin(["mexican_american", "other_hispanic"])).astype(int)
        df["race_asian"] = (r == "non_hispanic_asian").astype(int)
        df["race_other"] = (r.isin(["other_multiracial", "unknown"])).astype(int)

        # Physical Activity Level (Active / Moderate / Sedentary)
        pa = df["physical_activity_level"].fillna("unknown").astype(str).str.lower()
        df["phys_act_active"] = (pa == "active").astype(int)
        df["phys_act_moderate"] = (pa == "moderate").astype(int)
        df["phys_act_sedentary"] = (pa == "sedentary").astype(int)

        # Hypertension & High Cholesterol History
        df["hypertension"] = (df["hypertension"] == 1.0).astype(int)
        df["high_cholesterol"] = (df["high_cholesterol"] == 1.0).astype(int)

        # Gestational Diabetes History (Women only; Men coded as distinct NA category)
        gdm = df["gestational_diabetes"].fillna("unknown").astype(str).str.lower()
        df["gdm_positive"] = (gdm == "yes").astype(int)
        df["gdm_negative"] = (gdm == "no").astype(int)
        df["gdm_male_na"] = (gdm == "not_applicable").astype(int)
        df["gdm_unknown"] = (gdm.isin(["unknown", "nan"])).astype(int)

        # Central Obesity / Waist Circumference Missing Indicator
        df["waist_cm_missing"] = df["waist_circumference_cm"].isna().astype(int)

        # Diabetes Diagnosis (One-Hot)
        for diag in DIAGNOSIS_CATEGORIES:
            col_name = f"diag_{diag.replace(' ', '_').lower()}"
            df[col_name] = (df["diabetes_diagnosis"] == diag).astype(int)

        # BMI Category (One-Hot)
        df["bmi_category"] = encode_bmi_category(df["bmi"])
        for b_cat in BMI_CATEGORIES:
            col_name = f"bmi_cat_{b_cat}"
            df[col_name] = (df["bmi_category"] == b_cat).astype(int)

        # Clinical Binary Indicators
        df["family_history"] = df["family_history"].fillna(0).astype(int)
        df["smoking"] = df["smoking"].fillna(0).astype(int)
        df["fasting"] = df["fasting"].fillna(0).astype(int)
        df["bgl_is_hba1c_derived"] = df["bgl_is_hba1c_derived"].fillna(False).astype(int)
        med_df = encode_medication_features(df["medication"])
        for m_col in med_df.columns:
            df[m_col] = med_df[m_col]

    # Explicit Model Feature List
    diag_onehot_cols = [f"diag_{diag.replace(' ', '_').lower()}" for diag in DIAGNOSIS_CATEGORIES]
    bmi_onehot_cols = [f"bmi_cat_{b_cat}" for b_cat in BMI_CATEGORIES]
    med_cols = ["med_taking_insulin", "med_taking_oral", "med_taking_any"]
    race_cols = ["race_white", "race_black", "race_hispanic", "race_asian", "race_other"]
    pa_cols = ["phys_act_active", "phys_act_moderate", "phys_act_sedentary"]
    gdm_cols = ["gdm_positive", "gdm_negative", "gdm_male_na", "gdm_unknown"]
    clinical_binary_cols = [
        "gender_male", "family_history", "smoking", "fasting", "bgl_is_hba1c_derived",
        "hypertension", "high_cholesterol", "waist_cm_missing"
    ]

    feature_cols = (
        TABULAR_NUMERIC_RAW
        + diag_onehot_cols
        + bmi_onehot_cols
        + med_cols
        + race_cols
        + pa_cols
        + gdm_cols
        + clinical_binary_cols
    )

    # Confirm exclusion of diabetes_status and glycemic_state
    assert "diabetes_status" not in feature_cols, "CRITICAL ERROR: diabetes_status leaked into feature set!"
    assert not any("glycemic_state" in c for c in feature_cols), "CRITICAL ERROR: glycemic_state leaked into feature set!"
    print(f"  >>> LEAKAGE CHECK CONFIRMED: diabetes_status & glycemic_state excluded from model features ({len(feature_cols)} total features).")

    # Impute train medians for tabular numerics (age, bmi, waist_circumference_cm)
    for col in TABULAR_NUMERIC_RAW:
        med_val = float(train_tab[col].dropna().median())
        train_tab[col] = train_tab[col].fillna(med_val)
        test_tab[col] = test_tab[col].fillna(med_val)

    # Fit StandardScaler strictly on Train Split
    scaler = StandardScaler()
    train_scaled_arr = scaler.fit_transform(train_tab[TABULAR_NUMERIC_RAW])
    test_scaled_arr = scaler.transform(test_tab[TABULAR_NUMERIC_RAW])

    scaled_cols = [f"{col}_scaled" for col in TABULAR_NUMERIC_RAW]
    train_scaled_df = pd.DataFrame(train_scaled_arr, columns=scaled_cols, index=train_tab.index)
    test_scaled_df = pd.DataFrame(test_scaled_arr, columns=scaled_cols, index=test_tab.index)

    # Metadata & Target Columns
    meta_cols = [
        "reading_id",
        "participant_id",
        "data_source",
        "training_branch",
        "diabetes_diagnosis",
        "glycemic_state_at_reading",
        "diabetes_status",  # Stratification/metadata only
        "bgl_mg_dl"         # Target label
    ]

    # Combine into model-ready dataframe
    train_out = pd.concat([train_tab[meta_cols], train_tab[feature_cols], train_scaled_df], axis=1)
    test_out = pd.concat([test_tab[meta_cols], test_tab[feature_cols], test_scaled_df], axis=1)

    manifest = {
        "training_branch": "tabular_only",
        "num_train_samples": len(train_out),
        "num_test_samples": len(test_out),
        "target_column": "bgl_mg_dl",
        "metadata_columns": meta_cols,
        "raw_numeric_features": TABULAR_NUMERIC_RAW,
        "scaled_numeric_features": scaled_cols,
        "categorical_and_binary_features": diag_onehot_cols + bmi_onehot_cols + med_cols + race_cols + pa_cols + gdm_cols + clinical_binary_cols,
        "all_feature_columns": feature_cols
    }

    return train_out, test_out, scaler, manifest


# ==============================================================================
# Main Execution Entry Point
# ==============================================================================

def main():
    print("="*80)
    print("STARTING MODEL-READY FEATURE MATRIX GENERATION")
    print("="*80)

    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing train.csv or test.csv in {PROCESSED_DIR}")

    df_train_raw = pd.read_csv(TRAIN_CSV)
    df_test_raw = pd.read_csv(TEST_CSV)
    print(f"Loaded Raw Train: {len(df_train_raw):,} rows | Raw Test: {len(df_test_raw):,} rows")

    # 1. Process Full-Sensor Branch
    fs_train_df, fs_test_df, fs_scaler, fs_manifest = process_full_sensor_features(df_train_raw, df_test_raw)

    # Save Full-Sensor Artifacts
    fs_train_path = PROCESSED_DIR / "full_sensor_train_features.csv"
    fs_test_path = PROCESSED_DIR / "full_sensor_test_features.csv"
    fs_scaler_path = MODELS_DIR / "scaler_full_sensor.pkl"
    fs_manifest_path = REPORTS_DIR / "features_manifest_full_sensor.json"
    fs_manifest_proc_path = PROCESSED_DIR / "features_manifest_full_sensor.json"

    fs_train_df.to_csv(fs_train_path, index=False)
    fs_test_df.to_csv(fs_test_path, index=False)
    with open(fs_scaler_path, "wb") as f:
        pickle.dump(fs_scaler, f)
    with open(fs_manifest_path, "w") as f:
        json.dump(fs_manifest, f, indent=2)
    with open(fs_manifest_proc_path, "w") as f:
        json.dump(fs_manifest, f, indent=2)

    print(f"  Saved Full-Sensor Train: {fs_train_path} ({len(fs_train_df)} rows, {fs_train_df.shape[1]} cols)")
    print(f"  Saved Full-Sensor Test:  {fs_test_path} ({len(fs_test_df)} rows, {fs_test_df.shape[1]} cols)")
    print(f"  Saved Full-Sensor Scaler: {fs_scaler_path}")
    print(f"  Saved Full-Sensor Manifest: {fs_manifest_path}")

    # 2. Process Tabular-Only Branch
    tab_train_df, tab_test_df, tab_scaler, tab_manifest = process_tabular_features(df_train_raw, df_test_raw)

    # Save Tabular Artifacts
    tab_train_path = PROCESSED_DIR / "tabular_train_features.csv"
    tab_test_path = PROCESSED_DIR / "tabular_test_features.csv"
    tab_scaler_path = MODELS_DIR / "scaler_tabular.pkl"
    tab_manifest_path = REPORTS_DIR / "features_manifest_tabular.json"
    tab_manifest_proc_path = PROCESSED_DIR / "features_manifest_tabular.json"

    tab_train_df.to_csv(tab_train_path, index=False)
    tab_test_df.to_csv(tab_test_path, index=False)
    with open(tab_scaler_path, "wb") as f:
        pickle.dump(tab_scaler, f)
    with open(tab_manifest_path, "w") as f:
        json.dump(tab_manifest, f, indent=2)
    with open(tab_manifest_proc_path, "w") as f:
        json.dump(tab_manifest, f, indent=2)

    print(f"  Saved Tabular Train: {tab_train_path} ({len(tab_train_df)} rows, {tab_train_df.shape[1]} cols)")
    print(f"  Saved Tabular Test:  {tab_test_path} ({len(tab_test_df)} rows, {tab_test_df.shape[1]} cols)")
    print(f"  Saved Tabular Scaler: {tab_scaler_path}")
    print(f"  Saved Tabular Manifest: {tab_manifest_path}")

    # ==============================================================================
    # 3. Class Balance Report Across All Four Feature Matrices
    # ==============================================================================
    print("\n" + "="*80)
    print("CLASS BALANCE AUDIT ACROSS ALL FOUR OUTPUT FILES")
    print("="*80)

    files_dict = {
        "Full Sensor Train": fs_train_df,
        "Full Sensor Test": fs_test_df,
        "Tabular Train": tab_train_df,
        "Tabular Test": tab_test_df
    }

    all_statuses = sorted(list(set(df_train_raw["diabetes_status"].dropna().unique()).union(
        set(df_test_raw["diabetes_status"].dropna().unique())
    )))

    summary_records = []
    for status in all_statuses:
        rec = {"diabetes_status": status}
        for name, df in files_dict.items():
            cnt = int((df["diabetes_status"] == status).sum())
            pct = (cnt / len(df)) * 100 if len(df) > 0 else 0
            rec[f"{name} (N)"] = cnt
            rec[f"{name} (%)"] = f"{pct:.1f}%"
        summary_records.append(rec)

    balance_df = pd.DataFrame(summary_records)
    print("\n" + balance_df.to_string(index=False))

    # Save summary report to markdown
    report_md_path = REPORTS_DIR / "feature_matrices_report.md"
    with open(report_md_path, "w") as f:
        f.write("# Model-Ready Feature Matrices & Class Balance Report\n\n")
        f.write(f"**Generated On**: 2026-09-10\n\n")
        f.write("## 1. Feature Matrix Summary\n\n")
        f.write("| Feature Matrix File | Rows | Columns | Scaler | Manifest |\n")
        f.write("| :--- | :---: | :---: | :--- | :--- |\n")
        f.write(f"| `full_sensor_train_features.csv` | **{len(fs_train_df):,}** | **{fs_train_df.shape[1]}** | `models/scaler_full_sensor.pkl` | `reports/features_manifest_full_sensor.json` |\n")
        f.write(f"| `full_sensor_test_features.csv` | **{len(fs_test_df):,}** | **{fs_test_df.shape[1]}** | (Applied) | (Same Schema) |\n")
        f.write(f"| `tabular_train_features.csv` | **{len(tab_train_df):,}** | **{tab_train_df.shape[1]}** | `models/scaler_tabular.pkl` | `reports/features_manifest_tabular.json` |\n")
        f.write(f"| `tabular_test_features.csv` | **{len(tab_test_df):,}** | **{tab_test_df.shape[1]}** | (Applied) | (Same Schema) |\n\n")
        f.write("## 2. Class Balance per `diabetes_status` Across Feature Matrices\n\n")
        f.write(balance_df.to_markdown(index=False))
        f.write("\n\n## 3. Stratified Evaluation & Sample Sparsity Assessment for Phase 6\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> **Full-Sensor Sample Size Assessment**:\n")
        f.write(f"> - In `full_sensor_train_features.csv` ({len(fs_train_df)} rows across 120 participants) and `full_sensor_test_features.csv` ({len(fs_test_df)} rows across 30 participants), subcategories like `type2_controlled` ({int((fs_test_df['diabetes_status']=='type2_controlled').sum())} test rows) or `type2_uncontrolled` ({int((fs_test_df['diabetes_status']=='type2_uncontrolled').sum())} test rows) have modest sample counts.\n")
        f.write("> - For Phase 6 model evaluation, primary metrics (R², MAE, RMSE, Clarke Error Grid) should be evaluated on the aggregated test set and grouped by broader clinical categories (`diabetes_diagnosis`: None, Type 1, Type 2, Prediabetes) in addition to individual status slices to ensure statistical power.\n")

    print(f"\nSaved feature report to: {report_md_path}")
    print("\n" + "="*80)
    print("FEATURE PIPELINE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
