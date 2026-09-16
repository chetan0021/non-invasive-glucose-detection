"""
Synthetic Training Data Generator for Non-Invasive Blood Glucose Prediction.

Grounds PPG morphology and ECG-HRV feature distributions in real fingertip data (BIDMC)
from reports/ppg_feature_distributions.json and data/interim/real_feature_pool.parquet.

Key Design Principles & Tuning Fixes:
1. Non-conflated Clinical Taxonomy:
   - `diabetes_diagnosis`: Invariant known diagnosis (None / Prediabetes / Type 1 / Type 2).
   - `glycemic_state_at_reading`: Instantaneous reading state (normal / elevated / high / very_high).
   - `diabetes_status`: Composite human-readable label where 'healthy' is strictly None + normal.
2. Balanced Correlation Structure:
   - Saliva pH: Moderate correlation (univariate R^2 in 0.20-0.35, Ahadian et al. 2025).
   - HRV LF/HF ratio: Moderate correlation (univariate R^2 in 0.10-0.25).
   - PPG Morphology: Moderate correlation (r in 0.35-0.55 range).
   - Safeguard assertion: Max univariate R^2 <= 0.50 enforced across all features.
3. Perfusion Index: Strictly within [0.50%, 5.00%] with realistic MAX30102 ADC counts.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
INTERIM_DIR = BASE_DIR / "data" / "interim"
REPORTS_DIR = BASE_DIR / "reports"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth_distributions() -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Loads distribution JSON and real feature pool parquet."""
    json_path = REPORTS_DIR / "ppg_feature_distributions.json"
    parquet_path = INTERIM_DIR / "real_feature_pool.parquet"

    with open(json_path, "r", encoding="utf-8") as f:
        dist_json = json.load(f)

    fingertip_stats = dist_json.get("placement_comparison", {}).get("fingertip", {})
    pooled_stats = dist_json.get("pooled_distribution", {})

    real_df = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()
    return dist_json, real_df


def sample_grounded_feature(
    feat_name: str,
    fingertip_stats: Dict[str, Any],
    pooled_stats: Dict[str, Any],
    age_adjust_factor: float = 0.0,
    n_samples: int = 1
) -> np.ndarray:
    """Samples physiological features with explicit population skew adjustments."""
    stats = fingertip_stats.get(feat_name)
    if stats is None or stats.get("count", 0) < 50:
        stats = pooled_stats.get(feat_name, {})

    mean_val = stats.get("mean", 0.0)
    std_val = stats.get("std", 1.0)
    p05 = stats.get("p05", mean_val - 1.645 * std_val)
    p50 = stats.get("p50", mean_val)
    p95 = stats.get("p95", mean_val + 1.645 * std_val)
    min_val = stats.get("min", p05)
    max_val = stats.get("max", p95)

    raw_samples = np.random.normal(loc=p50, scale=max(0.35 * (p95 - p05), 1e-4), size=n_samples)
    raw_samples = np.clip(raw_samples, min_val, max_val)

    if feat_name in ["hr_bpm", "ppg_hr_bpm"]:
        adjustment = age_adjust_factor * 12.0
        samples = np.clip(raw_samples + adjustment, 52.0, 135.0)
    elif feat_name == "pulse_width_ms":
        adjustment = age_adjust_factor * 35.0
        samples = np.clip(raw_samples + adjustment, 140.0, 360.0)
    elif feat_name in ["hrv_sdnn", "hrv_rmssd"]:
        multiplier = max(1.0 - 0.50 * age_adjust_factor, 0.45)
        samples = np.clip(raw_samples * multiplier, 8.0, 160.0)
    elif feat_name == "hrv_pnn50":
        multiplier = max(1.0 - 0.55 * age_adjust_factor, 0.1)
        samples = np.clip(raw_samples * multiplier, 0.0, 75.0)
    elif feat_name == "apg_b_a_ratio":
        adjustment = age_adjust_factor * 0.18
        samples = np.clip(raw_samples + adjustment, -1.30, -0.35)
    elif feat_name == "apg_aging_index":
        adjustment = age_adjust_factor * 0.28
        samples = np.clip(raw_samples + adjustment, -1.95, -0.40)
    else:
        samples = raw_samples

    return samples


def generate_synthetic_dataset(
    n_participants: int = 160,
    readings_per_participant_range: Tuple[int, int] = (3, 5),
    random_seed: int = 42
) -> pd.DataFrame:
    """Generates synthetic multi-modal dataset with orthogonal diagnosis and glycemic states."""
    np.random.seed(random_seed)
    dist_json, real_df = load_ground_truth_distributions()
    fingertip_stats = dist_json.get("placement_comparison", {}).get("fingertip", {})
    pooled_stats = dist_json.get("pooled_distribution", {})

    rows = []

    # Distribution of latent diabetes states across population
    # REBALANCED EXTREME VALUE SAMPLING: 8% hypoglycemic, 10% severe hyperglycemic
    # Original: 34% Healthy, 18% Prediabetes, 30% Type 2 (18% ctrl, 12% unctrl), 18% Type 1
    # Rebalanced: Add explicit extreme value strata
    
    # Calculate stratum sizes for target extreme representation
    target_hypo_pct = 0.08  # 8% hypoglycemic (<70)
    target_severe_pct = 0.10  # 10% severe hyperglycemic (>250)
    target_normal_pct = 0.82  # 82% normal range (70-250)
    
    # Map diabetes states to glucose ranges for stratification
    # Hypoglycemic stratum: Type 1 fasting with increased hypoglycemic probability
    # Severe hyperglycemic stratum: Type 1 + Type 2 uncontrolled with increased severe probability
    # Normal stratum: All others
    
    # Adjusted state probabilities to achieve target extreme representation
    state_probs = [0.28, 0.16, 0.20, 0.16, 0.20]  # Rebalanced for extreme cases
    diabetes_classes = ["healthy", "prediabetic", "type2_controlled", "type2_uncontrolled", "type1_extreme"]

    for p_idx in range(n_participants):
        p_id = f"SYNTH_{p_idx+1:03d}"
        d_state = np.random.choice(diabetes_classes, p=state_probs)

        # 1. Invariant Clinical Diagnosis & Demographics
        if d_state == "healthy":
            age = int(np.clip(int(np.random.normal(38.0, 12.0)), 18, 72))
            bmi = float(np.clip(float(np.random.normal(23.8, 2.6)), 18.5, 31.0))
            fam_hist = int(np.random.choice([0, 1], p=[0.82, 0.18]))
            diagnosis = "None"
            medication = "None"
        elif d_state == "prediabetic":
            age = int(np.clip(int(np.random.normal(49.0, 10.0)), 28, 75))
            bmi = float(np.clip(float(np.random.normal(28.4, 3.2)), 22.5, 38.0))
            fam_hist = int(np.random.choice([0, 1], p=[0.48, 0.52]))
            diagnosis = "Prediabetes"
            medication = np.random.choice(["None", "Metformin_Low"], p=[0.75, 0.25])
        elif d_state == "type2_controlled":
            age = int(np.clip(int(np.random.normal(57.0, 9.0)), 35, 78))
            bmi = float(np.clip(float(np.random.normal(30.6, 3.8)), 24.0, 42.0))
            fam_hist = int(np.random.choice([0, 1], p=[0.25, 0.75]))
            diagnosis = "Type 2"
            medication = np.random.choice(["Metformin", "Metformin+Sulfonylurea", "Metformin+DPP4"], p=[0.50, 0.30, 0.20])
        elif d_state == "type2_uncontrolled":
            age = int(np.clip(int(np.random.normal(61.0, 8.5)), 40, 80))
            bmi = float(np.clip(float(np.random.normal(32.8, 4.2)), 25.5, 45.0))
            fam_hist = int(np.random.choice([0, 1], p=[0.15, 0.85]))
            diagnosis = "Type 2"
            medication = np.random.choice(["Metformin+Sulfonylurea+Insulin", "Insulin_High_Dose"], p=[0.60, 0.40])
        else:  # type1_extreme (Enhanced for extreme value generation - hypoglycemic + severe hyperglycemic)
            age = int(np.clip(int(np.random.normal(29.0, 9.0)), 18, 55))
            bmi = float(np.clip(float(np.random.normal(23.5, 2.5)), 18.5, 30.0))
            fam_hist = int(np.random.choice([0, 1], p=[0.70, 0.30]))
            diagnosis = "Type 1"
            medication = np.random.choice(["Basal_Bolus_Insulin", "Insulin_Pump", "Automated_Insulin_Delivery"], p=[0.55, 0.30, 0.15])

        gender = np.random.choice(["M", "F"], p=[0.51, 0.49])
        smoking = int(np.random.choice([0, 1], p=[0.82, 0.18]))

        # Calculate Age & Health Skew Factor for BIDMC ICU Adjustment
        age_skew = (age - 66.0) / 40.0
        health_skew = {
            "healthy": -0.35,
            "prediabetic": -0.05,
            "type2_controlled": +0.15,
            "type2_uncontrolled": +0.40,
            "type1_extreme": +0.05
        }[d_state]
        combined_skew = float(0.65 * age_skew + 0.35 * health_skew)

        p_base_hr = float(sample_grounded_feature("hr_bpm", fingertip_stats, pooled_stats, combined_skew, 1)[0])
        p_base_sdnn = float(sample_grounded_feature("hrv_sdnn", fingertip_stats, pooled_stats, combined_skew, 1)[0])
        p_base_rmssd = float(sample_grounded_feature("hrv_rmssd", fingertip_stats, pooled_stats, combined_skew, 1)[0])
        p_base_pw = float(sample_grounded_feature("pulse_width_ms", fingertip_stats, pooled_stats, combined_skew, 1)[0])
        p_base_ba = float(sample_grounded_feature("apg_b_a_ratio", fingertip_stats, pooled_stats, combined_skew, 1)[0])
        p_base_ai = float(sample_grounded_feature("apg_aging_index", fingertip_stats, pooled_stats, combined_skew, 1)[0])

        # Type 1 extreme physiological adjustment: enhanced autonomic attenuation for extreme cases
        if d_state == "type1_extreme":
            p_base_sdnn = float(np.clip(p_base_sdnn * 0.88, 12.0, 130.0))
            p_base_rmssd = float(np.clip(p_base_rmssd * 0.85, 10.0, 120.0))

        n_readings = np.random.randint(readings_per_participant_range[0], readings_per_participant_range[1] + 1)
        contexts = ["fasting"] * max(1, n_readings // 2) + ["post-prandial"] * (n_readings - max(1, n_readings // 2))
        np.random.shuffle(contexts)

        for r_idx, ctx in enumerate(contexts):
            is_fasting = 1 if ctx == "fasting" else 0

            # 2. Blood Glucose Level (bgl_mg_dl)
            if d_state == "healthy":
                if is_fasting:
                    bgl = np.random.normal(83.5, 6.5)
                    bgl = np.clip(bgl, 70.0, 99.0)
                else:
                    bgl = np.random.normal(111.0, 11.5)
                    bgl = np.clip(bgl, 86.0, 138.0)
            elif d_state == "prediabetic":
                if is_fasting:
                    bgl = np.random.normal(111.0, 6.0)
                    bgl = np.clip(bgl, 100.0, 125.0)
                else:
                    bgl = np.random.normal(152.0, 13.0)
                    bgl = np.clip(bgl, 128.0, 185.0)
            elif d_state == "type2_controlled":
                if is_fasting:
                    bgl = np.random.normal(117.0, 9.5)
                    bgl = np.clip(bgl, 94.0, 142.0)
                else:
                    bgl = np.random.normal(164.0, 16.0)
                    bgl = np.clip(bgl, 130.0, 210.0)
            elif d_state == "type2_uncontrolled":
                if is_fasting:
                    bgl = np.random.normal(168.0, 24.0)
                    bgl = np.clip(bgl, 126.0, 245.0)
                else:
                    bgl = np.random.normal(242.0, 32.0)
                    bgl = np.clip(bgl, 175.0, 335.0)
            else:  # type1_extreme (Stratified extreme value generation for safety rebalancing)
                # Stratified sampling: 50% hypoglycemic, 35% severe hyperglycemic, 15% normal
                extreme_stratum = np.random.choice(["hypoglycemic", "severe_hyperglycemic", "normal"], 
                                                 p=[0.50, 0.35, 0.15])
                
                if extreme_stratum == "hypoglycemic":
                    # Hypoglycemic cases (<70 mg/dL) - dawn phenomenon, missed meals, insulin overdose
                    if is_fasting:
                        bgl = np.random.normal(58.0, 8.0)  # Fasting hypoglycemia
                        bgl = np.clip(bgl, 40.0, 69.0)
                    else:
                        bgl = np.random.normal(62.0, 6.0)  # Post-meal reactive hypoglycemia 
                        bgl = np.clip(bgl, 45.0, 69.0)
                        
                elif extreme_stratum == "severe_hyperglycemic":
                    # Severe hyperglycemic cases (>250 mg/dL) - DKA risk, poor control
                    if is_fasting:
                        bgl = np.random.normal(285.0, 35.0)  # Fasting severe hyperglycemia
                        bgl = np.clip(bgl, 251.0, 380.0)
                    else:
                        bgl = np.random.normal(320.0, 45.0)  # Post-meal severe excursion
                        bgl = np.clip(bgl, 260.0, 450.0)
                        
                else:  # normal (standard Type 1 range for comparison)
                    if is_fasting:
                        bgl = np.random.normal(136.0, 36.0)
                        bgl = np.clip(bgl, 70.0, 250.0)  # Constrain to normal-high range
                    else:
                        bgl = np.random.normal(185.0, 35.0)
                        bgl = np.clip(bgl, 90.0, 250.0)

            bgl = float(np.round(bgl, 1))
            glucose_z = (bgl - 120.0) / 55.0

            # 3. Glycemic State & Non-Conflated Diabetes Status (updated for extreme values)
            if bgl < 70.0:
                glycemic_state = "hypoglycemic"  # New category for <70
            elif bgl < 100.0:
                glycemic_state = "normal"
            elif bgl < 180.0:
                glycemic_state = "elevated"
            elif bgl < 250.0:
                glycemic_state = "high"
            else:
                glycemic_state = "very_high"  # ≥250 severe hyperglycemic

            # Composite diabetes_status Construction (updated for hypoglycemic cases)
            if diagnosis == "None":
                if glycemic_state == "hypoglycemic":
                    status_label = "healthy_hypoglycemic"
                elif glycemic_state == "normal":
                    status_label = "healthy"
                elif glycemic_state == "elevated":
                    status_label = "undiagnosed_elevated"
                else:
                    status_label = "undiagnosed_high"
            elif diagnosis == "Prediabetes":
                if glycemic_state == "hypoglycemic":
                    status_label = "prediabetes_hypoglycemic"
                elif glycemic_state == "normal":
                    status_label = "prediabetes_normoglycemic"
                elif glycemic_state == "elevated":
                    status_label = "prediabetes_elevated"
                else:
                    status_label = "prediabetes_high"
            elif diagnosis == "Type 2":
                if glycemic_state == "hypoglycemic":
                    status_label = "type2_hypoglycemic"
                elif glycemic_state == "normal":
                    status_label = "type2_normoglycemic"
                elif glycemic_state == "elevated":
                    status_label = "type2_controlled"
                elif glycemic_state == "high":
                    status_label = "type2_uncontrolled"
                else:
                    status_label = "type2_severe"
            else:  # Type 1
                if glycemic_state == "hypoglycemic":
                    status_label = "type1_hypoglycemic"
                elif glycemic_state == "normal":
                    status_label = "type1_normoglycemic"
                elif glycemic_state == "elevated":
                    status_label = "type1_elevated"
                elif glycemic_state == "high":
                    status_label = "type1_uncontrolled"
                else:
                    status_label = "type1_severe"

            # 4. Saliva pH Derivation (Ahadian et al. 2025: Target R^2 in 0.20 - 0.35)
            ph_signal = -0.115 * glucose_z
            ph_noise = np.random.normal(0.0, 0.19)
            ph_val = 7.33 + ph_signal + ph_noise
            if bgl > 230.0:
                ph_val -= np.random.uniform(0.08, 0.18)
            ph_val = float(np.clip(ph_val, 6.20, 7.60))
            ph_val = float(np.round(ph_val, 2))

            # 5. Body Temperature (temp_c)
            temp_base = 36.64 + 0.10 * (1 - is_fasting) + 0.05 * glucose_z
            temp_c = float(np.random.normal(temp_base, 0.16))
            temp_c = float(np.clip(temp_c, 36.20, 37.35))
            temp_c = float(np.round(temp_c, 2))

            # 6. MAX30102 Physical ADC Counts & Perfusion Index
            raw_dc = float(np.random.uniform(155000.0, 195000.0))
            ac_base = 2500.0 + 260.0 * glucose_z + 280.0 * (temp_c - 36.6)
            raw_ac = float(np.random.normal(ac_base, 620.0))
            raw_ac = float(np.clip(raw_ac, 1100.0, 4850.0))

            perfusion_index = float(np.round((raw_ac / raw_dc) * 100.0, 3))
            raw_sys_peak = float(np.round(raw_dc + 0.5 * raw_ac, 1))
            raw_trough = float(np.round(raw_dc - 0.5 * raw_ac, 1))

            ac_half = 0.5 * raw_ac
            ppg_energy = float(0.5 * (ac_half ** 2) + np.random.normal(0.0, 0.12 * (ac_half ** 2)))
            ppg_energy = float(max(ppg_energy, 10000.0))

            # 7. Multi-Modal PPG & HRV Feature Coupling
            pw_noise = np.random.normal(0, 12.0)
            reading_pw = p_base_pw + 24.0 * glucose_z + pw_noise
            reading_pw = float(np.clip(reading_pw, 145.0, 350.0))

            hr_noise = np.random.normal(0, 6.0)
            reading_hr = p_base_hr + 3.5 * (1 - is_fasting) + 4.8 * glucose_z + hr_noise
            reading_hr = float(np.clip(reading_hr, 52.0, 130.0))
            t2t_ms = float(60000.0 / reading_hr)

            dicrotic_ratio = float(np.clip(0.48 - 0.032 * glucose_z + np.random.normal(0, 0.070), 0.22, 0.65))
            raw_dia_peak = float(np.round(raw_trough + raw_ac * dicrotic_ratio, 1))
            dicrotic_notch_amp = float(np.round(raw_trough + raw_ac * (dicrotic_ratio * 0.72), 1))

            sdnn_shift = -15.0 * glucose_z + np.random.normal(0, 7.0)
            reading_sdnn = float(np.clip(p_base_sdnn + sdnn_shift, 6.0, 140.0))

            rmssd_shift = -14.0 * glucose_z + np.random.normal(0, 6.5)
            reading_rmssd = float(np.clip(p_base_rmssd + rmssd_shift, 5.0, 135.0))
            reading_pnn50 = float(np.clip(0.45 * reading_sdnn + np.random.normal(0, 5.0), 0.0, 65.0))

            lf_hf_noise = np.random.normal(0, 0.90)
            base_lf_hf = 1.55 + 0.30 * (1 - is_fasting) + 0.42 * glucose_z + lf_hf_noise
            reading_lf_hf = float(np.clip(base_lf_hf, 0.40, 8.50))
            reading_hf = float(np.clip(reading_rmssd * 14.0 + np.random.normal(0, 75.0), 20.0, 1800.0))
            reading_lf = float(reading_hf * reading_lf_hf)

            apg_a = float(np.random.normal(75.0, 15.0))
            ba_noise = np.random.normal(0, 0.09)
            apg_ba = float(np.clip(p_base_ba + 0.11 * glucose_z + ba_noise, -1.25, -0.38))

            ai_noise = np.random.normal(0, 0.16)
            apg_ai = float(np.clip(p_base_ai + 0.22 * glucose_z + ai_noise, -1.90, -0.45))

            apg_b = float(apg_a * apg_ba)
            apg_c = float(apg_a * np.random.uniform(0.15, 0.35))
            apg_d = float(apg_a * np.random.uniform(-0.35, -0.15))
            apg_e = float(apg_a * np.random.uniform(0.08, 0.22))

            slope_factor = float(np.random.normal(1.65, 0.22))
            vpg_max = float(raw_ac * slope_factor * (reading_hr / 60.0) + np.random.normal(0, 450.0))
            vpg_min = float(-raw_ac * (slope_factor * 0.82) * (reading_hr / 60.0) + np.random.normal(0, 350.0))

            row = {
                "reading_id": f"{p_id}_R{r_idx+1:02d}",
                "participant_id": p_id,
                "data_source": "synthetic",
                "training_branch": "full_sensor",
                "placement": "fingertip",
                # Non-conflated Diagnosis & Glycemic States
                "diabetes_diagnosis": diagnosis,
                "glycemic_state_at_reading": glycemic_state,
                "diabetes_status": status_label,
                "diabetes_type": diagnosis,  # Alias
                "context": ctx,
                "fasting": is_fasting,
                # Target
                "bgl_mg_dl": bgl,
                # Saliva & Temp
                "saliva_ph": ph_val,
                "temperature_c": temp_c,
                # Demographics
                "age": age,
                "gender": gender,
                "bmi": round(bmi, 1),
                "family_history": fam_hist,
                "medication": medication,
                "smoking": smoking,
                # MAX30102 Signal
                "ppg_raw_dc_baseline": raw_dc,
                "ppg_raw_ac_p2p": raw_ac,
                "ppg_systolic_peak": raw_sys_peak,
                "ppg_diastolic_peak": raw_dia_peak,
                "ppg_trough": raw_trough,
                "perfusion_index": perfusion_index,
                "ppg_signal_energy": round(ppg_energy, 1),
                # Morphological Features
                "hr_bpm": round(reading_hr, 1),
                "ppg_hr_bpm": round(reading_hr, 1),
                "pulse_width_ms": round(reading_pw, 1),
                "trough_to_trough_ms": round(t2t_ms, 1),
                "dicrotic_notch_amp": dicrotic_notch_amp,
                "dicrotic_ratio": round(dicrotic_ratio, 3),
                # APG / VPG
                "vpg_max": round(vpg_max, 1),
                "vpg_min": round(vpg_min, 1),
                "apg_a": round(apg_a, 2),
                "apg_b": round(apg_b, 2),
                "apg_c": round(apg_c, 2),
                "apg_d": round(apg_d, 2),
                "apg_e": round(apg_e, 2),
                "apg_b_a_ratio": round(apg_ba, 3),
                "apg_aging_index": round(apg_ai, 3),
                # HRV
                "hrv_sdnn": round(reading_sdnn, 2),
                "hrv_rmssd": round(reading_rmssd, 2),
                "hrv_pnn50": round(reading_pnn50, 2),
                "hrv_lf": round(reading_lf, 1),
                "hrv_hf": round(reading_hf, 1),
                "hrv_lf_hf_ratio": round(reading_lf_hf, 3),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def validate_and_report(df: pd.DataFrame):
    """Validates physical and statistical constraints."""
    print("=" * 80)
    print("SYNTHETIC DATASET VALIDATION & TAXONOMY AUDIT")
    print("=" * 80)
    print(f"Total Rows Generated: {len(df):,} across {df['participant_id'].nunique()} participants")
    print("\nDiabetes Diagnosis Breakdown:\n", df["diabetes_diagnosis"].value_counts().to_string())
    print("\nGlycemic State Breakdown:\n", df["glycemic_state_at_reading"].value_counts().to_string())
    print("\nComposite Diabetes Status Breakdown:\n", df["diabetes_status"].value_counts().to_string())

    # Verify no diagnosed diabetic is labeled "healthy"
    diagnosed_diabetics = df[df["diabetes_diagnosis"].isin(["Type 1", "Type 2"])]
    healthy_labeled = (diagnosed_diabetics["diabetes_status"] == "healthy").sum()
    assert healthy_labeled == 0, f"TAXONOMY ERROR: {healthy_labeled} diagnosed diabetics labeled 'healthy'!"
    print("\n>>> TAXONOMY ASSERTION PASSED: Zero diagnosed diabetics labeled 'healthy'.")

    # Perfusion Index
    pi = df["perfusion_index"]
    assert pi.min() >= 0.50 and pi.max() <= 5.00, "PI out of bounds!"
    print(f">>> PERFUSION INDEX PASSED: mean={pi.mean():.3f}%, min={pi.min():.3f}%, max={pi.max():.3f}%")

    # Univariate R^2 check
    features_to_check = [
        "saliva_ph", "hrv_lf_hf_ratio", "pulse_width_ms", "ppg_raw_ac_p2p",
        "ppg_signal_energy", "perfusion_index", "dicrotic_ratio", "apg_b_a_ratio",
        "apg_aging_index", "vpg_max", "vpg_min", "hr_bpm", "hrv_sdnn",
        "hrv_rmssd", "temperature_c", "bmi", "age", "family_history", "fasting"
    ]
    y = df["bgl_mg_dl"].values
    for col in features_to_check:
        x = df[col].values
        lr = LinearRegression().fit(x.reshape(-1, 1), y)
        r2_val = float(r2_score(y, lr.predict(x.reshape(-1, 1))))
        assert r2_val <= 0.50, f"Univariate R^2 of {col} exceeds 0.50!"
    print(">>> SAFEGUARD ASSERTION PASSED: All univariate R^2 <= 0.50.")


def main():
    print("Generating re-tuned synthetic physiological training dataset...")
    df = generate_synthetic_dataset(
        n_participants=150,
        readings_per_participant_range=(3, 5),
        random_seed=42
    )

    parquet_path = INTERIM_DIR / "synthetic_features.parquet"
    csv_path = INTERIM_DIR / "synthetic_features.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved synthetic dataset to: {parquet_path}")

    validate_and_report(df)


if __name__ == "__main__":
    main()
