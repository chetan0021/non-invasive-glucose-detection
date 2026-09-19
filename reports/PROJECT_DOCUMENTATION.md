# Non-Invasive Blood Glucose Prediction — Complete Project Documentation

**Last Updated**: 2026-09-13  
**Purpose**: Full technical reference for developers, researchers, and reviewers. Covers every code file, every dataset, every model, the end-to-end data flow, and the known limitations of the current system.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [End-to-End Data Flow Diagram](#2-end-to-end-data-flow-diagram)
3. [Repository Structure at a Glance](#3-repository-structure-at-a-glance)
4. [Data Layer — Every File Explained](#4-data-layer--every-file-explained)
5. [Scripts — Grouped by Role](#5-scripts--grouped-by-role)
6. [Models — Every Artifact Explained](#6-models--every-artifact-explained)
7. [Feature Vector — All 50 Inputs Explained](#7-feature-vector--all-50-inputs-explained)
8. [How a Prediction Is Made — Step by Step](#8-how-a-prediction-is-made--step-by-step)
9. [Training Pipeline Deep Dive](#9-training-pipeline-deep-dive)
10. [Model Architecture — 4 Base Models + Stack](#10-model-architecture--4-base-models--stack)
11. [Validation Strategy](#11-validation-strategy)
12. [Test Files and What They Test](#12-test-files-and-what-they-test)
13. [Reports and What They Contain](#13-reports-and-what-they-contain)
14. [Known Limitations](#14-known-limitations)

---

## 1. What This System Does

This project estimates **blood glucose level (BGL) in mg/dL** from non-invasive physiological sensors — no needle, no blood sample. The system reads:

- A **PPG (photoplethysmography) sensor** on the fingertip (e.g. MAX30102) — measures light absorption to derive pulse waveform, heart rate, and blood volume changes
- A **saliva pH sensor** — lower pH correlates with higher blood glucose
- A **skin temperature sensor** — glucose affects peripheral circulation
- **ECG-derived heart rate variability (HRV)** — autonomic nervous system response varies with glycemic state
- **Patient demographics** — age, BMI, diagnosis, medication

These inputs are fed into a trained machine learning ensemble that outputs:
- **Predicted blood glucose (mg/dL)**
- **Confidence interval** (5th–95th percentile quantile)
- **Clarke Error Grid zone** (A=clinically accurate through E=dangerous)
- **Out-of-distribution warning** if any input is outside the training range

There is also a **second, independent model** (Model B) that takes demographics only (no sensors) and classifies a patient's diabetes risk tier — useful for community screening without hardware.

> ⚠️ **Validation status**: All training and evaluation use synthetic data. No real paired PPG+glucose patient data has been collected. Results reflect synthetic self-consistency, not real-world clinical accuracy.

---

## 2. End-to-End Data Flow Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         RAW DATA SOURCES                                    ║
║                                                                              ║
║  data/raw/nhanes/         data/raw/d1namo/       data/raw/bidmc/            ║
║  CDC survey, 2537 adults  HRV+glucose dataset    Real PPG waveforms         ║
║  (demographics + labs)    (real patient data)    (fingertip distributions)  ║
║                                                                              ║
║  data/raw/diabetes130/    data/raw/capnobase/    data/raw/kaggle_ppg/       ║
║  101,766 hospital records PPG+capnography data  PPG+glucose kaggle data     ║
╚══════════════════════════════════════════════════╤═══════════════════════════╝
                                                   │
                              ┌────────────────────┘
                              ▼
╔══════════════════════════════════════════════════╗
║  STEP 1: synth_generator.py                      ║
║                                                   ║
║  Generates 595 synthetic patient readings         ║
║  across 150 participants. Uses real BIDMC PPG     ║
║  distributions as anchor. Produces 5 clinical     ║
║  archetypes including hypoglycemic variants.      ║
║                                                   ║
║  OUTPUT: data/interim/synthetic_features.parquet  ║
╚══════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════╗
║  STEP 2: ingest_real_data.py                     ║
║                                                   ║
║  Merges synthetic (595 rows) + NHANES (2537) +    ║
║  UCI Diabetes-130 (21,612) into one unified CSV.  ║
║  Applies 80/20 participant-level train/test split ║
║  (ZERO participant leakage across split).         ║
║                                                   ║
║  OUTPUT: data/processed/train.csv (19,818 rows)   ║
║          data/processed/test.csv  (4,926 rows)    ║
╚══════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════╗
║  STEP 3: features.py                             ║
║                                                   ║
║  Builds TWO separate feature pipelines:           ║
║                                                   ║
║  A) full_sensor (486 train / 109 test rows)       ║
║     Only rows that have PPG+HRV+pH sensor data    ║
║     50 features: 34 scaled numerics + 16 binary   ║
║     Scaler fitted on train only → scaler_full.pkl ║
║                                                   ║
║  B) tabular_only (19,332 train / 4,817 test)      ║
║     All rows — demographics and diagnoses only    ║
║     35 features for risk classification           ║
║                                                   ║
║  OUTPUT: full_sensor_train/test_features.csv      ║
║          tabular_train/test_features.csv          ║
║          scaler_full_sensor.pkl                   ║
╚══════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════╗
║  STEP 4: train_models.py                         ║
║                                                   ║
║  MODEL A (Full-Sensor Regression):                ║
║  4 base models trained with GroupKFold(5) CV     ║
║  on participant_id → no data leakage.             ║
║  Ridge + RF + XGBoost + SVR → Stacked Ensemble   ║
║                                                   ║
║  MODEL B (Tabular Risk Classifier):               ║
║  XGBoost 3-class classifier on NHANES only.       ║
║  Predicts healthy_risk/elevated_risk/diabetic     ║
║                                                   ║
║  OUTPUT: production_model_full_sensor.pkl         ║
║          production_model_full_sensor_stacked.pkl ║
║          production_model_tabular_riskclass.pkl   ║
║          quantile_regressor_full_sensor.pkl       ║
╚══════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════╗
║  STEP 5: predict.py (inference)                  ║
║                                                   ║
║  Takes raw sensor dict from caller.               ║
║  Derives missing features (VPG/APG/energy)        ║
║  from raw PPG using training formulas.            ║
║  Scales via scaler_full_sensor.pkl.               ║
║  Runs stacked ensemble → point prediction.        ║
║  Runs quantile models → confidence interval.      ║
║  Checks OOD bounds on 9 input features.           ║
║                                                   ║
║  OUTPUT: predicted BGL, CI, Clarke zone, OOD flag ║
╚══════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════╗
║  app/dashboard.py (Streamlit UI)                  ║
║                                                   ║
║  Interactive web dashboard. User enters sensor    ║
║  readings via sliders. Calls predict.py.          ║
║  Shows predicted BGL, confidence interval,        ║
║  Clarke zone indicator, risk band, charts.        ║
╚══════════════════════════════════════════════════╝
```

---

## 3. Repository Structure at a Glance

```
glucose-prediction/
│
├── scripts/                  All Python source code (canonical pipeline + utilities)
├── app/                      Streamlit dashboard
├── data/
│   ├── raw/                  Original downloaded datasets (never modified)
│   ├── interim/              Intermediate processed files (synthetic data)
│   └── processed/            Final train/test CSVs ready for modelling
├── models/                   Trained model .pkl files + metadata .json
├── reports/                  Evaluation reports, feature manifests, figures
├── notebooks/                Jupyter notebooks (exploratory, not canonical)
├── scratch/                  Throwaway experimental scripts (not production)
├── predict.py                Root-level re-export of scripts/predict.py
└── requirements.txt          Pinned Python dependencies
```

---

## 4. Data Layer — Every File Explained

### 4.1 Raw Data (`data/raw/`) — Never Modified

| Folder | Dataset | Source | Size | Role |
|---|---|---|---|---|
| `nhanes/` | CDC NHANES 2017–2018 | US CDC | ~11 XPT files | Demographics, glucose labs, lifestyle for Model B |
| `d1namo/` | D1NAMO HRV+Glucose | EPFL | varies | Real ECG-HRV + fingerstick glucose pairs |
| `bidmc/` | BIDMC PPG Dataset | PhysioNet | ~1 ZIP | Real PPG waveform distributions (fingertip anchor) |
| `capnobase/` | CapnoBase | Univ. of BC | varies | Additional PPG reference signals |
| `kaggle_ppg_glucose/` | Kaggle PPG+Glucose | Kaggle | varies | Additional PPG+glucose reference |
| `diabetes130/` | UCI Diabetes-130 | UCI ML Repo | 101k rows ZIP | Hospital glucose records for tabular pipeline |
| `pima/` | Pima Indian Diabetes | UCI ML Repo | small | Reference demographics dataset |

### 4.2 Interim Data (`data/interim/`)

| File | Created by | Contents |
|---|---|---|
| `synthetic_features.parquet` | `synth_generator.py` | 595 rows of synthetic multi-modal readings across 150 participants |
| `synthetic_features.csv` | `synth_generator.py` | Same as above, CSV format for inspection |
| `real_feature_pool.parquet` | `ingest_real_data.py` | Real patient HRV/PPG features from D1NAMO used as distribution anchor |
| `d1namo_hrv_glucose.parquet` | `ingest_real_data.py` | Cleaned D1NAMO readings before merging |

### 4.3 Processed Data (`data/processed/`) — Model-Ready

| File | Rows | Features | Created by | Used by |
|---|---|---|---|---|
| `glucose_dataset.csv` | 24,744 | raw | `ingest_real_data.py` | `features.py` |
| `train.csv` | 19,818 | raw | `ingest_real_data.py` | `features.py` |
| `test.csv` | 4,926 | raw | `ingest_real_data.py` | `features.py` |
| `full_sensor_train_features.csv` | 486 | 92 cols (50 model features) | `features.py` | `train_models.py` |
| `full_sensor_test_features.csv` | 109 | 92 cols | `features.py` | `train_models.py` |
| `tabular_train_features.csv` | 19,332 | 46 cols (35 model features) | `features.py` | `train_models.py` |
| `tabular_test_features.csv` | 4,817 | 46 cols | `features.py` | `train_models.py` |
| `features_manifest_full_sensor.json` | — | — | `features.py` | `predict.py`, `train_models.py` |
| `features_manifest_tabular.json` | — | — | `features.py` | `predict.py` |

> **Why so few full_sensor rows (486)?** The full-sensor pipeline requires all 5 sensor modalities. Only the 595 synthetic rows have complete PPG+HRV+pH+temperature readings. Real dataset rows (NHANES, diabetes-130) are demographics-only and go through the tabular pipeline.

---

## 5. Scripts — Grouped by Role

### 5.1 Canonical Pipeline (run in this order to regenerate everything)

These 4 scripts are the only ones that produce the production artifacts. All others are utilities, tests, or exploratory scripts.

```
synth_generator.py → ingest_real_data.py → features.py → train_models.py
```

| Script | What it does |
|---|---|
| **`synth_generator.py`** | Generates synthetic patient data. Reads real PPG distributions from `reports/ppg_feature_distributions.json` and `data/interim/real_feature_pool.parquet` as anchors. Produces 595 readings across 150 participants with 5 clinical archetypes (healthy, prediabetic, type2_controlled, type2_uncontrolled, type1_extreme). type1_extreme stratum uses 5 physiological sub-archetypes for hypoglycemia (sympathetic, parasympathetic, exercise, nocturnal, unawareness) and 3 for severe hyperglycemia. Validates that no single feature's univariate R² exceeds 0.50 (anti-leakage safeguard). Writes `data/interim/synthetic_features.parquet`. |
| **`ingest_real_data.py`** | Merges synthetic data with NHANES and UCI diabetes-130 records. Performs participant-level 80/20 train/test split with verified zero leakage. Assigns `training_branch` tag (`full_sensor` vs `tabular_only`) so downstream scripts can filter correctly. Writes `data/processed/train.csv` and `test.csv`. |
| **`features.py`** | Builds two model-ready feature matrices. For `full_sensor`: filters to sensor-equipped rows, derives `pulse_pressure` and `ph_deviation_from_mean`, applies StandardScaler fitted on train only, one-hot encodes diagnosis, bins BMI, adds medication/lifestyle binary flags. Writes 4 CSV files and `scaler_full_sensor.pkl`. |
| **`train_models.py`** | Trains all models. Model A: 4 base regressors with GroupKFold(n=5) cross-validation on `participant_id`, then a Ridge meta-learner stacking ensemble. Also runs hardware ablation study and quantile regression for uncertainty. Model B: XGBoost 3-class risk classifier on NHANES only (with explicit rejection of the pooled dataset due to AUROC=1.0 shortcut). Writes all `.pkl` model files and `model_metadata_full_sensor.json`. |

### 5.2 Inference (Production)

| Script | What it does |
|---|---|
| **`scripts/predict.py`** | Core inference engine. Class `GlucosePredictor` has two methods: `predict_full_sensor()` for sensor-based regression, `predict_risk_band()` for demographic screening. Handles raw input dicts, derives missing features (APG/VPG/signal energy from PPG inputs using training formulas), scales via saved scaler, runs stacked ensemble, computes quantile CI, checks OOD bounds, assigns Clarke zone. |
| **`predict.py`** (root) | One-liner re-export: `from scripts.predict import GlucosePredictor`. Allows `from predict import GlucosePredictor` from project root. |
| **`scripts/predict_safe.py`** | Earlier draft of predict.py with additional defensive checks. Superseded by current predict.py but kept for reference. |

### 5.3 Dashboard

| Script | What it does |
|---|---|
| **`app/dashboard.py`** | Streamlit web application. Provides clinical preset buttons (Healthy Adult, Prediabetes, Type 2, Severe Hyperglycemia, Hypoglycemia), manual sensor input sliders, and displays predicted BGL, confidence interval, Clarke zone colour indicator, risk band, HRV radar chart, PPG signal energy chart, and feature contribution chart. Calls `GlucosePredictor` from `predict.py`. |

### 5.4 Test / Verification Scripts

These scripts check the system at various levels. They do not modify any production artifacts.

| Script | What it tests |
|---|---|
| **`test_hypo_diversity.py`** | Runs 5 physiologically-diverse hypoglycemia cases (sympathetic, exercise, parasympathetic, nocturnal, unawareness) through `predict.py`. Documents Zone D/E results for 3 in-distribution cases (known limitation). |
| **`test_benchmark_presets.py`** | Runs the 5 canonical benchmark cases (Healthy, Prediabetes, Type 2, Severe Hyperglycemia, Hypoglycemia). Checks Zone A/B assignment. |
| **`verify_canonical_pipeline.py`** | Checks that the 4 canonical pipeline scripts exist and produce expected file artifacts. |
| **`verify_reverted_model_safety.py`** | Loads the production model and runs Clarke zone checks on all benchmarks. Used during safety rollback verification. |
| **`audit_clarke_grid_function.py`** | Independently verifies the Clarke Error Grid zone boundary implementation in `train_models.py` against the original published zone definitions. |
| **`audit_hardcoded_defaults.py`** | Scans `predict.py` for any hardcoded default values that could diverge from training pipeline. |
| **`test_rebalanced_model_safety.py`** | Tests the rebalanced model (post extreme-value rebalancing) against all benchmarks. |
| **`test_rebalanced_generator.py`** | Validates that the rebalanced synthetic generator produces the target 8% hypoglycemic / 10% severe hyperglycemic representation. |
| **`test_ci_clustering.py`** | Checks that confidence intervals from the quantile regressors are not clustering at a constant value. |
| **`test_ppg_fix.py`** | Verifies the PPG signal energy derivation fix is working correctly. |
| **`test_dashboard_flow.py`** | End-to-end smoke test of the Streamlit dashboard data flow. |
| **`test_visualizations.py`** | Tests that all Plotly chart generation functions in the dashboard produce valid figures. |
| **`simple_benchmark_test.py`** | Minimal benchmark test with no dependencies beyond predict.py. Quick sanity check. |
| **`final_benchmark_verification.py`** | Extended benchmark run producing a full comparison table. |
| **`final_verification.py`** | Final pipeline verification — checks model artifacts load, features match, predictions are not constant. |
| **`final_stratified_clarke_report.py`** | Generates stratified Clarke Grid breakdown (Zone A/B/C/D/E) for hypoglycemic, normal, elevated, and severe subgroups. |

### 5.5 Debugging / Investigation Scripts

Scripts written during active bug investigation. Not part of the canonical pipeline but preserved because they document findings.

| Script | What it investigated |
|---|---|
| **`debug_benchmark_predictions.py`** | Diagnosed why benchmark predictions were wrong (traced to APG hardcoded defaults). |
| **`debug_extreme_cases.py`** | Investigated Zone D failures on severe hyperglycemia and hypoglycemia. |
| **`debug_feature_assembly.py`** | Traced how predict.py assembles feature vectors vs how training pipeline assembled them. Found the PPG signal energy, APG, and VPG hardcoded default divergence. |
| **`debug_model_loading.py`** | Checked that pickle loading of the stacked ensemble dict is handled correctly. |
| **`debug_pipeline_discrepancy.py`** | Side-by-side comparison of training-time vs inference-time feature values. |
| **`investigate_feature_mismatch.py`** | Detailed investigation of which specific features diverged between train and inference. |
| **`compare_model_performance.py`** | Compares old vs new model after retraining to validate improvement. |
| **`quantile_diagnostic.py`** | Diagnosed constant-prediction bug in quantile regressors. |

### 5.6 Historical / Experimental Scripts (not canonical)

These exist from earlier iterations and are preserved but are not part of the current production flow.

| Script | What it was |
|---|---|
| `conformal_calibration.py` | Attempted conformal prediction calibration on top of quantile regressors. Results in `reports/conformal_calibration_final_report.md`. |
| `retrain_quantile_improved.py` | Experimental re-fitting of quantile models with better hyperparameters. |
| `rebalance_training_data.py` | Standalone rebalancing (later superseded by canonical synth_generator.py approach). |
| `create_extreme_rebalanced_training.py` | Earlier ad-hoc extreme value generation. Superseded. |
| `enhance_hypoglycemic_training.py` | Experimental additional hypoglycemic augmentation. |
| `train_rebalanced_model.py` | Standalone retraining on rebalanced data. Superseded by canonical train_models.py. |
| `apply_rebalanced_training.py`, `regenerate_full_pipeline.py`, `reapply_preprocessing_fixes.py` | Migration scripts from earlier pipeline versions. |
| `fix_all_hardcoded_defaults.py`, `fix_pipeline_bug.py`, `fix_ppg_signal_energy_bug.py`, `optimize_remaining_features.py` | Targeted fixes applied during the hardcoded-default bug investigation. |
| `direct_rebalanced_model_test.py`, `final_quantile_evaluation.py`, `final_coverage_test.py` | One-off evaluation runs during the rebalancing investigation. |
| `clarke_grid_safety_assessment.py`, `generate_comprehensive_report.py`, `generate_report_figures.py` | Report generation utilities. |

---

## 6. Models — Every Artifact Explained

| File | Type | Purpose |
|---|---|---|
| **`production_model_full_sensor.pkl`** | Python dict (stacked ensemble) | Production Model A. Contains `base_models` (dict of 4 fitted models), `meta_learner` (Ridge), `feature_list`, `model_names`. **This is what predict.py uses.** |
| **`production_model_full_sensor_stacked.pkl`** | Python dict | Identical to above (saved separately during training for reference). |
| **`scaler_full_sensor.pkl`** | sklearn StandardScaler | Fitted on train split. Transforms the 34 continuous numeric features. Mean and scale stored in `scaler.mean_` and `scaler.scale_`. |
| **`quantile_regressor_full_sensor.pkl`** | Python dict | Contains `q05_model`, `q50_model`, `q95_model` (GradientBoostingRegressor each) and `feature_list`. Produces the confidence interval. |
| **`production_model_tabular_riskclass.pkl`** | XGBClassifier | Model B. 3-class demographic risk classifier. Predicts `healthy_risk`, `elevated_risk`, `diabetic_risk`. |
| **`scaler_tabular.pkl`** | sklearn StandardScaler | Fitted on NHANES tabular train split. Scales age, BMI, waist circumference. |
| **`model_metadata_full_sensor.json`** | JSON | Records: training date, sample counts, feature list, CV metrics, test metrics, stratified performance by diagnosis, stacking weights. |
| **`model_metadata_tabular_riskclass.json`** | JSON | Records: NHANES cohort details, AUROC by class, confusion matrix, feature list. |
| **`quantile_regressor_conformal_calibrated.pkl`** | Python dict | Experimental conformal-calibrated version. Not in production path. |

---

## 7. Feature Vector — All 50 Inputs Explained

The model receives a 50-dimensional vector. Features are grouped into 5 modalities:

### Group 1 — PPG Optical Sensor (14 features, from MAX30102)

| Feature (scaled) | Raw meaning | Unit |
|---|---|---|
| `ppg_raw_dc_baseline` | Mean light absorption (DC component) — reflects blood volume | ADC counts (~155k–195k) |
| `ppg_raw_ac_p2p` | Peak-to-peak AC amplitude — reflects pulse strength | ADC counts (~1100–4850) |
| `ppg_systolic_peak` | Amplitude at systolic peak | ADC counts |
| `ppg_diastolic_peak` | Amplitude at diastolic peak | ADC counts |
| `ppg_trough` | Amplitude at waveform trough | ADC counts |
| `perfusion_index` | AC/DC × 100 — tissue blood perfusion percentage | % (0.5–5.0) |
| `ppg_signal_energy` | Signal energy = 0.5 × (AC/2)² — proxy for pulse power | ADC² units |
| `pulse_pressure` | systolic_peak − diastolic_peak | ADC counts |
| `hr_bpm` | Heart rate from PPG timing | BPM |
| `ppg_hr_bpm` | Same as hr_bpm (alias) | BPM |
| `pulse_width_ms` | Duration of each pulse | ms |
| `trough_to_trough_ms` | 60000/HR — inter-beat interval | ms |
| `dicrotic_notch_amp` | Amplitude at dicrotic notch | ADC counts |
| `dicrotic_ratio` | Notch amplitude / systolic amplitude | ratio |

### Group 2 — VPG / APG Waveform Derivatives (9 features)

These are mathematical derivatives of the PPG waveform encoding arterial stiffness and vascular compliance.

| Feature (scaled) | Meaning |
|---|---|
| `vpg_max` | Maximum of velocity plethysmogram (1st derivative of PPG) |
| `vpg_min` | Minimum of VPG |
| `apg_a` | a-wave of acceleration plethysmogram (2nd derivative of PPG) |
| `apg_b` | b-wave of APG |
| `apg_c` | c-wave of APG |
| `apg_d` | d-wave of APG |
| `apg_e` | e-wave of APG |
| `apg_b_a_ratio` | b/a ratio — arterial stiffness index (more negative = stiffer) |
| `apg_aging_index` | (b−c−d−e)/a — composite arterial aging index |

> **Note on inference**: When a caller does not provide APG/VPG values, `predict.py` derives them from `raw_ac` and `hr_bpm` using the same formulas used during training. VPG is computed as `raw_ac × 1.6988 × (hr/60)` and `−raw_ac × 1.3799 × (hr/60)`. APG a-wave defaults to the training mean (75.95) because it is statistically independent of raw_ac in the training data (r=0.009) — a known generator design gap documented in the limitations.

### Group 3 — ECG-HRV Autonomic Features (6 features)

| Feature (scaled) | Meaning | Clinical significance |
|---|---|---|
| `hrv_sdnn` | Standard deviation of NN intervals | Overall HRV; lower = higher sympathetic tone |
| `hrv_rmssd` | Root mean square of successive differences | Parasympathetic activity |
| `hrv_pnn50` | % of successive differences > 50ms | Parasympathetic activity |
| `hrv_lf` | Low-frequency power (0.04–0.15 Hz) | Sympathetic + parasympathetic |
| `hrv_hf` | High-frequency power (0.15–0.4 Hz) | Parasympathetic (vagal) activity |
| `hrv_lf_hf_ratio` | LF/HF ratio | Sympathovagal balance; higher = more sympathetic |

### Group 4 — Biochemical Sensors (3 features)

| Feature (scaled) | Meaning | Correlation to glucose |
|---|---|---|
| `saliva_ph` | Salivary pH (6.2–7.6 range) | Negative: higher glucose → lower pH (r²≈0.25–0.35) |
| `ph_deviation_from_mean` | saliva_ph − train_mean(7.26) | Centred version; captures deviation from population baseline |
| `temperature_c` | Skin surface temperature | Modest positive correlation via peripheral circulation |

> **Most important features**: Saliva pH (36% of model importance) and ph_deviation (17%) dominate. This is by design — the synthetic generator correlates pH with glucose via a validated formula (Ahadian et al. 2025).

### Group 5 — Demographics and Clinical History (16 binary features)

| Feature | Values | Meaning |
|---|---|---|
| `diag_none` | 0/1 | No diabetes diagnosis |
| `diag_prediabetes` | 0/1 | Diagnosed prediabetes |
| `diag_type_1` | 0/1 | Type 1 diabetes |
| `diag_type_2` | 0/1 | Type 2 diabetes |
| `bmi_cat_underweight` | 0/1 | BMI < 18.5 |
| `bmi_cat_normal` | 0/1 | 18.5 ≤ BMI < 25 |
| `bmi_cat_overweight` | 0/1 | 25 ≤ BMI < 30 |
| `bmi_cat_obese` | 0/1 | BMI ≥ 30 |
| `bmi_cat_missing` | 0/1 | BMI not recorded |
| `med_taking_insulin` | 0/1 | Currently on insulin |
| `med_taking_oral` | 0/1 | Taking oral glucose medication |
| `med_taking_any` | 0/1 | On any glucose medication |
| `gender_male` | 0/1 | Male = 1, Female = 0 |
| `family_history` | 0/1 | Family history of diabetes |
| `smoking` | 0/1 | Current smoker |
| `fasting` | 0/1 | Reading taken while fasting |

---

## 8. How a Prediction Is Made — Step by Step

```
Caller provides raw input dict:
{
  "hr_bpm": 84,  "hrv_sdnn": 26,  "hrv_rmssd": 18,  "hrv_pnn50": 6,
  "hrv_lf_hf_ratio": 2.2,  "ppg_raw_dc_baseline": 164000,
  "ppg_raw_ac_p2p": 1850,  "perfusion_index": 1.12,
  "saliva_ph": 6.60,  "temperature_c": 36.9,
  "age": 59,  "bmi": 29.8,  "diabetes_diagnosis": "Type 2",
  "fasting": 0, "med_taking_any": 1
}
```

**Step 1 — OOD check**  
Each input is compared against `TRAINING_FEATURE_BOUNDS` (1st–99th percentile ranges). If any value is outside, the response includes `is_out_of_distribution: True` and an `ood_warning` string listing which features are out of range. Prediction still proceeds.

**Step 2 — Derive missing features**  
If the caller did not provide APG, VPG, or `ppg_signal_energy`:
```python
# ppg_signal_energy — generator formula, noise-free
ppg_signal_energy = max(0.5 * (raw_ac / 2) ** 2, 10000)

# vpg_max / vpg_min — OLS slopes from training data
vpg_max = raw_ac * 1.6988 * (hr / 60)
vpg_min = -raw_ac * 1.3799 * (hr / 60)

# apg_a — training mean (independent of raw_ac in training, r=0.009)
apg_a = 75.9535
# apg_b/c/d/e — from apg_a and training-mean ratios
apg_b = apg_a * -0.9243       # training mean b/a ratio
apg_c = apg_a * 0.25
apg_d = apg_a * -0.25
apg_e = apg_a * 0.15
```

**Step 3 — Scale continuous features**  
Build a 34-column DataFrame with all numeric features, pass through `scaler_full_sensor.pkl` (StandardScaler). Each feature is transformed as `(value − mean) / std` using statistics fitted on the training set only.

**Step 4 — Encode categorical features**  
Diagnosis → one-hot 4 columns. BMI → 5 category columns. Medication flags, gender, fasting → binary. 16 binary features appended to form the complete 50-feature vector.

**Step 5 — Stacked ensemble prediction**

```
50-feature vector
       │
       ├──→ Ridge regressor     → pred_ridge
       ├──→ Random Forest       → pred_rf
       ├──→ XGBoost             → pred_xgb
       └──→ SVR                 → pred_svr
                │
                ▼
   [pred_ridge, pred_rf, pred_xgb, pred_svr]
                │
                ▼
      Ridge meta-learner (weights: RF=0.455, XGB=0.325, Ridge=0.215, SVR=0.120)
                │
                ▼
      point_prediction (clamped to [35, 500] mg/dL)
```

**Step 6 — Quantile confidence interval**  
Same 50-feature vector through `q05_model` and `q95_model` (GradientBoosting quantile regressors). Produces `[lower_5th, upper_95th]` mg/dL bounds. Monotonicity enforced: lower ≤ point ≤ upper.

**Step 7 — Clarke zone assignment**  
If the caller provides `reference_bgl_mg_dl` (ground truth), the Clarke zone is computed using the verified boundary definitions:

```
Zone A:  (ref ≤70 and pred ≤70) or |pred−ref| ≤ 20% of ref
Zone E:  (ref ≥180 and pred ≤70) or (ref ≤70 and pred ≥180)   ← most dangerous
Zone D:  (ref ≤70 and pred >70) or (ref ≥240 and 70≤pred≤180) ← fails to detect
Zone C:  overreaction / overcorrection zone
Zone B:  benign non-actionable error
```

**Step 8 — Return**
```python
{
  "predicted_bgl_mg_dl": 173.7,
  "confidence_interval_5th_95th": [142.1, 215.4],
  "interval_width_mg_dl": 73.3,
  "clarke_zone": "Zone A (Clinically Accurate)",
  "is_out_of_distribution": False,
  "ood_features": [],
  "ood_warning": None,
  "trend": "N/A — Baseline reading",
  "diagnosis_stratum_confidence": "...",
  "validation_status": "synthetic_self_consistency_only"
}
```

---

## 9. Training Pipeline Deep Dive

### 9.1 Synthetic Data Generation (`synth_generator.py`)

The generator creates one row per patient reading. For each participant it draws:

1. **Diabetes state** from `[healthy, prediabetic, type2_controlled, type2_uncontrolled, type1_extreme]` with probabilities `[0.28, 0.16, 0.20, 0.16, 0.20]`
2. **Demographics**: age, BMI, gender, smoking, family history — distributions vary by state
3. **BGL**: drawn from state-specific ranges; `type1_extreme` uses stratified sampling: 50% hypoglycemic (<70), 35% severe (>250), 15% normal
4. **Saliva pH**: `7.33 − 0.115 × glucose_z + noise` where `glucose_z = (bgl−120)/55`
5. **Temperature**: `36.64 + 0.05 × glucose_z + noise`, clipped to [36.2, 37.35]°C
6. **PPG raw DC**: uniform [155k, 195k] ADC counts
7. **PPG raw AC**: `2500 + 260 × glucose_z + 280 × (temp−36.6) + noise`, clipped [1100, 4850]
8. **Signal energy**: `0.5 × (AC/2)²`, min-clamped to 10,000
9. **HRV**: derived from participant baseline + glucose_z shift. Archetype branches override for extreme values (e.g. hypo_sympathetic forces HR 95–120, SDNN 15–30)
10. **APG a-wave**: `45.03 + 0.011984 × raw_ac + N(0, 12)` — correlates with pulse amplitude (r≈0.60), preserves N(75,15) marginal
11. **APG b, c, d, e**: derived from `apg_a` × ratios; b uses glucose-correlated `apg_ba` ratio
12. **VPG**: `raw_ac × slope_factor × (hr/60)` where `slope_factor ~ N(1.65, 0.22)`

**Safeguard**: After generation, the script asserts no feature has univariate R² > 0.50 against BGL — prevents accidentally introducing unrealistically strong correlations.

### 9.2 Train/Test Split Design

The split is at **participant level**, not row level. All readings from a single synthetic participant go to either train or test — never both. This prevents the model from learning participant-specific patterns and reporting inflated metrics.

```
120 participants → train  (486 rows)
 30 participants → test   (109 rows)
Zero overlap verified at every step.
```

GroupKFold(n=5) during cross-validation uses the same `participant_id` grouping, so each fold treats all readings from one participant as a unit.

### 9.3 Feature Scaling

StandardScaler is fitted **only on the training split** and then applied to both train and test. Test data never influences the scaler parameters. This is enforced in `features.py` and verified by the feature manifest `ph_mean_train_baseline` value (7.257).

---

## 10. Model Architecture — 4 Base Models + Stack

```
                    ┌─────────────────────────────────────────────────────┐
                    │              TRAINING PHASE                         │
                    │                                                     │
                    │  GroupKFold(5) on participant_id                    │
                    │                                                     │
                    │  Fold 1─5 → each base model trained on 4/5 folds   │
                    │           → Out-Of-Fold (OOF) predictions on 1/5   │
                    │                                                     │
                    │  OOF matrix [N×4] → Ridge meta-learner fitted      │
                    └─────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────────┐
                    │              INFERENCE PHASE                        │
                    │                                                     │
                    │  Input: 50-feature vector                           │
                    │         │                                           │
                    │    ┌────┴──────────────────────────────────┐       │
                    │    ▼           ▼           ▼           ▼   │       │
                    │  Ridge       Random     XGBoost       SVR  │       │
                    │  (weight     Forest     (weight       (wt  │       │
                    │   0.215)     (0.455)     0.325)      0.120)│       │
                    │    │           │           │           │   │       │
                    │    └────┬──────────────────────────────┘   │       │
                    │         ▼                                   │       │
                    │    Ridge meta-learner                       │       │
                    │    (weighted sum + −18.03 intercept)        │       │
                    │         │                                   │       │
                    │         ▼                                   │       │
                    │    Predicted BGL (mg/dL)                    │       │
                    └─────────────────────────────────────────────┘
```

**Base model configurations** (from `train_models.py` hyperparameter search):

| Model | Key hyperparameters | CV R² | Test R² |
|---|---|---|---|
| Ridge regression | α=100 | 0.659 | 0.780 |
| Random Forest | 150 trees, max_depth=12 | 0.870 | 0.758 |
| XGBoost | lr=0.1, max_depth=4, 150 trees | 0.865 | 0.792 |
| SVR | C=50, ε=1.0, scale kernel | 0.794 | 0.865 |
| **Stacked ensemble** | Meta-Ridge on OOF | **0.876** | **0.830** |

**Quantile regressors** (for confidence intervals):

| Model | Purpose | Parameters |
|---|---|---|
| `q05_model` | 5th percentile | GradientBoosting with `loss='quantile', alpha=0.05` |
| `q50_model` | Median (point estimate alternative) | `alpha=0.50` |
| `q95_model` | 95th percentile | `alpha=0.95` |
| **Empirical coverage** | 74.31% (target 90%) | Interval too narrow — known limitation |

**Model B — Tabular Risk Classifier**:

| Class | AUROC | Precision | Recall |
|---|---|---|---|
| healthy_risk | 0.813 | 0.902 | 0.620 |
| elevated_risk (prediabetes) | 0.730 | 0.292 | 0.467 |
| diabetic_risk | 0.885 | 0.452 | 0.833 |
| **Macro AUROC** | **0.809** | — | — |

---

## 11. Validation Strategy

### Train/Test Performance (Model A)

| Metric | Overall | None | Prediabetes | Type 1 | Type 2 |
|---|---|---|---|---|---|
| Test N | 109 | 31 | 17 | 18 | 43 |
| R² | 0.830 | 0.603 | 0.638 | 0.791 | 0.761 |
| MAE (mg/dL) | 21.56 | 10.92 | 11.03 | 53.61 | 19.98 |
| MARD (%) | 17.82 | 11.23 | 8.68 | 53.43 | 11.28 |
| Zone A (%) | 81.65 | 90.32 | 94.12 | 38.89 | 88.37 |
| Zone A+B (%) | 93.58 | 100.0 | 100.0 | 61.11 | 100.0 |

### OOD Detection Bounds

| Feature | Training p01 | Training p99 |
|---|---|---|
| Age | 18.0 | 70.0 |
| BMI | 19.4 | 40.6 |
| Saliva pH | 6.60 | 7.60 |
| Temperature (°C) | 36.2 | 37.2 |
| Heart Rate (BPM) | 52.0 | 122.0 |
| PPG DC Baseline | 155,000 | 195,000 |
| PPG AC Amplitude | 1,100 | 4,250 |
| Perfusion Index | 0.60 | 2.60 |
| Pulse Width (ms) | 145 | 350 |

---

## 12. Test Files and What They Test

Quick reference: which script tests which behaviour.

| Test Script | Tests |
|---|---|
| `test_benchmark_presets.py` | 5 canonical clinical cases → Zone A/B expected |
| `test_hypo_diversity.py` | 5 autonomic hypoglycemia archetypes → documents 3 Zone D failures (known limitation) |
| `verify_canonical_pipeline.py` | Pipeline file artifacts exist and have correct row counts |
| `verify_reverted_model_safety.py` | Production model loads and predicts non-constant, Zone A/B on benchmarks |
| `audit_clarke_grid_function.py` | Clarke zone boundaries match published definitions |
| `audit_hardcoded_defaults.py` | No APG/VPG/signal_energy defaults are static numbers |
| `test_rebalanced_generator.py` | Generator produces ≥8% hypoglycemic, ≥10% severe hyperglycemic rows |
| `test_ci_clustering.py` | Confidence intervals are not constant (past bug: all intervals were ≈ fixed width) |
| `final_stratified_clarke_report.py` | Full stratified Zone breakdown (A/B/C/D/E) by glucose subgroup |
| `simple_benchmark_test.py` | Smoke test: import predict.py, call predict(), get a number |

---

## 13. Reports and What They Contain

| File | Contents |
|---|---|
| `reports/model_comparison.md` | Main evaluation report. Executive summary, all model metrics, stratified diagnosis performance, ablation study, uncertainty quantification, known limitations (Section 8). |
| `reports/ablation_study.md` | Hardware BOM ablation — what happens to R² and MAE when each sensor modality is removed. Key finding: removing PPG morphology slightly improves RF performance due to collinearity. |
| `reports/features_manifest_full_sensor.json` | Authoritative list of all 50 model features in exact order, scaler baseline pH mean, train/test sample counts. Used by `predict.py` at runtime. |
| `reports/features_manifest_tabular.json` | Same for Model B tabular features. |
| `reports/feature_matrices_report.md` | Class balance audit across train/test splits by `diabetes_status`. |
| `reports/merge_log.md` | Record of ingest_real_data.py run: source row counts, deduplication, split sizes. |
| `reports/ppg_feature_distributions.json` | Real PPG feature statistics (mean, std, percentiles) from BIDMC dataset. Used by synth_generator.py as anchor. |
| `reports/conformal_calibration_final_report.md` | Results of experimental conformal calibration on quantile regressors. |
| `reports/quantile_regressor_diagnostic_report.md` | Analysis of the CI coverage gap (74% vs 90% target). |
| `reports/rebalancing_success_report.md` | Confirms the extreme-value rebalancing achieved target hypoglycemic and severe hyperglycemic representation. |
| `reports/root_cause_analysis_final.md` | Post-mortem of the APG/VPG/signal_energy hardcoded defaults bug in predict.py. |

---

## 14. Known Limitations

### L1 — Hypoglycemia Detection: 3 of 5 In-Distribution Archetypes Fail

3 of 5 physiologically-realistic hypoglycemic test cases (parasympathetic dominant, nocturnal, autonomic unawareness) produce Zone D predictions despite in-distribution inputs and a verified-correct inference pipeline. Root cause: the model assigns only 1.80% combined importance to all HRV features. The archetype-discriminating signal exists in scaled HRV (confirmed: HRV scaled spread 0.64–0.83 vs pH spread 0.57 within the hypoglycemic subset), but the 9% class size is insufficient to drive tree splits over the globally dominant pH signal. This is a sample-size / class-balance issue, not a scaling or pipeline bug. See `reports/model_comparison.md` Section 8.

### L2 — APG a-Wave Independence (Generator Design Gap, Partially Fixed)

The synthetic generator previously drew `apg_a` from `N(75, 15)` independently of `raw_ac`. Fixed 2026-09-13: `apg_a = 45.03 + 0.011984 × raw_ac + N(0, 12)` now achieves r≈0.60 with raw_ac while preserving the same marginal distribution. However, because `apg_a/b/c/d/e` features carry near-zero model importance (0.09–0.13% each), this fix did not change the 3 failing hypoglycemia archetypes, as predicted by the importance check.

### L3 — Confidence Interval Coverage Below Target

Empirical 90% CI coverage is 74.31% (target: 90%). Intervals are correctly ordered (lower ≤ point ≤ upper) but too narrow. The quantile regressors are trained on the same synthetic data and cannot capture real-world residual variance from unmeasured confounders.

### L4 — Synthetic Data Only

All training and evaluation uses synthetic patient data generated from real statistical distributions but not from real paired sensor+glucose measurements. Reported metrics (R²=0.83, Zone A+B=93.6%) reflect the model's ability to recover patterns in synthetic data, not its real-world clinical performance.

### L5 — Type 1 Subgroup Performance

Type 1 diabetes test performance is poor: MAE=53.6 mg/dL, Zone A=38.9%, Zone A+B=61.1%. This is because the Type 1 test set contains the hypoglycemic and severe hyperglycemic extreme cases (the most challenging range), while other diagnosis groups have narrower glucose ranges in the test set.
