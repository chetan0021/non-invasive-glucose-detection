# Non-Invasive Blood Glucose Prediction: Model Benchmark & Comparison Report

**Generated On**: 2026-09-10  
**Target Metric**: Blood Glucose Level (`bgl_mg_dl` in mg/dL) & Clinical Risk Tier  
**Evaluation Design**: Participant-level 80/20 train/test holdout evaluation with zero participant leakage.

---

## 1. Executive Summary: Production Models vs. Published Literature & User Targets

| Architecture / Benchmark | $R^2$ | Pearson $R$ | RMSE (mg/dL) | MAE (mg/dL) | MARD (%) | Clarke Zone A (%) | Clarke Zone A+B (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **User Target Thresholds** | — | — | **< 20.0** | **< 15.0** | **< 10.0%** | **> 85.0%** | **100.0%** |
| *Frontiers Digital Health 2026* | 0.9200 | 0.9592 | — | 4.80 | — | — | — |
| *Measurement 2025* | 0.8649 | 0.9300 | — | — | 5.15% | — | — |
| *Algorithms 2025* | — | — | 15.36 | 13.17 | — | 94.74% | — |
| *Informatics in Med. Unlocked 2024* | — | — | 43.28 | — | — | — | 100.0% |
| **Model A: Full-Sensor (Random Forest)** | **0.8557** | **0.9250** | **16.60** | **12.13** | **8.78%** | **93.75%** | **99.22%** |

---

## 2. Model A Multi-Model & Stacking Ensemble Comparison on Held-Out Test Data

### Full-Sensor Regression Models ($N_{\text{train}}=488$, $N_{\text{test}}=128$, $50$ Features)

| Algorithm | GroupKFold CV $R^2$ | Test $R^2$ | Test RMSE (mg/dL) | Test MAE (mg/dL) | Test MARD (%) | Clarke Zone A | Clarke Zone A+B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Ridge)** | 0.8087 | 0.8031 | 19.39 | 15.38 | 11.65% | 85.94% | 99.22% |
| **Random Forest** | 0.8182 | 0.8557 | 16.60 | 12.13 | 8.78% | 93.75% | 99.22% |
| **XGBoost** | 0.8480 | 0.8686 | 15.84 | 11.95 | 8.71% | 95.31% | 99.22% |
| **Support Vector Regressor (SVR)** | 0.7596 | 0.7257 | 22.88 | 17.63 | 13.02% | 76.56% | 99.22% |
| **Stacked Ensemble (Ridge Meta)** | 0.8617 | 0.8621 | 16.22 | 12.41 | 9.11% | 92.19% | 99.22% |

---

## 3. Stratified Evaluation by Clinical Diagnosis (`diabetes_diagnosis`)

| `diabetes_diagnosis` | Test $N$ | Confidence Status | $R^2$ | RMSE (mg/dL) | MAE (mg/dL) | MARD (%) | Clarke Zone A (%) | Clarke Zone A+B (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **None** | **35** | High Confidence ($N \ge 15$) | 0.6123 | 10.01 | 7.99 | 7.43% | 100.00% | 100.00% |
| **Prediabetes** | **27** | High Confidence ($N \ge 15$) | 0.8436 | 9.16 | 7.39 | 5.59% | 100.00% | 100.00% |
| **Type 1** | **24** | High Confidence ($N \ge 15$) | 0.7391 | 26.71 | 20.87 | 13.65% | 79.17% | 95.83% |
| **Type 2** | **42** | High Confidence ($N \ge 15$) | 0.7934 | 17.16 | 13.64 | 9.16% | 92.86% | 100.00% |

---

## 4. Model B: Production Tabular 3-Class Risk Classifier (NHANES Outpatient Cohort)

**Task**: 3-Class Demographic Pre-Diagnostic Screening (`healthy_risk`, `elevated_risk`, `diabetic_risk`)  
**Validated Population Scope**: CDC NHANES Community Outpatient Cohort ($N_{\text{train}}=2,029$, $N_{\text{test}}=508$)  
**Clean Label Formulation**: `elevated_risk` strictly for diagnosed Prediabetes; `healthy_risk` for diagnosis None (zero feature overlap).  
**Independent Features**: `age_scaled`, `bmi_scaled`, `gender_male`, `bmi_cat_*`, `family_history`, `smoking` (Strictly excludes diagnosis, medications, glucose targets, and dataset shortcut features).  
**Production Macro AUROC**: **0.7296**  

| Risk Class | Test $N$ | AUROC (OvR) | Precision | Recall | F1-Score | Brier Calibration Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **healthy_risk** | 403.0 | **0.7904** | 0.9397 | 0.5409 | 0.6866 | 0.2826 |
| **elevated_risk** | 11.0 | **0.5733** | 0.0241 | 0.1818 | 0.0426 | 0.099 |
| **diabetic_risk** | 94.0 | **0.8253** | 0.3990 | 0.8191 | 0.5366 | 0.1351 |

```
Confusion Matrix [Healthy, Elevated, Diabetic]:
[[218  73 112]
 [  5   2   4]
 [  9   8  77]]
```

### Clinical Evaluation & Integrity Audits:
1. **Prediabetes (`elevated_risk`) Screening Finding**: With class weighting, the model achieves **AUROC = 0.5733** and **Recall = 18.2%** with **Precision = 2.41%**. Distinguishing prediabetes from healthy adults using pure demographics yields low precision because prediabetic and normoglycemic individuals share heavily overlapping age/BMI distributions without biochemical fasting glucose or HbA1c testing.
2. **Rejection of Pooled 0.9825 Model**: The pooled model AUROC was rejected for production because demographic features (fasting survey indicator and inpatient missing BMI patterns) predict dataset origin (UCI 130 inpatient vs NHANES outpatient) with **AUROC = 1.0000**, creating an artificial shortcut between 100% diabetic inpatient charts and outpatient surveys.
3. **Production Recommendation**: The scoped NHANES model (**Macro AUROC = 0.7296**, Diabetic AUROC = **0.8253**, Healthy AUROC = **0.7904**) is established as the honest production baseline for outpatient screening.

---

## 5. Hardware BOM Ablation Study Summary

| Experiment | Features | Test $R^2$ | $\Delta R^2$ | Test MAE (mg/dL) | $\Delta$ MAE | Clarke Zone A (%) | Clarke Zone A+B (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Sensor Baseline (All Modalities)** | 50 | **0.8536** | -0.0021 | **12.24** | +0.11 | 96.09% | 99.22% |
| **Ablation 1: No PPG Morphology / Waveform Features** | 29 | **0.8763** | +0.0206 | **11.73** | -0.40 | 93.75% | 99.22% |
| **Ablation 2: No ECG-HRV Autonomic Features** | 44 | **0.8375** | -0.0182 | **12.62** | +0.49 | 94.53% | 99.22% |
| **Ablation 3: No Saliva pH Biochemical Sensor** | 48 | **0.8476** | -0.0081 | **12.33** | +0.20 | 94.53% | 99.22% |
| **Ablation 4: No Skin Temperature Sensor** | 49 | **0.8484** | -0.0073 | **12.27** | +0.14 | 94.53% | 99.22% |
| **PPG-Only Isolated Transducer Benchmark** | 21 | **0.2348** | -0.6209 | **29.58** | +17.45 | 52.34% | 96.88% |

> [!NOTE]
> **PPG Standalone Performance Framing**: Synthetic PPG features were deliberately designed with bounded individual correlations ($r=0.35-0.55$), resulting in isolated PPG $R^2=0.2322$. This reflects synthetic generator design choices rather than a definitive biological ceiling for real-world optical transducers.

---

## 6. Uncertainty Quantification & Interval Coverage

- **Method**: Quantile Gradient Boosting Regression at 5th, 50th, and 95th Percentiles  
- **Empirical 90% Confidence Interval Coverage on Holdout Test Set**: **85.16%** (Target: 90.0%)  
- **Mean Prediction Interval Width (MPIW)**: **96.56 mg/dL**  

---

## 7. Saved Production Artifacts & Metadata Guardrails

- **Full-Sensor Production Model**: `models/production_model_full_sensor.pkl` (Random Forest)
- **Full-Sensor Stacked Ensemble**: `models/production_model_full_sensor_stacked.pkl`
- **Quantile Regressor Bundle**: `models/quantile_regressor_full_sensor.pkl`
- **Tabular Risk Classifier**: `models/production_model_tabular_riskclass.pkl` (XGBoost Classifier)
- **Guardrail Metadata**: `models/model_metadata_full_sensor.json` (`validation_status = 'synthetic_self_consistency_only'`)
