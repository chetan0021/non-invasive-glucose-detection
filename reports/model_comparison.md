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
| **Model A: Full-Sensor (Random Forest)** | **0.8528** | **0.9235** | **16.89** | **12.28** | **8.83%** | **93.50%** | **99.19%** |

---

## 2. Model A Multi-Model & Stacking Ensemble Comparison on Held-Out Test Data

### Full-Sensor Regression Models ($N_{\text{train}}=488$, $N_{\text{test}}=128$, $50$ Features)

| Algorithm | GroupKFold CV $R^2$ | Test $R^2$ | Test RMSE (mg/dL) | Test MAE (mg/dL) | Test MARD (%) | Clarke Zone A | Clarke Zone A+B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Ridge)** | 0.8175 | 0.7991 | 19.73 | 15.74 | 11.85% | 85.37% | 99.19% |
| **Random Forest** | 0.8205 | 0.8528 | 16.89 | 12.28 | 8.83% | 93.50% | 99.19% |
| **XGBoost** | 0.8480 | 0.8610 | 16.41 | 12.31 | 8.86% | 92.68% | 99.19% |
| **Support Vector Regressor (SVR)** | 0.7593 | 0.7196 | 23.31 | 18.08 | 13.30% | 74.80% | 99.19% |
| **Stacked Ensemble (Ridge Meta)** | 0.8599 | 0.8567 | 16.66 | 12.76 | 9.17% | 92.68% | 99.19% |

---

## 3. Stratified Evaluation by Clinical Diagnosis (`diabetes_diagnosis`)

| `diabetes_diagnosis` | Test $N$ | Confidence Status | $R^2$ | RMSE (mg/dL) | MAE (mg/dL) | MARD (%) | Clarke Zone A (%) | Clarke Zone A+B (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **None** | **30** | High Confidence ($N \ge 15$) | 0.6475 | 9.74 | 7.80 | 7.33% | 100.00% | 100.00% |
| **Prediabetes** | **27** | High Confidence ($N \ge 15$) | 0.8566 | 8.77 | 6.88 | 5.23% | 100.00% | 100.00% |
| **Type 1** | **24** | High Confidence ($N \ge 15$) | 0.7407 | 26.62 | 20.96 | 13.85% | 79.17% | 95.83% |
| **Type 2** | **42** | High Confidence ($N \ge 15$) | 0.7803 | 17.70 | 14.00 | 9.35% | 92.86% | 100.00% |

---

## 4. Model B: Production Tabular 3-Class Risk Classifier (NHANES Outpatient Cohort)

**Task**: 3-Class Demographic Pre-Diagnostic Screening (`healthy_risk`, `elevated_risk`, `diabetic_risk`)  
**Validated Population Scope**: CDC NHANES Community Outpatient Cohort ($N_{\text{train}}=2,029$, $N_{\text{test}}=508$)  
**Clean Label Formulation**: `elevated_risk` strictly for diagnosed Prediabetes; `healthy_risk` for diagnosis None (zero feature overlap).  
**Expanded Clinically-Grounded Risk Features (ADA / FINDRISC)**: 23 features including `age`, `bmi`, `waist_circumference_cm`, `gender_male`, `race_white`, `race_black`, `race_hispanic`, `race_asian`, `race_other`, `phys_act_active`, `phys_act_moderate`, `phys_act_sedentary`, `hypertension`, `high_cholesterol`, `gdm_positive`, `gdm_negative`, `gdm_male_na`, `family_history`, and `smoking`.  
**Production Macro AUROC**: **0.8092** (Prior 9-feature baseline: 0.7296)  

> [!IMPORTANT]
> **Honest Clinical Screening Framing (`elevated_risk`)**:  
> While `elevated_risk` (prediabetes) AUROC improved to **0.7302** (up from the 0.5369 baseline), precision is **29.17%** (with 28 of 60 true prediabetics captured, roughly 1 in 3.4 flagged cases is truly prediabetic), meaning this output must be communicated to users as **"worth a follow-up test"** rather than a reliable standalone diagnosis. Pure demographic and anthropometric biometrics cannot substitute for biochemical fasting plasma glucose or laboratory HbA1c testing.

| Risk Class | Test $N$ | AUROC (OvR) | Precision | Recall | F1-Score | Brier Calibration Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **healthy_risk** | 358.0 | **0.8127** | 0.9024 | 0.6201 | 0.7351 | 0.2385 |
| **elevated_risk** | 60.0 | **0.7302** | 0.2917 | 0.4667 | 0.3590 | 0.1284 |
| **diabetic_risk** | 90.0 | **0.8847** | 0.4518 | 0.8333 | 0.5859 | 0.1127 |

```
Confusion Matrix [Healthy, Elevated, Diabetic]:
[[222  62  74]
 [ 15  28  17]
 [  9   6  75]]
```

> [!NOTE]
> **Clinical Basis for Expanded Risk Factors (ADA Diabetes Risk Test & FINDRISC)**:
> - **Waist Circumference (`waist_circumference_cm_scaled`)**: Core FINDRISC metric reflecting central/visceral adiposity, which correlates more directly with hepatic insulin resistance and metabolic dysfunction than BMI alone.
> - **Physical Activity (`phys_act_*`)**: Direct ADA & FINDRISC factor; physical inactivity (<150 min/wk moderate-to-vigorous exercise) downregulates skeletal muscle GLUT4 glucose transporter expression and elevates T2D onset risk.
> - **Hypertension (`hypertension`)**: Established metabolic syndrome component; vascular stiffness and microvascular rarefaction exacerbate peripheral insulin resistance.
> - **High Cholesterol (`high_cholesterol`)**: Dyslipidemia (low HDL, high triglycerides) is pathobiologically linked to non-esterified fatty acid overload and beta-cell lipotoxicity.
> - **Gestational Diabetes History (`gdm_*`)**: Prominent ADA screening indicator; women with a history of gestational diabetes exhibit a 7- to 10-fold higher lifetime risk of conversion to Type 2 diabetes. Derived strictly from `RHQ162` ("During pregnancy, told you have diabetes") and `RHQ131` ("Ever been pregnant?"), with `DIQ160` strictly excluded to prevent label leakage. Men are assigned a distinct non-applicable category (`gdm_male_na`) rather than being incorrectly imputed.
> - **Race/Ethnicity (`race_*`)**: Explicitly included in the American Diabetes Association (ADA) Risk Test as a recognized, empirical epidemiological risk factor. Certain populations (Asian American, African American, Hispanic/Latino, Native American) experience significantly higher rates of insulin resistance and Type 2 diabetes at substantially lower BMI cutoffs (e.g., Asian BMI screening threshold is 23 kg/m² vs 25 kg/m² for general populations). This feature is utilized transparently as an evidence-based population risk modifier, not as an unexplained categorical confounder.

### Clinical Evaluation & Integrity Audits:
1. **Prediabetes (`elevated_risk`) Screening Finding & Honest Framing**: `elevated_risk` AUROC reached **0.7302**, but precision remains modest at **29.17%** (roughly 1 in 3.4 flagged cases is truly prediabetic), meaning this output should be communicated to users as **"worth a follow-up test"** rather than a reliable standalone diagnosis. Distinguishing prediabetes from healthy adults using pure demographics yields modest precision because prediabetic and normoglycemic individuals share heavily overlapping age/BMI distributions without biochemical fasting glucose or HbA1c testing.
2. **Rejection of Pooled 0.9825 Model**: The pooled model AUROC was rejected for production because demographic features (fasting survey indicator and inpatient missing BMI patterns) predict dataset origin (UCI 130 inpatient vs NHANES outpatient) with **AUROC = 1.0000**, creating an artificial shortcut between 100% diabetic inpatient charts and outpatient surveys.
3. **Production Recommendation**: The scoped NHANES model (**Macro AUROC = 0.8092**, Diabetic AUROC = **0.8847**, Healthy AUROC = **0.8127**) is established as the honest production baseline for outpatient screening.

---

## 5. Hardware BOM Ablation Study Summary

| Experiment | Features | Test $R^2$ | $\Delta R^2$ | Test MAE (mg/dL) | $\Delta$ MAE | Clarke Zone A (%) | Clarke Zone A+B (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Sensor Baseline (All Modalities)** | 50 | **0.8541** | +0.0013 | **12.27** | -0.01 | 93.50% | 99.19% |
| **Ablation 1: No PPG Morphology / Waveform Features** | 29 | **0.8658** | +0.0130 | **12.06** | -0.22 | 92.68% | 99.19% |
| **Ablation 2: No ECG-HRV Autonomic Features** | 44 | **0.8390** | -0.0138 | **12.49** | +0.21 | 93.50% | 99.19% |
| **Ablation 3: No Saliva pH Biochemical Sensor** | 48 | **0.8565** | +0.0037 | **12.22** | -0.06 | 95.12% | 99.19% |
| **Ablation 4: No Skin Temperature Sensor** | 49 | **0.8544** | +0.0016 | **12.20** | -0.08 | 93.50% | 99.19% |
| **PPG-Only Isolated Transducer Benchmark** | 21 | **0.2427** | -0.6101 | **29.78** | +17.50 | 52.85% | 96.75% |

> [!NOTE]
> **PPG Standalone Performance Framing**: Synthetic PPG features were deliberately designed with bounded individual correlations ($r=0.35-0.55$), resulting in isolated PPG $R^2=0.2322$. This reflects synthetic generator design choices rather than a definitive biological ceiling for real-world optical transducers.

---

## 6. Uncertainty Quantification & Interval Coverage

- **Method**: Quantile Gradient Boosting Regression at 5th, 50th, and 95th Percentiles  
- **Empirical 90% Confidence Interval Coverage on Holdout Test Set**: **87.8%** (Target: 90.0%)  
- **Mean Prediction Interval Width (MPIW)**: **99.78 mg/dL**  

---

## 7. Saved Production Artifacts & Metadata Guardrails

- **Full-Sensor Production Model**: `models/production_model_full_sensor.pkl` (Random Forest)
- **Full-Sensor Stacked Ensemble**: `models/production_model_full_sensor_stacked.pkl`
- **Quantile Regressor Bundle**: `models/quantile_regressor_full_sensor.pkl`
- **Tabular Risk Classifier**: `models/production_model_tabular_riskclass.pkl` (XGBoost Classifier)
- **Guardrail Metadata**: `models/model_metadata_full_sensor.json` (`validation_status = 'synthetic_self_consistency_only'`)
