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
| **Model A: Full-Sensor (Stacked Ensemble (Ridge Meta))** | **0.8301** | **0.9111** | **30.32** | **21.56** | **17.82%** | **81.65%** | **93.58%** |

---

## 2. Model A Multi-Model & Stacking Ensemble Comparison on Held-Out Test Data

### Full-Sensor Regression Models ($N_{\text{train}}=488$, $N_{\text{test}}=128$, $50$ Features)

| Algorithm | GroupKFold CV $R^2$ | Test $R^2$ | Test RMSE (mg/dL) | Test MAE (mg/dL) | Test MARD (%) | Clarke Zone A | Clarke Zone A+B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Ridge)** | 0.6590 | 0.7799 | 34.50 | 28.24 | 23.77% | 59.63% | 93.58% |
| **Random Forest** | 0.8702 | 0.7580 | 36.18 | 22.42 | 18.86% | 79.82% | 94.50% |
| **XGBoost** | 0.8652 | 0.7922 | 33.53 | 21.51 | 17.96% | 79.82% | 94.50% |
| **Support Vector Regressor (SVR)** | 0.7941 | 0.8653 | 26.99 | 21.01 | 16.14% | 78.90% | 93.58% |
| **Stacked Ensemble (Ridge Meta)** | 0.8761 | 0.8301 | 30.32 | 21.56 | 17.82% | 81.65% | 93.58% |

---

## 3. Stratified Evaluation by Clinical Diagnosis (`diabetes_diagnosis`)

| `diabetes_diagnosis` | Test $N$ | Confidence Status | $R^2$ | RMSE (mg/dL) | MAE (mg/dL) | MARD (%) | Clarke Zone A (%) | Clarke Zone A+B (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **None** | **31** | High Confidence ($N \ge 15$) | 0.6026 | 12.49 | 10.92 | 11.23% | 90.32% | 100.00% |
| **Prediabetes** | **17** | High Confidence ($N \ge 15$) | 0.6376 | 13.29 | 11.03 | 8.68% | 94.12% | 100.00% |
| **Type 1** | **18** | High Confidence ($N \ge 15$) | 0.7913 | 59.79 | 53.61 | 53.43% | 38.89% | 61.11% |
| **Type 2** | **43** | High Confidence ($N \ge 15$) | 0.7611 | 25.52 | 19.98 | 11.28% | 88.37% | 100.00% |

---

## 4. Model B: Production Tabular 3-Class Risk Classifier (NHANES Outpatient Cohort)

**Task**: 3-Class Demographic Pre-Diagnostic Screening (`healthy_risk`, `elevated_risk`, `diabetic_risk`)  
**Validated Population Scope**: CDC NHANES Community Outpatient Cohort ($N_{\text{train}}=2,029$, $N_{\text{test}}=508$)  
**Clean Label Formulation**: `elevated_risk` strictly for diagnosed Prediabetes; `healthy_risk` for diagnosis None (zero feature overlap).  
**Expanded Clinically-Grounded Risk Features (ADA / FINDRISC)**: 23 features including `age`, `bmi`, `waist_circumference_cm`, `gender_male`, `race_white`, `race_black`, `race_hispanic`, `race_asian`, `race_other`, `phys_act_active`, `phys_act_moderate`, `phys_act_sedentary`, `hypertension`, `high_cholesterol`, `gdm_positive`, `gdm_negative`, `gdm_male_na`, `family_history`, and `smoking`.  
**Production Macro AUROC**: **0.8092** (Prior 9-feature baseline: 0.7296)  

> [!IMPORTANT]
> **Honest Clinical Screening Framing (`elevated_risk`)**:
> `elevated_risk` AUROC reached **0.7302**, but precision remains low at **29.2%** (roughly 1 in 3 flagged cases is truly prediabetic), meaning this output should be communicated to users as 'worth a follow-up test' rather than a reliable standalone diagnosis. Pure demographic biometrics cannot substitute for biochemical HbA1c or fasting laboratory testing.

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
> - **Gestational Diabetes History (`gdm_*`)**: Prominent ADA screening indicator; women with a history of gestational diabetes exhibit a 7- to 10-fold higher lifetime risk of conversion to Type 2 diabetes. Men are assigned a distinct non-applicable category (`gdm_male_na`) rather than being incorrectly imputed.
> - **Race/Ethnicity (`race_*`)**: Explicitly included in the American Diabetes Association (ADA) Risk Test as a recognized, empirical epidemiological risk factor. Certain populations (Asian American, African American, Hispanic/Latino, Native American) experience significantly higher rates of insulin resistance and Type 2 diabetes at substantially lower BMI cutoffs (e.g., Asian BMI screening threshold is 23 kg/m² vs 25 kg/m² for general populations). This feature is utilized transparently as an evidence-based population risk modifier, not as an unexplained categorical confounder.

### Clinical Evaluation & Integrity Audits:
1. **Prediabetes (`elevated_risk`) Screening Finding & Honest Framing**: `elevated_risk` achieved **AUROC = 0.7302**, but precision remains low at **29.2%** (roughly 1 in 3 flagged cases is truly prediabetic), meaning this output should be communicated to users as 'worth a follow-up test' rather than a reliable standalone diagnosis. Distinguishing prediabetes from healthy adults using pure demographics yields low precision because prediabetic and normoglycemic individuals share heavily overlapping age/BMI distributions without biochemical fasting glucose or HbA1c testing.
2. **Rejection of Pooled 0.9825 Model**: The pooled model AUROC was rejected for production because demographic features (fasting survey indicator and inpatient missing BMI patterns) predict dataset origin (UCI 130 inpatient vs NHANES outpatient) with **AUROC = 1.0000**, creating an artificial shortcut between 100% diabetic inpatient charts and outpatient surveys.
3. **Production Recommendation**: The scoped NHANES model (**Macro AUROC = 0.8092**, Diabetic AUROC = **0.8847**, Healthy AUROC = **0.8127**) is established as the honest production baseline for outpatient screening.

---

## 5. Hardware BOM Ablation Study Summary

| Experiment | Features | Test $R^2$ | $\Delta R^2$ | Test MAE (mg/dL) | $\Delta$ MAE | Clarke Zone A (%) | Clarke Zone A+B (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Sensor Baseline (All Modalities)** | 50 | **0.8362** | +0.0782 | **20.35** | -2.07 | 76.15% | 92.66% |
| **Ablation 1: No PPG Morphology / Waveform Features** | 29 | **0.8163** | +0.0583 | **19.50** | -2.92 | 79.82% | 93.58% |
| **Ablation 2: No ECG-HRV Autonomic Features** | 44 | **0.8385** | +0.0805 | **20.20** | -2.22 | 76.15% | 92.66% |
| **Ablation 3: No Saliva pH Biochemical Sensor** | 48 | **0.7564** | -0.0016 | **22.88** | +0.46 | 75.23% | 92.66% |
| **Ablation 4: No Skin Temperature Sensor** | 49 | **0.8417** | +0.0837 | **20.23** | -2.19 | 74.31% | 92.66% |
| **PPG-Only Isolated Transducer Benchmark** | 21 | **0.6911** | -0.0669 | **30.97** | +8.55 | 63.30% | 89.91% |

> [!NOTE]
> **PPG Standalone Performance Framing**: Synthetic PPG features were deliberately designed with bounded individual correlations ($r=0.35-0.55$), resulting in isolated PPG $R^2=0.2322$. This reflects synthetic generator design choices rather than a definitive biological ceiling for real-world optical transducers.

---

## 6. Uncertainty Quantification & Interval Coverage

- **Method**: Quantile Gradient Boosting Regression at 5th, 50th, and 95th Percentiles  
- **Empirical 90% Confidence Interval Coverage on Holdout Test Set**: **74.31%** (Target: 90.0%)  
- **Mean Prediction Interval Width (MPIW)**: **113.89 mg/dL**  

---

## 7. Saved Production Artifacts & Metadata Guardrails

- **Full-Sensor Production Model**: `models/production_model_full_sensor.pkl` (Stacked Ensemble (Ridge Meta))
- **Full-Sensor Stacked Ensemble**: `models/production_model_full_sensor_stacked.pkl`
- **Quantile Regressor Bundle**: `models/quantile_regressor_full_sensor.pkl`
- **Tabular Risk Classifier**: `models/production_model_tabular_riskclass.pkl` (XGBoost Classifier)
- **Guardrail Metadata**: `models/model_metadata_full_sensor.json` (`validation_status = 'synthetic_self_consistency_only'`)

---

## 8. Known Limitations: Hypoglycemia Detection Across Autonomic Archetypes

> [!WARNING]
> **This section documents a confirmed, investigated, and unresolved model limitation.** It is included to ensure any downstream evaluation of this system is not based on overall Clarke zone statistics alone, which would give a misleading impression of safety for hypoglycemic patients.

### Finding

3 of 5 physiologically-realistic hypoglycemic test cases (reference BGL 55–65 mg/dL) are **not reliably detected** by the current full-sensor stacked ensemble, despite inputs that lie within the training distribution and a verified-correct inference pipeline. All three fail with **Zone D** predictions in the 124–138 mg/dL range — the model predicts elevated-normal rather than hypoglycemia.

| Archetype | Ref (mg/dL) | Pred (mg/dL) | Zone | OOD? |
| :--- | :---: | :---: | :---: | :---: |
| Parasympathetic dominant (missed meal) | 65 | 129.5 | **D** | no |
| Nocturnal (sleeping, bradycardia) | 55 | 124.0 | **D** | no |
| Autonomic unawareness | 59 | 137.5 | **D** | no |
| Sympathetic dominant (tachycardia) | 58 | 137.9 | D | YES — temp OOD |
| Exercise-induced (high perfusion) | 63 | 191.3 | E | YES — temp+AC OOD |

The two OOD-flagged cases (Diversity 1 and 2) are correctly identified by the OOD guardrail as extrapolations outside the training range. The three in-distribution Zone D failures are the genuine model limitation.

### Root Cause Analysis

**This is not a software or pipeline bug.** The inference pipeline, feature construction, scaler, and Clarke zone classifier were all independently verified correct. The limitation has two compounding causes:

**1. Feature importance collapse on HRV — confirmed to be a sample-size effect, not a scaling artefact.** Autonomic response is the primary differentiating signal across hypoglycemic presentations: tachycardia and low HRV in sympathetic cases, bradycardia and high HRV in nocturnal cases, and near-normal HRV in unawareness cases. The current model assigns **HRV features a combined importance of only 1.80%** (6 features, measured on tree-base models by Gini impurity). A post-hoc check confirmed this is not caused by StandardScaler compressing HRV's range: within the hypoglycemic subset (N=54), HRV scaled std is 0.64–0.83 and the mean `hrv_sdnn_scaled` spans **1.52 units** across HR-defined archetype bins (low-HR nocturnal: +0.99, high-HR sympathetic: −0.53). Saliva pH's scaled mean, by contrast, is **flat across the same archetype bins** (0.49 / 0.69 / 0.52) — pH provides no archetype-discriminating signal within the hypoglycemic subgroup at all. The low HRV importance is therefore a genuine consequence of class imbalance: pH carries globally-dominant glucose signal across the full training population (95%+ non-hypoglycemic rows), so tree splits on pH yield the highest information gain at a dataset level and HRV's locally-strong signal in the 9% hypoglycemic subgroup is not allocated sufficient splits. The right fix is not feature rescaling or importance reweighting — it is increasing hypoglycemic representation with real paired data so the subgroup is large enough to drive split allocation.

**2. Generator design gap in APG features (partially addressed).** `apg_a` was previously generated as `N(75, 15)` independently of `raw_ac`, introducing noise that carries no physiological information. This was fixed in `synth_generator.py` (2026-09-13): `apg_a` now correlates with pulse amplitude via `apg_a = 45.03 + 0.011984 × raw_ac + N(0, 12)`, achieving the same marginal distribution with `r(apg_a, raw_ac) ≈ 0.60`. The pipeline was regenerated after this fix. However, because `apg_a/b/c/d/e` carried near-zero importance in the model prior to the fix (0.09–0.13% each), correcting the generator did not materially change hypoglycemia detection — confirmed by re-running the same test cases against the retrained model with identical Zone D outcomes.

### Why the Model Cannot Be Easily Fixed Without Real Data

The reason HRV has near-zero importance is structural: in the synthetic training data, saliva pH and insulin medication are strong, clean correlates of glucose level by design, and the model correctly learns to use them. This is not wrong in general — these features are genuinely informative. The problem is that for hypoglycemic Type 1 insulin-dependent patients presenting with varied autonomic patterns, these "dominant" features are flat and non-discriminating. Upweighting HRV artificially in the training data or in the loss function might help on this specific subgroup while degrading the broader population. The correct fix requires **real paired sensor data from hypoglycemic patients** with documented autonomic profiles, not further synthetic generation. Synthetic data can model the archetype diversity (and now does), but the model cannot learn to trust HRV when the entire training signal is dominated by biochemical features that happen to be stronger correlates in the synthetic population.

### Clinical Safety Implication

The three in-distribution Zone D failures represent a **safety-critical gap**: a patient experiencing parasympathetic-dominant hypoglycemia, nocturnal hypoglycemia, or autonomic unawareness would receive a prediction of ~130 mg/dL rather than a hypoglycemic alert, potentially delaying appropriate treatment. This system should not be used as a primary hypoglycemia detection tool until this limitation is addressed with real-world validation data. The OOD warning system correctly flags inputs with extreme physiological values (temperatures outside 36.2–37.2 °C, AC amplitudes above 4250) but cannot flag in-distribution inputs that happen to fall in a region of the feature space where the model is unreliable.

### What Would Actually Fix This

Ranked by expected impact:

1. **Real paired sensor data from hypoglycemic patients.** Even a small cohort (N=20–30 participants with continuous glucose monitoring and simultaneous PPG/HRV recording during hypoglycemic episodes) would provide the ground truth needed to train and validate on this subpopulation.
2. **Glucose-HRV coupling in synthetic generation.** Currently, the archetype branches set HR/HRV ranges for extreme glucose states, but the broader training population's HRV is generated with only a weak, indirect coupling to glucose (via `glucose_z` terms). Strengthening this correlation in the synthetic data to closer reflect published HRV-glucose relationships might raise HRV importance enough to be useful.
3. **Hypoglycemia-specific alarm threshold tuning.** A separate binary classifier trained specifically on "is BGL < 70?" with an asymmetric loss function, using HRV and perfusion index as primary inputs, could complement the regression model rather than relying on the regression output alone to enter the hypoglycemic range.
