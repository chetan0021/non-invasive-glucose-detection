# Dataset Merge & Ingestion Log (Refined & Non-Conflated Taxonomy)

**Generated On**: 2026-09-10  
**Output File**: `glucose-prediction/data/processed/glucose_dataset.csv`  
**Train / Test Splits**: `train.csv` (80%), `test.csv` (20%) — strictly partitioned by `participant_id`.

---

## 1. Summary of Ingested Sources & Adult Filtering

| Source Name | Raw Ingested | Adult Usable (Age >= 18 with Glucose) | Training Branch | PPG / Bio Signal Coverage |
| :--- | :---: | :---: | :--- | :---: |
| **CDC NHANES 2017-2018** | **3,036** | **2,507** (384 pediatric age < 18 filtered) | `tabular_only` | Tabular Only (`has_ppg=False`) |
| **UCI Diabetes 130** | **101,766** | **21,612** (454 pediatric age < 20 filtered) | `tabular_only` | Tabular Only (`has_ppg=False`) |
| **Synthetic Multi-Modal** | **609** | **609** (Adult 18–78 cohort) | `full_sensor` | Full Multi-Modal (`has_ppg=True`, `has_ph=True`) |
| **TOTAL UNIFIED** | **105,411** | **24,765** (Total 1,236 pediatric rows removed) | *(Two-Branch Schema)* | — |

---

## 2. Two-Branch Architecture Breakdown

- **`full_sensor` Branch** (Rows with `has_ppg=True` AND `has_ph=True`): **616** rows (2.5% of dataset).
- **`tabular_only` Branch** (Rows with `has_ppg=False`): **24,149** rows (97.5% of dataset).

---

## 3. Corrected Non-Conflated Clinical Taxonomy

### Cross-Tab: `diabetes_diagnosis` $\times$ `glycemic_state_at_reading`
| diabetes_diagnosis   |   elevated |   high |   normal |   very_high |   All |
|:---------------------|-----------:|-------:|---------:|------------:|------:|
| None                 |       1044 |      4 |      910 |           5 |  1963 |
| Prediabetes          |        374 |      8 |       71 |           3 |   456 |
| Type 1               |        280 |    748 |      273 |         588 |  1889 |
| Type 2               |       6059 |   5815 |     5084 |        3499 | 20457 |
| All                  |       7757 |   6575 |     6338 |        4095 | 24765 |

### Value Counts of Unified `diabetes_status`:
| diabetes_status           |   count |
|:--------------------------|--------:|
| type2_controlled          |    6059 |
| type2_uncontrolled        |    5815 |
| type2_normoglycemic       |    5084 |
| type2_severe              |    3499 |
| type1_uncontrolled        |    1336 |
| undiagnosed_elevated      |    1044 |
| healthy                   |     910 |
| prediabetes_elevated      |     374 |
| type1_elevated            |     280 |
| type1_normoglycemic       |     273 |
| prediabetes_normoglycemic |      71 |
| prediabetes_high          |      11 |
| undiagnosed_high          |       9 |

> [!NOTE]
> **Taxonomy Audit**:
> 1. `diabetes_diagnosis` is invariant and reflects known medical diagnosis.
> 2. `glycemic_state_at_reading` reflects the instantaneous measurement (<100: normal, 100-179: elevated, 180-249: high, >=250: very_high).
> 3. Diagnosed Type 2 patients with a normal glucose reading are accurately classified as `type2_normoglycemic` (5,084 rows) rather than being conflated as `healthy`.
> 4. The label `healthy` is strictly reserved for `diabetes_diagnosis == "None"` AND `glycemic_state_at_reading == "normal"` (910 rows).

---

## 4. Real Tabular vs. HbA1c Continuous Re-Derivation Audit

- **True Measured Fasting/Serum Glucose (`real_tabular`)**: **7,868** rows.
- **HbA1c-Derived Continuous Glucose (`real_tabular_hba1c_derived`)**: **16,281** rows (67.4% of real tabular).
- **Continuous Distribution**: Values sampled smoothly within physiological clinical bounds.


> [!WARNING]
> **HbA1c-Derived Proportion Alert**: HbA1c-derived rows account for **67.4%** (16,281 rows) of the total real tabular data (24,149 rows).
> **Sampling Update**: Values are now smoothly sampled from within-bucket physiological distributions rather than repeated constants.
> The column `bgl_is_hba1c_derived=True` is explicitly preserved so models can filter or weight them.


---

## 5. Participant-Level Train / Test Split (80 / 20)

- **Partitioning Method**: Stratified by `training_branch` and `diabetes_diagnosis` at the unique `participant_id` level.
- **Data Leakage Guarantee**: **0 overlapping `participant_id` values** between train and test splits.
- **Train Split (`train.csv`)**: **19,825** rows (16,792 unique participants).
- **Test Split (`test.csv`)**: **4,940** rows (4,198 unique participants).

---

## 6. Column-Level Missingness Audit

| Canonical Column | NHANES Missing | D130 Missing | Synthetic Missing | Full Dataset Missing |
| :--- | :---: | :---: | :---: | :---: |
| `reading_id` | 0.0% | 0.0% | 0.0% | 0.0% |
| `participant_id` | 0.0% | 0.0% | 0.0% | 0.0% |
| `data_source` | 0.0% | 0.0% | 0.0% | 0.0% |
| `training_branch` | 0.0% | 0.0% | 0.0% | 0.0% |
| `placement` | 0.0% | 0.0% | 0.0% | 0.0% |
| `has_ppg` | 0.0% | 0.0% | 0.0% | 0.0% |
| `has_ph` | 0.0% | 0.0% | 0.0% | 0.0% |
| `has_temp` | 0.0% | 0.0% | 0.0% | 0.0% |
| `bgl_is_hba1c_derived` | 0.0% | 0.0% | 0.0% | 0.0% |
| `usable_for_training` | 0.0% | 0.0% | 0.0% | 0.0% |
| `diabetes_diagnosis` | 0.0% | 0.0% | 0.0% | 0.0% |
| `glycemic_state_at_reading` | 0.0% | 0.0% | 0.0% | 0.0% |
| `diabetes_status` | 0.0% | 0.0% | 0.0% | 0.0% |
| `diabetes_type` | 0.0% | 0.0% | 0.0% | 0.0% |
| `context` | 0.0% | 0.0% | 0.0% | 0.0% |
| `fasting` | 0.0% | 100.0% | 0.0% | 87.3% |
| `bgl_mg_dl` | 4.3% | 78.6% | 0.0% | 0.0% |
| `saliva_ph` | 100.0% | 100.0% | 0.0% | 97.5% |
| `temperature_c` | 100.0% | 100.0% | 0.0% | 97.5% |
| `age` | 0.0% | 0.0% | 0.0% | 0.0% |
| `gender` | 0.0% | 0.0% | 0.0% | 0.0% |
| `bmi` | 1.9% | 100.0% | 0.0% | 87.5% |
| `family_history` | 0.0% | 100.0% | 0.0% | 87.3% |
| `medication` | 0.0% | 0.0% | 0.0% | 0.0% |
| `smoking` | 0.0% | 100.0% | 0.0% | 87.3% |
| `race_ethnicity` | 0.0% | 0.0% | 100.0% | 2.5% |
| `waist_circumference_cm` | 6.0% | 100.0% | 100.0% | 90.3% |
| `physical_activity_level` | 0.0% | 0.0% | 100.0% | 2.5% |
| `hypertension` | 0.2% | 100.0% | 100.0% | 89.8% |
| `high_cholesterol` | 0.9% | 100.0% | 100.0% | 89.8% |
| `hdl_cholesterol_mg_dl` | 5.8% | 100.0% | 100.0% | 89.9% |
| `gestational_diabetes` | 0.0% | 53.7% | 100.0% | 47.9% |
| `ppg_raw_dc_baseline` | 100.0% | 100.0% | 0.0% | 97.5% |
| `ppg_raw_ac_p2p` | 100.0% | 100.0% | 0.0% | 97.5% |
| `ppg_systolic_peak` | 100.0% | 100.0% | 0.0% | 97.5% |
| `ppg_diastolic_peak` | 100.0% | 100.0% | 0.0% | 97.5% |
| `ppg_trough` | 100.0% | 100.0% | 0.0% | 97.5% |
| `perfusion_index` | 100.0% | 100.0% | 0.0% | 97.5% |
| `ppg_signal_energy` | 100.0% | 100.0% | 0.0% | 97.5% |
| `hr_bpm` | 100.0% | 100.0% | 0.0% | 97.5% |
| `ppg_hr_bpm` | 100.0% | 100.0% | 0.0% | 97.5% |
| `pulse_width_ms` | 100.0% | 100.0% | 0.0% | 97.5% |
| `trough_to_trough_ms` | 100.0% | 100.0% | 0.0% | 97.5% |
| `dicrotic_notch_amp` | 100.0% | 100.0% | 0.0% | 97.5% |
| `dicrotic_ratio` | 100.0% | 100.0% | 0.0% | 97.5% |
| `vpg_max` | 100.0% | 100.0% | 0.0% | 97.5% |
| `vpg_min` | 100.0% | 100.0% | 0.0% | 97.5% |
| `apg_a` | 100.0% | 100.0% | 0.0% | 97.5% |
| `apg_b` | 100.0% | 100.0% | 0.0% | 97.5% |
| `apg_c` | 100.0% | 100.0% | 0.0% | 97.5% |
| `apg_d` | 100.0% | 100.0% | 0.0% | 97.5% |
| `apg_e` | 100.0% | 100.0% | 0.0% | 97.5% |
| `apg_b_a_ratio` | 100.0% | 100.0% | 0.0% | 97.5% |
| `apg_aging_index` | 100.0% | 100.0% | 0.0% | 97.5% |
| `hrv_sdnn` | 100.0% | 100.0% | 0.0% | 97.5% |
| `hrv_rmssd` | 100.0% | 100.0% | 0.0% | 97.5% |
| `hrv_pnn50` | 100.0% | 100.0% | 0.0% | 97.5% |
| `hrv_lf` | 100.0% | 100.0% | 0.0% | 97.5% |
| `hrv_hf` | 100.0% | 100.0% | 0.0% | 97.5% |
| `hrv_lf_hf_ratio` | 100.0% | 100.0% | 0.0% | 97.5% |

---

## 7. Sample Rows per Data Source

### A. `real_tabular` (CDC NHANES & D130 direct glucose)
```
  participant_id training_branch  bgl_mg_dl   age gender   bmi diabetes_diagnosis glycemic_state_at_reading       diabetes_status
0   nhanes_93708    tabular_only      122.0  66.0      F  23.7        Prediabetes                  elevated  prediabetes_elevated
1   nhanes_93711    tabular_only      107.0  56.0      M  21.3               None                  elevated  undiagnosed_elevated
2   nhanes_93717    tabular_only       91.0  22.0      M  24.5               None                    normal               healthy
3   nhanes_93718    tabular_only       89.0  45.0      M  22.0               None                    normal               healthy
4   nhanes_93721    tabular_only      104.0  60.0      F  35.9               None                  elevated  undiagnosed_elevated
```

### B. `real_tabular_hba1c_derived` (UCI Diabetes 130 smoothly derived)
```
     participant_id training_branch  bgl_mg_dl  bgl_is_hba1c_derived   age gender diabetes_diagnosis glycemic_state_at_reading      diabetes_status
2537  d130_40523301    tabular_only      163.9                  True  85.0      M             Type 2                  elevated     type2_controlled
2538  d130_93196251    tabular_only      178.8                  True  75.0      F             Type 2                  elevated     type2_controlled
2539  d130_84488562    tabular_only      250.1                  True  55.0      F             Type 1                 very_high   type1_uncontrolled
2540  d130_67897251    tabular_only      101.3                  True  65.0      M             Type 2                  elevated     type2_controlled
2541  d130_96440301    tabular_only       87.4                  True  85.0      F             Type 2                    normal  type2_normoglycemic
```

### C. `synthetic` (Multi-Modal Full Sensor Branch)
```
      participant_id training_branch  bgl_mg_dl  saliva_ph  perfusion_index  pulse_width_ms  hrv_sdnn diabetes_diagnosis       diabetes_status
24149      SYNTH_001     full_sensor      145.1       7.26            1.291           176.0      6.00        Prediabetes  prediabetes_elevated
24150      SYNTH_001     full_sensor      106.8       7.10            0.619           157.1      6.00        Prediabetes  prediabetes_elevated
24151      SYNTH_001     full_sensor      131.7       7.15            1.707           173.1     15.09        Prediabetes  prediabetes_elevated
24152      SYNTH_002     full_sensor      124.6       7.19            2.016           207.7    121.93             Type 1        type1_elevated
24153      SYNTH_002     full_sensor      205.0       7.21            1.150           260.8     96.41             Type 1    type1_uncontrolled
```
