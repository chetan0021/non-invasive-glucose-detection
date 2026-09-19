# Model-Ready Feature Matrices & Class Balance Report

**Generated On**: 2026-09-10

## 1. Feature Matrix Summary

| Feature Matrix File | Rows | Columns | Scaler | Manifest |
| :--- | :---: | :---: | :--- | :--- |
| `full_sensor_train_features.csv` | **486** | **92** | `models/scaler_full_sensor.pkl` | `reports/features_manifest_full_sensor.json` |
| `full_sensor_test_features.csv` | **109** | **92** | (Applied) | (Same Schema) |
| `tabular_train_features.csv` | **19,332** | **46** | `models/scaler_tabular.pkl` | `reports/features_manifest_tabular.json` |
| `tabular_test_features.csv` | **4,817** | **46** | (Applied) | (Same Schema) |

## 2. Class Balance per `diabetes_status` Across Feature Matrices

| diabetes_status           |   Full Sensor Train (N) | Full Sensor Train (%)   |   Full Sensor Test (N) | Full Sensor Test (%)   |   Tabular Train (N) | Tabular Train (%)   |   Tabular Test (N) | Tabular Test (%)   |
|:--------------------------|------------------------:|:------------------------|-----------------------:|:-----------------------|--------------------:|:--------------------|-------------------:|:-------------------|
| healthy                   |                      58 | 11.9%                   |                     16 | 14.7%                  |                 643 | 3.3%                |                178 | 3.7%               |
| prediabetes_elevated      |                      70 | 14.4%                   |                     17 | 15.6%                  |                 200 | 1.0%                |                 53 | 1.1%               |
| prediabetes_high          |                       0 | 0.0%                    |                      0 | 0.0%                   |                   9 | 0.0%                |                  2 | 0.0%               |
| prediabetes_normoglycemic |                       0 | 0.0%                    |                      0 | 0.0%                   |                  59 | 0.3%                |                 12 | 0.2%               |
| type1_elevated            |                       7 | 1.4%                    |                      1 | 0.9%                   |                 191 | 1.0%                |                 40 | 0.8%               |
| type1_hypoglycemic        |                      46 | 9.5%                    |                      8 | 7.3%                   |                   0 | 0.0%                |                  0 | 0.0%               |
| type1_normoglycemic       |                       0 | 0.0%                    |                      0 | 0.0%                   |                 200 | 1.0%                |                 64 | 1.3%               |
| type1_severe              |                      33 | 6.8%                    |                      8 | 7.3%                   |                   0 | 0.0%                |                  0 | 0.0%               |
| type1_uncontrolled        |                       7 | 1.4%                    |                      1 | 0.9%                   |                1018 | 5.3%                |                258 | 5.4%               |
| type2_controlled          |                     142 | 29.2%                   |                     26 | 23.9%                  |                4734 | 24.5%               |               1188 | 24.7%              |
| type2_normoglycemic       |                       3 | 0.6%                    |                      0 | 0.0%                   |                4071 | 21.1%               |               1010 | 21.0%              |
| type2_severe              |                      19 | 3.9%                    |                      7 | 6.4%                   |                2783 | 14.4%               |                699 | 14.5%              |
| type2_uncontrolled        |                      40 | 8.2%                    |                     10 | 9.2%                   |                4637 | 24.0%               |               1133 | 23.5%              |
| undiagnosed_elevated      |                      61 | 12.6%                   |                     15 | 13.8%                  |                 779 | 4.0%                |                179 | 3.7%               |
| undiagnosed_high          |                       0 | 0.0%                    |                      0 | 0.0%                   |                   8 | 0.0%                |                  1 | 0.0%               |

## 3. Stratified Evaluation & Sample Sparsity Assessment for Phase 6

> [!IMPORTANT]
> **Full-Sensor Sample Size Assessment**:
> - In `full_sensor_train_features.csv` (486 rows across 120 participants) and `full_sensor_test_features.csv` (109 rows across 30 participants), subcategories like `type2_controlled` (26 test rows) or `type2_uncontrolled` (10 test rows) have modest sample counts.
> - For Phase 6 model evaluation, primary metrics (R², MAE, RMSE, Clarke Error Grid) should be evaluated on the aggregated test set and grouped by broader clinical categories (`diabetes_diagnosis`: None, Type 1, Type 2, Prediabetes) in addition to individual status slices to ensure statistical power.
