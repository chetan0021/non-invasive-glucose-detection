# Model-Ready Feature Matrices & Class Balance Report

**Generated On**: 2026-09-10

## 1. Feature Matrix Summary

| Feature Matrix File | Rows | Columns | Scaler | Manifest |
| :--- | :---: | :---: | :--- | :--- |
| `full_sensor_train_features.csv` | **488** | **92** | `models/scaler_full_sensor.pkl` | `reports/features_manifest_full_sensor.json` |
| `full_sensor_test_features.csv` | **128** | **92** | (Applied) | (Same Schema) |
| `tabular_train_features.csv` | **19,317** | **29** | `models/scaler_tabular.pkl` | `reports/features_manifest_tabular.json` |
| `tabular_test_features.csv` | **4,832** | **29** | (Applied) | (Same Schema) |

## 2. Class Balance per `diabetes_status` Across Feature Matrices

| diabetes_status           |   Full Sensor Train (N) | Full Sensor Train (%)   |   Full Sensor Test (N) | Full Sensor Test (%)   |   Tabular Train (N) | Tabular Train (%)   |   Tabular Test (N) | Tabular Test (%)   |
|:--------------------------|------------------------:|:------------------------|-----------------------:|:-----------------------|--------------------:|:--------------------|-------------------:|:-------------------|
| healthy                   |                      74 | 15.2%                   |                     15 | 11.7%                  |                 694 | 3.6%                |                182 | 3.8%               |
| prediabetes_elevated      |                      94 | 19.3%                   |                     27 | 21.1%                  |                  49 | 0.3%                |                 13 | 0.3%               |
| prediabetes_high          |                       0 | 0.0%                    |                      0 | 0.0%                   |                   1 | 0.0%                |                  1 | 0.0%               |
| prediabetes_normoglycemic |                       0 | 0.0%                    |                      0 | 0.0%                   |                  14 | 0.1%                |                  2 | 0.0%               |
| type1_elevated            |                      38 | 7.8%                    |                     11 | 8.6%                   |                 190 | 1.0%                |                 41 | 0.8%               |
| type1_normoglycemic       |                       8 | 1.6%                    |                      1 | 0.8%                   |                 196 | 1.0%                |                 68 | 1.4%               |
| type1_uncontrolled        |                      48 | 9.8%                    |                     12 | 9.4%                   |                1018 | 5.3%                |                258 | 5.3%               |
| type2_controlled          |                     103 | 21.1%                   |                     34 | 26.6%                  |                4751 | 24.6%               |               1171 | 24.2%              |
| type2_normoglycemic       |                       3 | 0.6%                    |                      0 | 0.0%                   |                4049 | 21.0%               |               1032 | 21.4%              |
| type2_severe              |                      16 | 3.3%                    |                      1 | 0.8%                   |                2773 | 14.4%               |                709 | 14.7%              |
| type2_uncontrolled        |                      38 | 7.8%                    |                      7 | 5.5%                   |                4641 | 24.0%               |               1129 | 23.4%              |
| undiagnosed_elevated      |                      66 | 13.5%                   |                     20 | 15.6%                  |                 928 | 4.8%                |                221 | 4.6%               |
| undiagnosed_high          |                       0 | 0.0%                    |                      0 | 0.0%                   |                  13 | 0.1%                |                  5 | 0.1%               |

## 3. Stratified Evaluation & Sample Sparsity Assessment for Phase 6

> [!IMPORTANT]
> **Full-Sensor Sample Size Assessment**:
> - In `full_sensor_train_features.csv` (488 rows across 120 participants) and `full_sensor_test_features.csv` (128 rows across 30 participants), subcategories like `type2_controlled` (34 test rows) or `type2_uncontrolled` (7 test rows) have modest sample counts.
> - For Phase 6 model evaluation, primary metrics (R², MAE, RMSE, Clarke Error Grid) should be evaluated on the aggregated test set and grouped by broader clinical categories (`diabetes_diagnosis`: None, Type 1, Type 2, Prediabetes) in addition to individual status slices to ensure statistical power.
