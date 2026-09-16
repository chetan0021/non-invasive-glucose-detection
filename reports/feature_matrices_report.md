# Model-Ready Feature Matrices & Class Balance Report

**Generated On**: 2026-09-10

## 1. Feature Matrix Summary

| Feature Matrix File | Rows | Columns | Scaler | Manifest |
| :--- | :---: | :---: | :--- | :--- |
| `full_sensor_train_features.csv` | **469** | **92** | `models/scaler_full_sensor.pkl` | `reports/features_manifest_full_sensor.json` |
| `full_sensor_test_features.csv` | **125** | **92** | (Applied) | (Same Schema) |
| `tabular_train_features.csv` | **19,318** | **46** | `models/scaler_tabular.pkl` | `reports/features_manifest_tabular.json` |
| `tabular_test_features.csv` | **4,831** | **46** | (Applied) | (Same Schema) |

## 2. Class Balance per `diabetes_status` Across Feature Matrices

| diabetes_status           |   Full Sensor Train (N) | Full Sensor Train (%)   |   Full Sensor Test (N) | Full Sensor Test (%)   |   Tabular Train (N) | Tabular Train (%)   |   Tabular Test (N) | Tabular Test (%)   |
|:--------------------------|------------------------:|:------------------------|-----------------------:|:-----------------------|--------------------:|:--------------------|-------------------:|:-------------------|
| healthy                   |                      60 | 12.8%                   |                     15 | 12.0%                  |                 636 | 3.3%                |                185 | 3.8%               |
| prediabetes_elevated      |                      80 | 17.1%                   |                     28 | 22.4%                  |                 202 | 1.0%                |                 51 | 1.1%               |
| prediabetes_high          |                       0 | 0.0%                    |                      0 | 0.0%                   |                   9 | 0.0%                |                  2 | 0.0%               |
| prediabetes_normoglycemic |                       0 | 0.0%                    |                      0 | 0.0%                   |                  57 | 0.3%                |                 14 | 0.3%               |
| type1_elevated            |                       9 | 1.9%                    |                      1 | 0.8%                   |                 193 | 1.0%                |                 38 | 0.8%               |
| type1_hypoglycemic        |                      41 | 8.7%                    |                     12 | 9.6%                   |                   0 | 0.0%                |                  0 | 0.0%               |
| type1_normoglycemic       |                       1 | 0.2%                    |                      0 | 0.0%                   |                 198 | 1.0%                |                 66 | 1.4%               |
| type1_severe              |                      30 | 6.4%                    |                     11 | 8.8%                   |                   0 | 0.0%                |                  0 | 0.0%               |
| type1_uncontrolled        |                       8 | 1.7%                    |                      0 | 0.0%                   |                1010 | 5.2%                |                266 | 5.5%               |
| type2_controlled          |                     108 | 23.0%                   |                     27 | 21.6%                  |                4733 | 24.5%               |               1189 | 24.6%              |
| type2_normoglycemic       |                       3 | 0.6%                    |                      0 | 0.0%                   |                4063 | 21.0%               |               1018 | 21.1%              |
| type2_severe              |                      17 | 3.6%                    |                      2 | 1.6%                   |                2790 | 14.4%               |                692 | 14.3%              |
| type2_uncontrolled        |                      50 | 10.7%                   |                     16 | 12.8%                  |                4633 | 24.0%               |               1137 | 23.5%              |
| undiagnosed_elevated      |                      62 | 13.2%                   |                     13 | 10.4%                  |                 786 | 4.1%                |                172 | 3.6%               |
| undiagnosed_high          |                       0 | 0.0%                    |                      0 | 0.0%                   |                   8 | 0.0%                |                  1 | 0.0%               |

## 3. Stratified Evaluation & Sample Sparsity Assessment for Phase 6

> [!IMPORTANT]
> **Full-Sensor Sample Size Assessment**:
> - In `full_sensor_train_features.csv` (469 rows across 120 participants) and `full_sensor_test_features.csv` (125 rows across 30 participants), subcategories like `type2_controlled` (27 test rows) or `type2_uncontrolled` (16 test rows) have modest sample counts.
> - For Phase 6 model evaluation, primary metrics (R², MAE, RMSE, Clarke Error Grid) should be evaluated on the aggregated test set and grouped by broader clinical categories (`diabetes_diagnosis`: None, Type 1, Type 2, Prediabetes) in addition to individual status slices to ensure statistical power.
