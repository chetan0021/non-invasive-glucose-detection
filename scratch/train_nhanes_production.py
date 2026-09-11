import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb

BASE_DIR = Path(r"c:\Users\Chetan\Documents\Non Invasive Glucouse Detection\glucose-prediction")
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

df_train = pd.read_csv(DATA_DIR / "tabular_train_features.csv")
df_test = pd.read_csv(DATA_DIR / "tabular_test_features.csv")

def derive_clean_risk_label(row: pd.Series) -> int:
    diag = str(row.get("diabetes_diagnosis", "None"))
    med_any = int(row.get("med_taking_any", 0)) if "med_taking_any" in row else 0
    med_ins = int(row.get("med_taking_insulin", 0)) if "med_taking_insulin" in row else 0
    med_oral = int(row.get("med_taking_oral", 0)) if "med_taking_oral" in row else 0

    if diag in ["Type 1", "Type 2", "Gestational"] or med_any == 1 or med_ins == 1 or med_oral == 1:
        return 2  # diabetic_risk
    elif diag == "Prediabetes":
        return 1  # elevated_risk
    else:
        return 0  # healthy_risk

df_train["risk_label"] = df_train.apply(derive_clean_risk_label, axis=1)
df_test["risk_label"] = df_test.apply(derive_clean_risk_label, axis=1)

df_train["source"] = df_train["participant_id"].apply(lambda x: str(x).split("_")[0])
df_test["source"] = df_test["participant_id"].apply(lambda x: str(x).split("_")[0])

# Genuine outpatient demographic features
risk_feature_cols = [
    "age_scaled",
    "bmi_scaled",
    "gender_male",
    "bmi_cat_underweight",
    "bmi_cat_normal",
    "bmi_cat_overweight",
    "bmi_cat_obese",
    "family_history",
    "smoking"
]

class_names = ["healthy_risk", "elevated_risk", "diabetic_risk"]

# Filter strictly to NHANES community cohort
nhanes_train = df_train[df_train["source"] == "nhanes"].copy()
nhanes_test = df_test[df_test["source"] == "nhanes"].copy()

X_tr = nhanes_train[risk_feature_cols]
y_tr = nhanes_train["risk_label"].values

X_te = nhanes_test[risk_feature_cols]
y_te = nhanes_test["risk_label"].values

# Apply SMOTE oversampling to training set only
smote = SMOTE(random_state=42, k_neighbors=3)
X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

# Train XGBoost Multi-Class Classifier
sw = compute_sample_weight("balanced", y_tr_res)
clf = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_tr_res, y_tr_res, sample_weight=sw)

# Evaluate on held-out test set
test_probs = clf.predict_proba(X_te)
test_preds = np.argmax(test_probs, axis=1)

macro_auroc = float(roc_auc_score(y_te, test_probs, multi_class="ovr", average="macro"))
per_class_auroc = roc_auc_score(y_te, test_probs, multi_class="ovr", average=None)
cm = confusion_matrix(y_te, test_preds).tolist()
report_dict = classification_report(y_te, test_preds, target_names=class_names, output_dict=True, zero_division=0)

brier_scores = {}
for i, c_name in enumerate(class_names):
    y_bin = (y_te == i).astype(int)
    brier_scores[c_name] = round(float(brier_score_loss(y_bin, test_probs[:, i])), 4)

print("="*80)
print(f"PRODUCTION TABULAR RISK CLASSIFIER (COMMUNITY OUTPATIENT SCOPE - NHANES)")
print(f"Macro AUROC: {macro_auroc:.4f}")
for i, c_name in enumerate(class_names):
    print(f"  - {c_name:15s} | AUROC: {per_class_auroc[i]:.4f} | Prec: {report_dict[c_name]['precision']:.4f} | Rec: {report_dict[c_name]['recall']:.4f} | F1: {report_dict[c_name]['f1-score']:.4f} | Support: {report_dict[c_name]['support']} | Brier: {brier_scores[c_name]}")
print(f"Confusion Matrix [rows=True, cols=Pred]:\n{np.array(cm)}")
print("="*80)

# Also test evaluated on pooled test set for reference
pooled_probs = clf.predict_proba(df_test[risk_feature_cols])
pooled_macro_auc = float(roc_auc_score(df_test["risk_label"], pooled_probs, multi_class="ovr", average="macro"))
print(f"Generalization of NHANES-Trained Model to Pooled Inpatient/Outpatient Test Set: Macro AUROC = {pooled_macro_auc:.4f}")
