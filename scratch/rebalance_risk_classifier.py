import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(r"c:\Users\Chetan\Documents\Non Invasive Glucouse Detection\glucose-prediction")
DATA_DIR = BASE_DIR / "data" / "processed"

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

class_names = ["healthy_risk", "elevated_risk", "diabetic_risk"]

# Genuine demographic screening features (excluding dataset-shortcut features like fasting and bmi_cat_missing if appropriate)
features_all = [
    "age_scaled",
    "bmi_scaled",
    "gender_male",
    "bmi_cat_underweight",
    "bmi_cat_normal",
    "bmi_cat_overweight",
    "bmi_cat_obese",
    "bmi_cat_missing",
    "family_history",
    "smoking",
    "fasting"
]

features_demographics_pure = [
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

print("="*80)
print("EXPERIMENTING WITH RISK CLASSIFIER REBALANCING & DATASET POPULATION SCOPING")
print("="*80)

# Isolate NHANES
nhanes_train = df_train[df_train["source"] == "nhanes"].copy()
nhanes_test = df_test[df_test["source"] == "nhanes"].copy()

print("NHANES Train Distribution:", nhanes_train["risk_label"].value_counts().to_dict())
print("NHANES Test Distribution:", nhanes_test["risk_label"].value_counts().to_dict())

def test_pipeline(X_tr, y_tr, X_te, y_te, method_name="Baseline", f_cols=features_demographics_pure, resampler=None):
    if resampler is not None:
        X_tr_res, y_tr_res = resampler.fit_resample(X_tr[f_cols], y_tr)
    else:
        X_tr_res, y_tr_res = X_tr[f_cols], y_tr

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

    probs = clf.predict_proba(X_te[f_cols])
    preds = np.argmax(probs, axis=1)

    macro_auc = roc_auc_score(y_te, probs, multi_class="ovr", average="macro")
    per_cls_auc = roc_auc_score(y_te, probs, multi_class="ovr", average=None)
    rep = classification_report(y_te, preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, preds)

    print(f"\n--- Method: {method_name} ---")
    print(f"Macro AUROC: {macro_auc:.4f}")
    for idx, c in enumerate(class_names):
        print(f"  {c:15s} | AUROC: {per_cls_auc[idx]:.4f} | Prec: {rep[c]['precision']:.4f} | Rec: {rep[c]['recall']:.4f} | F1: {rep[c]['f1-score']:.4f} | Support: {rep[c]['support']}")
    print(f"Confusion Matrix [rows=True, cols=Pred]:\n{cm}")
    return clf, macro_auc, rep, cm

print("\n" + "#"*80)
print("1. EVALUATION ON NHANES COMMUNITY OUTPATIENT COHORT (ZERO-CONFOUND REAL DATA)")
print("#"*80)

test_pipeline(nhanes_train, nhanes_train["risk_label"], nhanes_test, nhanes_test["risk_label"], 
              method_name="NHANES-Only (Class Weighting)", f_cols=features_demographics_pure)

test_pipeline(nhanes_train, nhanes_train["risk_label"], nhanes_test, nhanes_test["risk_label"], 
              method_name="NHANES-Only (SMOTE Rebalancing)", f_cols=features_demographics_pure,
              resampler=SMOTE(random_state=42, k_neighbors=3))

test_pipeline(nhanes_train, nhanes_train["risk_label"], nhanes_test, nhanes_test["risk_label"], 
              method_name="NHANES-Only (RandomOverSampler)", f_cols=features_demographics_pure,
              resampler=RandomOverSampler(random_state=42))

print("\n" + "#"*80)
print("2. EVALUATION ON POOLED SET WITH CLEAN DEMOGRAPHIC FEATURES & SMOTE")
print("#"*80)

test_pipeline(df_train, df_train["risk_label"], df_test, df_test["risk_label"], 
              method_name="Pooled (Pure Demographics + SMOTE)", f_cols=features_demographics_pure,
              resampler=SMOTE(random_state=42, k_neighbors=3))
