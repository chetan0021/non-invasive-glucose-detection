import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

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

risk_features = [
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

class_names = ["healthy_risk", "elevated_risk", "diabetic_risk"]

# Experiment with and without source_dataset feature
df_train["source_dataset_d130"] = (df_train["source"] == "d130").astype(int)
df_test["source_dataset_d130"] = (df_test["source"] == "d130").astype(int)

risk_features_with_source = risk_features + ["source_dataset_d130"]

def run_eval(X_tr, y_tr, X_te, y_te, name):
    sw = compute_sample_weight("balanced", y_tr)
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr, sample_weight=sw)
    probs = clf.predict_proba(X_te)
    preds = np.argmax(probs, axis=1)

    macro_auc = roc_auc_score(y_te, probs, multi_class="ovr", average="macro")
    per_cls = roc_auc_score(y_te, probs, multi_class="ovr", average=None)
    brier = {class_names[i]: brier_score_loss((y_te == i).astype(int), probs[:, i]) for i in range(3)}
    
    print(f"\n==========================================")
    print(f"{name}")
    print(f"Macro AUROC: {macro_auc:.4f}")
    for idx, c in enumerate(class_names):
        print(f"  {c} AUROC: {per_cls[idx]:.4f} | Brier: {brier[c]:.4f}")
    
    return clf, probs

# Run baseline on clean labels
run_eval(df_train[risk_features], df_train["risk_label"], df_test[risk_features], df_test["risk_label"], "A. Combined Data (Standard Independent Features)")

# Run with explicit source_dataset feature
run_eval(df_train[risk_features_with_source], df_train["risk_label"], df_test[risk_features_with_source], df_test["risk_label"], "B. Combined Data WITH Explicit source_dataset feature")

# Run NHANES only
nhanes_tr = df_train[df_train["source"] == "nhanes"]
nhanes_te = df_test[df_test["source"] == "nhanes"]
run_eval(nhanes_tr[risk_features], nhanes_tr["risk_label"], nhanes_te[risk_features], nhanes_te["risk_label"], "C. NHANES-Only (Real Cohort Zero-Confound Demographic Benchmark)")

