import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, brier_score_loss
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
        return 2
    elif diag == "Prediabetes":
        return 1
    else:
        return 0

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
    "family_history",
    "smoking"
]

class_names = ["healthy_risk", "elevated_risk", "diabetic_risk"]

nhanes_train = df_train[df_train["source"] == "nhanes"].copy()
nhanes_test = df_test[df_test["source"] == "nhanes"].copy()

X_tr = nhanes_train[risk_features]
y_tr = nhanes_train["risk_label"].values
X_te = nhanes_test[risk_features]
y_te = nhanes_test["risk_label"].values

sw = compute_sample_weight("balanced", y_tr)

for md in [2, 3, 4, 5]:
    for lr in [0.03, 0.05, 0.1]:
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=100,
            max_depth=md,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_tr, y_tr, sample_weight=sw)
        probs = clf.predict_proba(X_te)
        macro_auc = roc_auc_score(y_te, probs, multi_class="ovr", average="macro")
        per_cls = roc_auc_score(y_te, probs, multi_class="ovr", average=None)
        preds = np.argmax(probs, axis=1)
        rep = classification_report(y_te, preds, target_names=class_names, output_dict=True, zero_division=0)
        elev_rec = rep["elevated_risk"]["recall"]
        elev_prec = rep["elevated_risk"]["precision"]
        print(f"Depth {md}, LR {lr} -> Macro AUC: {macro_auc:.4f} | Diab AUC: {per_cls[2]:.4f} | Health AUC: {per_cls[0]:.4f} | Elev AUC: {per_cls[1]:.4f} | Elev Prec: {elev_prec:.4f} | Elev Rec: {elev_rec:.4f}")

