# Non-Invasive Blood Glucose Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An interactive machine learning system and clinical research dashboard for **multi-modal non-invasive blood glucose estimation** and **demographic cardiometabolic risk screening**.

---

## 🌟 Key Features

1. **Multi-Modal Point & Quantile BGL Estimation (Model A)**:
   - Evaluates optical PPG features (MAX30102 DC/AC counts, pulse width), biochemical markers (saliva pH), skin temperature, and autonomic heart rate variability (ECG HRV SDNN/RMSSD).
   - Random Forest Regressor ($R^2 = 0.8557$, $\text{MAE} = 12.13\text{ mg/dL}$, $99.22\%$ Clarke Zone A+B).
   - Gradient Boosting Quantile Interval $[q_{0.05}, q_{0.95}]$ ($85.16\%$ empirical test coverage).
   - Dynamic glycemic trend and Clarke Error Grid tiering.

2. **2-Band Demographic Risk Screening (Model B)**:
   - Population-level metabolic screening based on age, BMI, family history, and smoking status.
   - Validated on CDC NHANES community survey outpatients ($\text{Macro AUROC} = 0.7296$).
   - Returns honest 2-band classification (`LOWER BASELINE RISK` vs. `ELEVATED RISK — CONSULT RECOMMENDED`) with transparent clinical screening disclaimers.

3. **Clinical PDF Summary Export & Audit Logging**:
   - Automated ReportLab PDF generation with session biometrics, prediction bounds, and clinical guidance.
   - Local CSV audit log tracking.

---

## 🚀 Quickstart (Local)

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit Dashboard
streamlit run app/dashboard.py
```

---

## 📋 Repository Structure

```
├── app/
│   └── dashboard.py          # Interactive Streamlit Web Application
├── models/
│   ├── production_model_full_sensor.pkl
│   ├── quantile_regressor_full_sensor.pkl
│   ├── production_model_tabular_riskclass.pkl
│   └── *.json                # Model manifests and scalers
├── predict.py                # Core Python Inference Engine & API
├── requirements.txt          # Deployment dependencies
└── README.md
```

---

## ⚠️ Research Prototype Notice
This project is an experimental research prototype. The full-sensor regression model is validated on synthetic multi-modal self-consistency data only. The tabular risk classifier is validated on CDC NHANES community survey outpatients only. This software is **not a certified medical device** and is **not a substitute for certified clinical laboratory blood testing**.
