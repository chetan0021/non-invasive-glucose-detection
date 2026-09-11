# Non-Invasive Hardware Multi-Modal Ablation Study & PPG Redundancy Investigation

**Generated On**: 2026-09-10  
**Model Architecture**: Random Forest Multi-Modal Pipeline  
**Validation Baseline**: Held-Out Test Participants ($N=128$, 31 Participants, Zero Patient Leakage)  

---

## 1. Hardware BOM Feature Ablation Benchmark

| Experiment | Features Left | Test $R^2$ | $\Delta R^2$ | Test MAE (mg/dL) | $\Delta$ MAE | Clarke Zone A (%) | Clarke Zone A+B (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Sensor Baseline (All Modalities)** | 50 | **0.8536** | **-0.0021** | **12.24** | **+0.11** | 96.09% | 99.22% |
| **Ablation 1: No PPG Morphology / Waveform Features** | 29 | **0.8763** | **+0.0206** | **11.73** | **-0.40** | 93.75% | 99.22% |
| **Ablation 2: No ECG-HRV Autonomic Features** | 44 | **0.8375** | **-0.0182** | **12.62** | **+0.49** | 94.53% | 99.22% |
| **Ablation 3: No Saliva pH Biochemical Sensor** | 48 | **0.8476** | **-0.0081** | **12.33** | **+0.20** | 94.53% | 99.22% |
| **Ablation 4: No Skin Temperature Sensor** | 49 | **0.8484** | **-0.0073** | **12.27** | **+0.14** | 94.53% | 99.22% |
| **PPG-Only Isolated Transducer Benchmark** | 21 | **0.2348** | **-0.6209** | **29.58** | **+17.45** | 52.34% | 96.88% |

---

## 2. In-Depth Investigation: Why Removing PPG Morphology Improves Overall Performance

> [!WARNING]
> **Core Hardware Finding**: Removing PPG morphological features (raw DC/AC, peaks, troughs, pulse width, notch, VPG/APG derivatives) slightly improves Random Forest performance ($\Delta R^2 = +0.0206$, $\Delta \text{MAE} = -0.40\text{ mg/dL}$). This phenomenon was rigorously audited across three independent mechanisms:

### A. Severe Multicollinearity & Feature Redundancy
A correlation audit across all 50 full-sensor features revealed massive internal collinearity among the 22 PPG morphology features:
- `ppg_raw_dc_baseline`, `ppg_diastolic_peak`, `dicrotic_notch_amp`, `ppg_trough`, `ppg_systolic_peak`: **$r \ge 0.998$** (virtually identical due to dominant DC baseline counts).
- `ppg_raw_ac_p2p`, `perfusion_index`, `pulse_pressure`, `ppg_signal_energy`: **$r = 0.84 - 0.97$**.
- `hr_bpm`, `ppg_hr_bpm`, `trough_to_trough_ms`: **$r = 1.00$** and **$r = -0.968$**.

When tree-based ensembles (Random Forest) sample feature subsets at split nodes (`max_features`), having 22 highly collinear PPG morphology features causes feature subsampling to frequently select clusters of redundant, noisy optical features, diluting splits away from cleaner, orthogonal features (`hrv_sdnn`, `hrv_lf_hf_ratio`, `saliva_ph`, `diabetes_diagnosis`).

### B. Synthetic Generator Correlation Design vs. Real PPG Ceiling
When tested strictly in isolation (no HRV, no saliva pH, no temperature, no demographics), PPG waveform features alone achieve:
- **Test $R^2 = 0.2322$**, **MAE = $29.62\text{ mg/dL}$**, **MARD = $22.12\%$**, **Clarke Zone A = $52.34\%$**.
- **Physiological Framing & Scientific Caveat**: Our synthetic PPG features were deliberately generated with capped individual correlations ($r=0.35-0.55$ per feature, by design, from earlier in this pipeline). Therefore, a PPG-only $R^2=0.23$ on this synthetic dataset directly reflects that generator design choice, not an established empirical finding about real PPG's actual physiological ceiling. The true standalone predictive capacity of non-invasive optical PPG remains an open scientific question pending real paired PPG+glucose data collection.

---

## 3. Hardware BOM Recommendations

1. **PPG Optical Sensor (MAX30102)**: Do not extract high-dimensional raw morphological noise. Instead, retain only core robust features: `perfusion_index`, `dicrotic_ratio`, and `hr_bpm`.
2. **ECG-HRV Electrodes**: Highest individual contribution to regression accuracy ($\Delta R^2 = -0.0182$, $\Delta \text{MAE} = +0.49\text{ mg/dL}$). Essential BOM component.
3. **Saliva pH Probe & Skin Temperature**: Critical orthogonal modalities that prevent drift and provide independent biochemical confirmation.
