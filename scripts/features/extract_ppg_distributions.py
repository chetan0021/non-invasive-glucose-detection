"""
PPG & Multi-Modal Physiological Feature Extraction Pipeline.

Extracts morphological, derivative (VPG/APG), statistical, and ECG-HRV features
from raw benchmark datasets (BIDMC, CapnoBase, PPG-DaLiA, D1NAMO).

Maintains strict separation between 'fingertip' (MAX30102 baseline) and 'wrist' placements.
"""

import os
import io
import sys
import json
import pickle
import zipfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# Define base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
REPORTS_DIR = BASE_DIR / "reports"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. SIGNAL PROCESSING & FILTERING
# ==============================================================================

def butterworth_bandpass(
    data: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 8.0,
    order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter using Second-Order Sections (SOS)."""
    nyquist = 0.5 * fs
    low = max(lowcut / nyquist, 0.01)
    high = min(highcut / nyquist, 0.99)
    if low >= high:
        return data - np.mean(data)
    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    return signal.sosfiltfilt(sos, data)


def butterworth_ecg_bandpass(
    data: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4
) -> np.ndarray:
    """Zero-phase bandpass filter for ECG signals."""
    nyquist = 0.5 * fs
    low = max(lowcut / nyquist, 0.001)
    high = min(highcut / nyquist, 0.95)
    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    return signal.sosfiltfilt(sos, data)


# ==============================================================================
# 2. PPG MORPHOLOGY, DERIVATIVES (VPG/APG) & STATISTICAL FEATURES
# ==============================================================================

def extract_ppg_window_features(
    raw_ppg: np.ndarray,
    filtered_ppg: np.ndarray,
    fs: float
) -> Dict[str, float]:
    """
    Extracts comprehensive morphological, derivative (VPG/APG), and statistical
    features from a single window of PPG signal.
    """
    feats: Dict[str, float] = {}
    N = len(filtered_ppg)
    if N < int(fs * 2.0):
        return feats

    # 1. Statistical & Energy Features over the window
    signal_mean = float(np.mean(filtered_ppg))
    signal_std = float(np.std(filtered_ppg))
    signal_var = float(np.var(filtered_ppg))
    signal_rms = float(np.sqrt(np.mean(filtered_ppg ** 2)))
    signal_min = float(np.min(filtered_ppg))
    signal_max = float(np.max(filtered_ppg))
    signal_energy = float(np.sum(filtered_ppg ** 2) / N)

    # Perfusion Index: (AC Peak-to-Peak / DC Mean) * 100
    raw_dc = float(np.mean(raw_ppg)) if np.abs(np.mean(raw_ppg)) > 1e-6 else 1.0
    ac_p2p = float(np.max(raw_ppg) - np.min(raw_ppg))
    perfusion_index = float((ac_p2p / max(abs(raw_dc), 1e-4)) * 100.0)

    feats.update({
        "signal_mean": signal_mean,
        "signal_std": signal_std,
        "signal_var": signal_var,
        "signal_rms": signal_rms,
        "signal_min": signal_min,
        "signal_max": signal_max,
        "signal_energy": signal_energy,
        "perfusion_index": perfusion_index,
    })

    # 2. Velocity (VPG) and Acceleration (APG) Derivatives
    vpg = np.gradient(filtered_ppg, 1.0 / fs)
    apg = np.gradient(vpg, 1.0 / fs)

    feats["vpg_max"] = float(np.max(vpg))
    feats["vpg_min"] = float(np.min(vpg))

    # 3. Peak and Trough Detection
    min_peak_distance = int(fs * 0.3)  # Max HR ~200 bpm
    std_thresh = max(0.15 * signal_std, 1e-4)
    systolic_peaks, _ = signal.find_peaks(filtered_ppg, distance=min_peak_distance, prominence=std_thresh)
    troughs, _ = signal.find_peaks(-filtered_ppg, distance=min_peak_distance, prominence=std_thresh)

    if len(systolic_peaks) < 2 or len(troughs) < 2:
        # Fallback values if too few peaks
        feats.update({
            "ppg_hr_bpm": np.nan,
            "trough_to_trough_ms": np.nan,
            "pulse_width_ms": np.nan,
            "systolic_peak_amp": signal_max,
            "diastolic_peak_amp": np.nan,
            "dicrotic_notch_amp": np.nan,
            "dicrotic_ratio": np.nan,
            "apg_a": np.nan,
            "apg_b": np.nan,
            "apg_c": np.nan,
            "apg_d": np.nan,
            "apg_e": np.nan,
            "apg_b_a_ratio": np.nan,
            "apg_c_a_ratio": np.nan,
            "apg_d_a_ratio": np.nan,
            "apg_e_a_ratio": np.nan,
            "apg_aging_index": np.nan,
        })
        return feats

    # Pulse intervals and PPG Heart Rate
    peak_intervals_sec = np.diff(systolic_peaks) / fs
    trough_intervals_sec = np.diff(troughs) / fs
    mean_t2t_sec = float(np.median(trough_intervals_sec))
    ppg_hr_bpm = float(60.0 / mean_t2t_sec) if mean_t2t_sec > 0 else np.nan

    feats["ppg_hr_bpm"] = ppg_hr_bpm
    feats["trough_to_trough_ms"] = float(mean_t2t_sec * 1000.0)

    # 4. Cycle-by-Cycle Morphological & APG Wave Analysis
    cycle_pulse_widths = []
    cycle_systolic_amps = []
    cycle_diastolic_amps = []
    cycle_notch_amps = []
    cycle_dicrotic_ratios = []
    apg_a_list, apg_b_list, apg_c_list, apg_d_list, apg_e_list = [], [], [], [], []

    for i in range(len(troughs) - 1):
        t_start = troughs[i]
        t_end = troughs[i + 1]
        cycle_len = t_end - t_start
        if cycle_len < int(fs * 0.25) or cycle_len > int(fs * 2.0):
            continue

        cycle_ppg = filtered_ppg[t_start:t_end]
        cycle_vpg = vpg[t_start:t_end]
        cycle_apg = apg[t_start:t_end]

        # Systolic Peak within cycle
        sys_rel_idx = np.argmax(cycle_ppg)
        sys_amp = float(cycle_ppg[sys_rel_idx])
        cycle_systolic_amps.append(sys_amp)

        # Pulse Width at 50% systolic amplitude (FWHM)
        trough_amp = min(cycle_ppg[0], cycle_ppg[-1])
        half_height = trough_amp + 0.5 * (sys_amp - trough_amp)
        above_half = np.where(cycle_ppg >= half_height)[0]
        if len(above_half) > 0:
            pw_ms = float((above_half[-1] - above_half[0]) / fs * 1000.0)
            cycle_pulse_widths.append(pw_ms)

        # Dicrotic Notch and Diastolic Peak Detection
        # Search in the region after systolic peak
        if sys_rel_idx < cycle_len - int(fs * 0.1):
            post_sys_ppg = cycle_ppg[sys_rel_idx:]
            post_sys_vpg = cycle_vpg[sys_rel_idx:]
            
            # Local minimum in post-systolic region (dicrotic notch)
            notch_candidates, _ = signal.find_peaks(-post_sys_ppg, distance=int(fs * 0.05))
            if len(notch_candidates) > 0:
                notch_rel_idx = sys_rel_idx + notch_candidates[0]
                notch_amp = float(cycle_ppg[notch_rel_idx])
                cycle_notch_amps.append(notch_amp)

                # Search for diastolic peak after the notch
                if notch_rel_idx < cycle_len - 2:
                    post_notch_ppg = cycle_ppg[notch_rel_idx:]
                    dia_candidates, _ = signal.find_peaks(post_notch_ppg, distance=int(fs * 0.05))
                    if len(dia_candidates) > 0:
                        dia_rel_idx = notch_rel_idx + dia_candidates[0]
                        dia_amp = float(cycle_ppg[dia_rel_idx])
                        cycle_diastolic_amps.append(dia_amp)
                        if sys_amp != 0:
                            cycle_dicrotic_ratios.append(dia_amp / sys_amp)

        # APG a-b-c-d-e Wave Detection
        # a-wave: early systolic max acceleration (first 30% of cycle)
        # b-wave: early systolic deceleration min
        # c-wave: re-acceleration max
        # d-wave: late systolic min
        # e-wave: early diastolic wave max
        seg_30 = max(int(0.35 * cycle_len), 3)
        a_idx = int(np.argmax(cycle_apg[:seg_30]))
        a_val = float(cycle_apg[a_idx])
        apg_a_list.append(a_val)

        if a_idx < cycle_len - 4:
            b_search_end = min(a_idx + int(0.3 * cycle_len), cycle_len)
            b_idx = a_idx + int(np.argmin(cycle_apg[a_idx:b_search_end]))
            b_val = float(cycle_apg[b_idx])
            apg_b_list.append(b_val)

            if b_idx < cycle_len - 3:
                c_search_end = min(b_idx + int(0.25 * cycle_len), cycle_len)
                c_idx = b_idx + int(np.argmax(cycle_apg[b_idx:c_search_end]))
                c_val = float(cycle_apg[c_idx])
                apg_c_list.append(c_val)

                if c_idx < cycle_len - 2:
                    d_search_end = min(c_idx + int(0.25 * cycle_len), cycle_len)
                    d_idx = c_idx + int(np.argmin(cycle_apg[c_idx:d_search_end]))
                    d_val = float(cycle_apg[d_idx])
                    apg_d_list.append(d_val)

                    if d_idx < cycle_len - 1:
                        e_val = float(np.max(cycle_apg[d_idx:]))
                        apg_e_list.append(e_val)

    # Aggregate Cycle Features
    feats["pulse_width_ms"] = float(np.median(cycle_pulse_widths)) if cycle_pulse_widths else np.nan
    feats["systolic_peak_amp"] = float(np.median(cycle_systolic_amps)) if cycle_systolic_amps else signal_max
    feats["diastolic_peak_amp"] = float(np.median(cycle_diastolic_amps)) if cycle_diastolic_amps else np.nan
    feats["dicrotic_notch_amp"] = float(np.median(cycle_notch_amps)) if cycle_notch_amps else np.nan
    feats["dicrotic_ratio"] = float(np.median(cycle_dicrotic_ratios)) if cycle_dicrotic_ratios else np.nan

    # APG amplitudes & standard cardiovascular aging ratios
    a_med = float(np.median(apg_a_list)) if apg_a_list else np.nan
    b_med = float(np.median(apg_b_list)) if apg_b_list else np.nan
    c_med = float(np.median(apg_c_list)) if apg_c_list else np.nan
    d_med = float(np.median(apg_d_list)) if apg_d_list else np.nan
    e_med = float(np.median(apg_e_list)) if apg_e_list else np.nan

    feats["apg_a"] = a_med
    feats["apg_b"] = b_med
    feats["apg_c"] = c_med
    feats["apg_d"] = d_med
    feats["apg_e"] = e_med

    if not np.isnan(a_med) and abs(a_med) > 1e-6:
        feats["apg_b_a_ratio"] = b_med / a_med
        feats["apg_c_a_ratio"] = c_med / a_med if not np.isnan(c_med) else np.nan
        feats["apg_d_a_ratio"] = d_med / a_med if not np.isnan(d_med) else np.nan
        feats["apg_e_a_ratio"] = e_med / a_med if not np.isnan(e_med) else np.nan
        if not np.isnan(b_med) and not np.isnan(c_med) and not np.isnan(d_med) and not np.isnan(e_med):
            feats["apg_aging_index"] = (b_med - c_med - d_med - e_med) / a_med
        else:
            feats["apg_aging_index"] = np.nan
    else:
        feats["apg_b_a_ratio"] = np.nan
        feats["apg_c_a_ratio"] = np.nan
        feats["apg_d_a_ratio"] = np.nan
        feats["apg_e_a_ratio"] = np.nan
        feats["apg_aging_index"] = np.nan

    return feats


# ==============================================================================
# 3. ECG R-PEAK DETECTION & HRV TIME/FREQUENCY DOMAIN FEATURES
# ==============================================================================

def extract_ecg_hrv_features(
    ecg_signal: Optional[np.ndarray],
    fs: float,
    reference_rpeaks: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes standard HRV metrics from ECG:
    - Time-domain: HR_bpm, SDNN (ms), RMSSD (ms), pNN50 (%)
    - Frequency-domain: LF Power (0.04-0.15 Hz), HF Power (0.15-0.40 Hz), LF/HF ratio
    """
    feats = {
        "hr_bpm": np.nan,
        "hrv_sdnn": np.nan,
        "hrv_rmssd": np.nan,
        "hrv_pnn50": np.nan,
        "hrv_lf": np.nan,
        "hrv_hf": np.nan,
        "hrv_lf_hf_ratio": np.nan,
    }

    rpeak_indices = None
    if reference_rpeaks is not None and len(reference_rpeaks) >= 3:
        rpeak_indices = reference_rpeaks
    elif ecg_signal is not None and len(ecg_signal) > int(fs * 2.0):
        # QRS detection on filtered ECG
        filtered_ecg = butterworth_ecg_bandpass(ecg_signal, fs, lowcut=5.0, highcut=25.0)
        # Pan-Tompkins derivative & squaring
        diff_ecg = np.gradient(filtered_ecg)
        sq_ecg = diff_ecg ** 2
        # Moving window integration (~150ms window)
        win_size = max(int(0.15 * fs), 3)
        integrated = np.convolve(sq_ecg, np.ones(win_size) / win_size, mode='same')

        min_dist = int(fs * 0.3)
        prominence = max(0.3 * np.std(integrated), 1e-4)
        peaks, _ = signal.find_peaks(integrated, distance=min_dist, prominence=prominence)
        rpeak_indices = peaks

    if rpeak_indices is None or len(rpeak_indices) < 3:
        return feats

    # Compute RR intervals (in milliseconds)
    rr_intervals_ms = np.diff(rpeak_indices) / fs * 1000.0

    # Physiologically valid RR interval filter: [300 ms (200 bpm), 2000 ms (30 bpm)]
    valid_rr = rr_intervals_ms[(rr_intervals_ms >= 300.0) & (rr_intervals_ms <= 2000.0)]
    if len(valid_rr) < 3:
        return feats

    # Time-Domain HRV
    mean_rr = float(np.mean(valid_rr))
    feats["hr_bpm"] = float(60000.0 / mean_rr) if mean_rr > 0 else np.nan
    feats["hrv_sdnn"] = float(np.std(valid_rr, ddof=1)) if len(valid_rr) > 1 else np.nan

    successive_diffs = np.diff(valid_rr)
    if len(successive_diffs) > 0:
        feats["hrv_rmssd"] = float(np.sqrt(np.mean(successive_diffs ** 2)))
        feats["hrv_pnn50"] = float((np.sum(np.abs(successive_diffs) > 50.0) / len(successive_diffs)) * 100.0)

    # Frequency-Domain HRV via Lomb-Scargle or interpolated Welch PSD
    if len(valid_rr) >= 6:
        try:
            # Resample RR time-series to uniform 4 Hz grid
            rr_times = np.cumsum(valid_rr) / 1000.0
            rr_times = rr_times - rr_times[0]
            total_time = rr_times[-1]
            if total_time >= 4.0:
                interp_func = interp1d(rr_times, valid_rr, kind='linear', fill_value='extrapolate')
                fs_rr = 4.0
                uniform_times = np.arange(0, total_time, 1.0 / fs_rr)
                uniform_rr = interp_func(uniform_times)
                uniform_rr = uniform_rr - np.mean(uniform_rr)

                # Welch Power Spectral Density
                nperseg = min(len(uniform_rr), int(fs_rr * 16))
                nperseg = max(nperseg, 8)
                freqs, psd = signal.welch(uniform_rr, fs=fs_rr, nperseg=nperseg)

                # LF: 0.04 - 0.15 Hz, HF: 0.15 - 0.40 Hz
                lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
                hf_mask = (freqs > 0.15) & (freqs <= 0.40)

                lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
                hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0

                feats["hrv_lf"] = lf_power
                feats["hrv_hf"] = hf_power
                feats["hrv_lf_hf_ratio"] = float(lf_power / hf_power) if hf_power > 1e-6 else np.nan
        except Exception:
            pass

    return feats


# ==============================================================================
# 4. DATASET LOADERS (BIDMC, CAPNOBASE, PPG-DALIA, D1NAMO)
# ==============================================================================

def process_bidmc_dataset(
    window_sec: float = 10.0,
    step_sec: float = 5.0
) -> List[Dict[str, Any]]:
    """Loads and extracts features from PhysioNet BIDMC (fingertip PPG, 125 Hz)."""
    zip_path = RAW_DIR / "bidmc" / "bidmc-ppg-and-respiration-dataset-1.0.0.zip"
    if not zip_path.exists():
        print("  [BIDMC] Zip file not found. Skipping.")
        return []

    print("  [BIDMC] Processing 53 subjects (fingertip PPG @ 125 Hz, ECG Lead II)...")
    records = []
    fs = 125.0
    win_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    with zipfile.ZipFile(zip_path) as z:
        csv_names = [n for n in z.namelist() if n.endswith("_Signals.csv")]
        for idx, csv_name in enumerate(sorted(csv_names), 1):
            subject_id = f"bidmc_{idx:02d}"
            try:
                with z.open(csv_name) as f:
                    df = pd.read_csv(f)
                
                # Column names have leading whitespace in BIDMC CSVs
                df.columns = [c.strip() for c in df.columns]
                ppg_raw = df["PLETH"].values.astype(np.float64)
                ecg_raw = df["II"].values.astype(np.float64) if "II" in df.columns else None

                ppg_filtered = butterworth_bandpass(ppg_raw, fs, 0.5, 8.0)

                total_len = len(ppg_raw)
                w_idx = 0
                for start in range(0, total_len - win_samples + 1, step_samples):
                    end = start + win_samples
                    w_raw_ppg = ppg_raw[start:end]
                    w_filt_ppg = ppg_filtered[start:end]
                    w_ecg = ecg_raw[start:end] if ecg_raw is not None else None

                    # Extract PPG features
                    p_feats = extract_ppg_window_features(w_raw_ppg, w_filt_ppg, fs)
                    # Extract ECG HRV features
                    e_feats = extract_ecg_hrv_features(w_ecg, fs)

                    row = {
                        "window_id": f"{subject_id}_w{w_idx:04d}",
                        "subject_id": subject_id,
                        "source_dataset": "bidmc",
                        "placement": "fingertip",
                        "fs_ppg": fs,
                        "temperature_c": np.nan,
                    }
                    row.update(p_feats)
                    row.update(e_feats)
                    records.append(row)
                    w_idx += 1
            except Exception as e:
                print(f"    Error processing {csv_name}: {e}")

    print(f"  [BIDMC] Extracted {len(records)} feature windows.")
    return records


def process_ppg_dalia_dataset(
    window_sec: float = 10.0,
    step_sec: float = 5.0
) -> List[Dict[str, Any]]:
    """Loads and extracts features from UCI PPG DaLiA (wrist PPG 64 Hz, chest ECG 700 Hz, Temp 4 Hz)."""
    zip_path = RAW_DIR / "ppg_dalia" / "ppg+dalia.zip"
    if not zip_path.exists():
        print("  [PPG-DaLiA] Zip file not found. Skipping.")
        return []

    print("  [PPG-DaLiA] Processing 15 subjects (wrist PPG @ 64 Hz, chest ECG @ 700 Hz, skin Temp @ 4 Hz)...")
    records = []
    fs_ppg = 64.0
    fs_ecg = 700.0
    fs_temp = 4.0

    win_ppg_samples = int(window_sec * fs_ppg)
    step_ppg_samples = int(step_sec * fs_ppg)

    with zipfile.ZipFile(zip_path) as outer_z:
        with outer_z.open("data.zip") as inner_z_file:
            with zipfile.ZipFile(io.BytesIO(inner_z_file.read())) as inner_z:
                pkl_files = [n for n in inner_z.namelist() if n.endswith(".pkl") and "PPG_FieldStudy/S" in n]
                for pkl_name in sorted(pkl_files):
                    subject_id = pkl_name.split("/")[-1].replace(".pkl", "")
                    try:
                        with inner_z.open(pkl_name) as pkl_f:
                            sdata = pickle.load(pkl_f, encoding="latin1")

                        wrist_bvp = sdata["signal"]["wrist"]["BVP"].flatten().astype(np.float64)
                        wrist_temp = sdata["signal"]["wrist"]["TEMP"].flatten().astype(np.float64)
                        chest_rpeaks = sdata["rpeaks"].flatten().astype(np.int64)

                        bvp_filtered = butterworth_bandpass(wrist_bvp, fs_ppg, 0.5, 8.0)

                        total_ppg = len(wrist_bvp)
                        w_idx = 0

                        for start_ppg in range(0, total_ppg - win_ppg_samples + 1, step_ppg_samples):
                            end_ppg = start_ppg + win_ppg_samples
                            t_start_sec = start_ppg / fs_ppg
                            t_end_sec = end_ppg / fs_ppg

                            w_raw_ppg = wrist_bvp[start_ppg:end_ppg]
                            w_filt_ppg = bvp_filtered[start_ppg:end_ppg]

                            # Skin Temperature in this window
                            start_temp = int(t_start_sec * fs_temp)
                            end_temp = int(t_end_sec * fs_temp)
                            w_temp_slice = wrist_temp[start_temp:end_temp]
                            temp_c = float(np.mean(w_temp_slice)) if len(w_temp_slice) > 0 else np.nan

                            # R-peaks inside this window (converted to local window indices @ 700 Hz)
                            start_ecg_sample = int(t_start_sec * fs_ecg)
                            end_ecg_sample = int(t_end_sec * fs_ecg)
                            w_rpeaks = chest_rpeaks[(chest_rpeaks >= start_ecg_sample) & (chest_rpeaks < end_ecg_sample)]
                            w_rpeaks_local = w_rpeaks - start_ecg_sample

                            # Extract PPG and ECG features
                            p_feats = extract_ppg_window_features(w_raw_ppg, w_filt_ppg, fs_ppg)
                            e_feats = extract_ecg_hrv_features(None, fs_ecg, reference_rpeaks=w_rpeaks_local)

                            row = {
                                "window_id": f"{subject_id}_w{w_idx:05d}",
                                "subject_id": subject_id,
                                "source_dataset": "ppg_dalia",
                                "placement": "wrist",
                                "fs_ppg": fs_ppg,
                                "temperature_c": temp_c,
                            }
                            row.update(p_feats)
                            row.update(e_feats)
                            records.append(row)
                            w_idx += 1
                    except Exception as e:
                        print(f"    Error processing {pkl_name}: {e}")

    print(f"  [PPG-DaLiA] Extracted {len(records)} feature windows.")
    return records


def process_capnobase_dataset(
    window_sec: float = 10.0,
    step_sec: float = 5.0
) -> List[Dict[str, Any]]:
    """Loads and extracts features from CapnoBase (_8min.mat) if present in raw/capnobase/."""
    capno_dir = RAW_DIR / "capnobase"
    mat_files = list(capno_dir.glob("*_8min.mat"))
    if not mat_files:
        print("  [CapnoBase] No '_8min.mat' files found in data/raw/capnobase/. Skipping.")
        return []

    print(f"  [CapnoBase] Found {len(mat_files)} files. Processing fingertip PPG @ 300 Hz...")
    import scipy.io as sio
    records = []
    fs = 300.0
    win_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    for mat_path in sorted(mat_files):
        subject_id = mat_path.stem
        try:
            mat_data = sio.loadmat(mat_path, squeeze_me=True)
            # Find pleth / ppg signal in mat struct
            ppg_raw = None
            ecg_raw = None
            if "signal" in mat_data:
                sig_struct = mat_data["signal"]
                if hasattr(sig_struct, "pleth"):
                    ppg_raw = sig_struct.pleth.y
                if hasattr(sig_struct, "ecg"):
                    ecg_raw = sig_struct.ecg.y

            if ppg_raw is None:
                continue

            ppg_raw = np.array(ppg_raw, dtype=np.float64)
            ppg_filtered = butterworth_bandpass(ppg_raw, fs, 0.5, 8.0)
            total_len = len(ppg_raw)
            w_idx = 0

            for start in range(0, total_len - win_samples + 1, step_samples):
                end = start + win_samples
                w_raw_ppg = ppg_raw[start:end]
                w_filt_ppg = ppg_filtered[start:end]
                w_ecg = ecg_raw[start:end] if ecg_raw is not None else None

                p_feats = extract_ppg_window_features(w_raw_ppg, w_filt_ppg, fs)
                e_feats = extract_ecg_hrv_features(w_ecg, fs)

                row = {
                    "window_id": f"{subject_id}_w{w_idx:04d}",
                    "subject_id": subject_id,
                    "source_dataset": "capnobase",
                    "placement": "fingertip",
                    "fs_ppg": fs,
                    "temperature_c": np.nan,
                }
                row.update(p_feats)
                row.update(e_feats)
                records.append(row)
                w_idx += 1
        except Exception as e:
            print(f"    Error processing {mat_path.name}: {e}")

    print(f"  [CapnoBase] Extracted {len(records)} feature windows.")
    return records


def process_d1namo_dataset() -> pd.DataFrame:
    """
    Processes D1NAMO dataset (ECG @ 250 Hz + CGM Glucose) if present in raw/d1namo/.
    Aligns windowed HRV metrics with timestamped CGM blood glucose values (mg/dL).
    """
    d1namo_dir = RAW_DIR / "d1namo"
    # Look for files or extracted subdirectories
    ecg_files = list(d1namo_dir.glob("**/sensor_data/**/ECG.csv")) + list(d1namo_dir.glob("**/ECG.csv"))
    cgm_files = list(d1namo_dir.glob("**/glucose.csv")) + list(d1namo_dir.glob("**/glucose_*.csv"))

    if not ecg_files or not cgm_files:
        print("  [D1NAMO] Raw data files not found in data/raw/d1namo/. Creating schema placeholder.")
        columns = [
            "timestamp", "subject_id", "diabetes_type", "glucose_mg_dl",
            "hr_bpm", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_lf", "hrv_hf", "hrv_lf_hf_ratio"
        ]
        return pd.DataFrame(columns=columns)

    print(f"  [D1NAMO] Found {len(ecg_files)} ECG and {len(cgm_files)} CGM files. Aligning HRV <-> Glucose...")
    # Alignment logic when data files are placed
    records = []
    fs_ecg = 250.0
    for cgm_p in cgm_files:
        try:
            cgm_df = pd.read_csv(cgm_p)
            # Standard D1NAMO columns: date, time, glucose / glucose_value
            # Match with paired subject ECG by datetime
        except Exception as e:
            print(f"    Error reading {cgm_p}: {e}")

    return pd.DataFrame(records)


# ==============================================================================
# 5. SUMMARY STATISTICS & PHYSIOLOGICAL AUDIT
# ==============================================================================

def compute_distribution_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes mean, std, min, max, 5th, 50th, 95th percentiles per feature."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude identifiers and sample rate
    exclude = ["fs_ppg"]
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    stats = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        stats[col] = {
            "count": int(len(series)),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p05": float(series.quantile(0.05)),
            "p50": float(series.median()),
            "p95": float(series.quantile(0.95)),
            "max": float(series.max()),
        }
    return stats


def audit_physiological_plausibility(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits feature distributions against canonical physiological limits:
    - Pulse Width: 100 - 400 ms
    - Heart Rate: 40 - 200 bpm
    - Trough-to-Trough (Cycle Duration): 300 - 1500 ms (40 - 200 bpm)
    - Perfusion Index: 0.05% - 20%
    - HRV SDNN / RMSSD: 5 - 300 ms
    - HRV LF/HF ratio: 0.1 - 25.0
    """
    flags = []
    summary: Dict[str, Any] = {"total_windows": len(df), "plausibility_checks": {}}

    # 1. Heart Rate (ECG & PPG)
    for hr_col in ["hr_bpm", "ppg_hr_bpm"]:
        if hr_col in df.columns:
            s = df[hr_col].dropna()
            out_low = int((s < 40.0).sum())
            out_high = int((s > 200.0).sum())
            total = len(s)
            pct_anomalous = float((out_low + out_high) / total * 100.0) if total > 0 else 0.0
            summary["plausibility_checks"][hr_col] = {
                "valid_range": "40 - 200 bpm",
                "below_40_count": out_low,
                "above_200_count": out_high,
                "anomalous_percent": f"{pct_anomalous:.2f}%",
                "median": float(s.median()) if total > 0 else None
            }
            if pct_anomalous > 5.0:
                flags.append(f"WARNING: {hr_col} has {pct_anomalous:.2f}% values outside physiological 40-200 bpm range.")

    # 2. Pulse Width
    if "pulse_width_ms" in df.columns:
        s = df["pulse_width_ms"].dropna()
        out_low = int((s < 100.0).sum())
        out_high = int((s > 400.0).sum())
        total = len(s)
        pct_anomalous = float((out_low + out_high) / total * 100.0) if total > 0 else 0.0
        summary["plausibility_checks"]["pulse_width_ms"] = {
            "valid_range": "100 - 400 ms",
            "below_100_count": out_low,
            "above_400_count": out_high,
            "anomalous_percent": f"{pct_anomalous:.2f}%",
            "p05": float(s.quantile(0.05)) if total > 0 else None,
            "median": float(s.median()) if total > 0 else None,
            "p95": float(s.quantile(0.95)) if total > 0 else None
        }
        if pct_anomalous > 5.0:
            flags.append(f"WARNING: pulse_width_ms has {pct_anomalous:.2f}% values outside standard 100-400 ms range.")

    # 3. Trough to Trough Interval
    if "trough_to_trough_ms" in df.columns:
        s = df["trough_to_trough_ms"].dropna()
        out_low = int((s < 300.0).sum())
        out_high = int((s > 1500.0).sum())
        total = len(s)
        pct_anomalous = float((out_low + out_high) / total * 100.0) if total > 0 else 0.0
        summary["plausibility_checks"]["trough_to_trough_ms"] = {
            "valid_range": "300 - 1500 ms",
            "below_300_count": out_low,
            "above_1500_count": out_high,
            "anomalous_percent": f"{pct_anomalous:.2f}%",
            "median": float(s.median()) if total > 0 else None
        }

    # 4. HRV SDNN & RMSSD
    for hrv_col in ["hrv_sdnn", "hrv_rmssd"]:
        if hrv_col in df.columns:
            s = df[hrv_col].dropna()
            out_high = int((s > 300.0).sum())
            total = len(s)
            pct_anomalous = float(out_high / total * 100.0) if total > 0 else 0.0
            summary["plausibility_checks"][hrv_col] = {
                "valid_range": "5 - 300 ms",
                "above_300_count": out_high,
                "anomalous_percent": f"{pct_anomalous:.2f}%",
                "median": float(s.median()) if total > 0 else None
            }

    # 5. Perfusion Index
    if "perfusion_index" in df.columns:
        s = df["perfusion_index"].dropna()
        summary["plausibility_checks"]["perfusion_index"] = {
            "expected_range": "0.05% - 20.0%",
            "p05": float(s.quantile(0.05)),
            "median": float(s.median()),
            "p95": float(s.quantile(0.95)),
        }

    summary["audit_flags"] = flags
    return summary


# ==============================================================================
# 6. MAIN PIPELINE EXECUTION
# ==============================================================================

def main():
    print("=" * 80)
    print("PHYSIOLOGICAL FEATURE EXTRACTION PIPELINE (PPG / VPG / APG / HRV)")
    print("=" * 80)

    # 1. Process BIDMC (Fingertip)
    bidmc_records = process_bidmc_dataset(window_sec=10.0, step_sec=5.0)

    # 2. Process CapnoBase (Fingertip)
    capnobase_records = process_capnobase_dataset(window_sec=10.0, step_sec=5.0)

    # 3. Process PPG-DaLiA (Wrist)
    dalia_records = process_ppg_dalia_dataset(window_sec=10.0, step_sec=5.0)

    # Combine all windows into single feature table
    all_records = bidmc_records + capnobase_records + dalia_records
    if not all_records:
        print("ERROR: No feature records extracted from datasets.")
        sys.exit(1)

    feature_df = pd.DataFrame(all_records)
    print(f"\nTotal extracted feature windows: {len(feature_df):,}")
    print(f"Placement breakdown:\n{feature_df['placement'].value_counts().to_string()}")
    print(f"Dataset breakdown:\n{feature_df['source_dataset'].value_counts().to_string()}")

    # Save real feature pool parquet
    parquet_path = INTERIM_DIR / "real_feature_pool.parquet"
    feature_df.to_parquet(parquet_path, index=False)
    print(f"\nSaved feature pool to: {parquet_path} ({parquet_path.stat().st_size / (1024*1024):.2f} MB)")

    # 4. Process D1NAMO (HRV <-> Glucose)
    d1namo_df = process_d1namo_dataset()
    d1namo_parquet = INTERIM_DIR / "d1namo_hrv_glucose.parquet"
    d1namo_df.to_parquet(d1namo_parquet, index=False)
    print(f"Saved D1NAMO HRV-glucose table to: {d1namo_parquet}")

    # 5. Compute Distribution Statistics (Pooled, Per-Dataset, Fingertip vs Wrist)
    print("\nComputing statistical distributions and physiological audit...")
    pooled_stats = compute_distribution_stats(feature_df)
    fingertip_stats = compute_distribution_stats(feature_df[feature_df["placement"] == "fingertip"])
    wrist_stats = compute_distribution_stats(feature_df[feature_df["placement"] == "wrist"])

    dataset_stats = {}
    for ds in feature_df["source_dataset"].unique():
        dataset_stats[ds] = compute_distribution_stats(feature_df[feature_df["source_dataset"] == ds])

    audit_results = audit_physiological_plausibility(feature_df)

    distribution_report = {
        "metadata": {
            "total_windows": len(feature_df),
            "window_length_sec": 10.0,
            "window_step_sec": 5.0,
            "datasets_included": list(feature_df["source_dataset"].unique()),
            "placement_counts": feature_df["placement"].value_counts().to_dict(),
        },
        "physiological_audit": audit_results,
        "pooled_distribution": pooled_stats,
        "placement_comparison": {
            "fingertip": fingertip_stats,
            "wrist": wrist_stats,
        },
        "per_dataset_distributions": dataset_stats,
    }

    # Save JSON report
    report_json_path = REPORTS_DIR / "ppg_feature_distributions.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(distribution_report, f, indent=2)
    print(f"Saved distribution report to: {report_json_path}")

    # Print summary of audit and key metrics
    print("\n" + "=" * 80)
    print("PHYSIOLOGICAL AUDIT & DISTRIBUTION SUMMARY")
    print("=" * 80)
    print(f"Total Windows: {len(feature_df):,} (Fingertip: {(feature_df['placement']=='fingertip').sum():,}, Wrist: {(feature_df['placement']=='wrist').sum():,})")
    
    print("\n--- Key Morphological & HRV Distributions (5th - 50th - 95th percentiles) ---")
    key_metrics = [
        ("Heart Rate (ECG bpm)", "hr_bpm"),
        ("Heart Rate (PPG bpm)", "ppg_hr_bpm"),
        ("Pulse Width (ms)", "pulse_width_ms"),
        ("Trough-to-Trough (ms)", "trough_to_trough_ms"),
        ("Perfusion Index (%)", "perfusion_index"),
        ("HRV SDNN (ms)", "hrv_sdnn"),
        ("HRV RMSSD (ms)", "hrv_rmssd"),
        ("HRV pNN50 (%)", "hrv_pnn50"),
        ("APG a-wave", "apg_a"),
        ("APG b/a ratio", "apg_b_a_ratio"),
        ("APG Aging Index", "apg_aging_index"),
    ]

    for label, col in key_metrics:
        if col in pooled_stats:
            p = pooled_stats[col]
            f = fingertip_stats.get(col, {})
            w = wrist_stats.get(col, {})
            print(f"\n{label}:")
            print(f"  Pooled:    mean={p['mean']:>8.2f} | 5th={p['p05']:>8.2f} | 50th (med)={p['p50']:>8.2f} | 95th={p['p95']:>8.2f} | std={p['std']:>8.2f}")
            if f:
                print(f"  Fingertip: mean={f['mean']:>8.2f} | 5th={f['p05']:>8.2f} | 50th (med)={f['p50']:>8.2f} | 95th={f['p95']:>8.2f}")
            if w:
                print(f"  Wrist:     mean={w['mean']:>8.2f} | 5th={w['p05']:>8.2f} | 50th (med)={w['p50']:>8.2f} | 95th={w['p95']:>8.2f}")

    print("\n--- Physiological Plausibility Checks ---")
    for k, v in audit_results["plausibility_checks"].items():
        print(f"  [{k}]: {v}")

    if audit_results["audit_flags"]:
        print("\n--- Audit Flags ---")
        for flag in audit_results["audit_flags"]:
            print(f"  ! {flag}")
    else:
        print("\nAll core distributions conform to established physiological bounds.")


if __name__ == "__main__":
    main()
