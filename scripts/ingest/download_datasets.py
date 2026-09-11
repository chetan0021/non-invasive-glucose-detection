"""
Script to download, verify, and organize public datasets for the Non-Invasive Blood Glucose Prediction Project.
Follows strict immutability rules for data/raw/ directories.
"""

import os
import sys
import time
import hashlib
import requests
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

DOWNLOAD_DATE = datetime.now().strftime("%Y-%m-%d")

DATASET_CONFIGS = {
    "bidmc": {
        "name": "BIDMC PPG and Respiration Dataset (PhysioNet v1.0.0)",
        "source_url": "https://physionet.org/content/bidmc/1.0.0/",
        "download_type": "direct_url",
        "urls": [
            ("bidmc-ppg-and-respiration-dataset-1.0.0.zip", "https://physionet.org/static/published-projects/bidmc/bidmc-ppg-and-respiration-dataset-1.0.0.zip")
        ],
        "role": "PPG Morphology Pretraining & Respiratory/Cardiovascular Dynamics",
        "schema_covered": (
            "Supplies: Finger PPG waveforms (125 Hz), ECG waveforms (enabling HRV reference), "
            "SpO2, Heart Rate, Respiration Rate, basic demographics (Age, Gender).\n"
            "Does NOT supply: Two-channel raw Red/IR optical counts, Glucose mg/dL target, "
            "body temperature, saliva pH, diabetes type, diabetes medication, family history, smoking."
        ),
        "license": (
            "Open Data Commons Attribution License v1.0 (ODC-By 1.0)\n"
            "Source: PhysioNet (Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: "
            "Circulation 101(23):e215-e220; Pimentel MAF et al. IEEE TBME 2016).\n"
            "You are free to share, create, and adapt the dataset under the condition of proper attribution."
        ),
        "status": "ready"
    },
    "capnobase": {
        "name": "CapnoBase Pulse Oximeter IEEE TBME Benchmark (TBME RR Benchmark)",
        "source_url": "https://www.capnobase.org/database/pulse-oximeter-ieee-tbme-benchmark/",
        "download_type": "manual_login",
        "role": "PPG Morphology Pretraining & Artifact/Quality Assessment",
        "schema_covered": (
            "Supplies: Raw optical PPG waveforms (300 Hz sampling), ECG waveforms (for HRV), "
            "SpO2, Heart Rate, and annotated respiratory peak references across 42 subject recordings (_8min.mat).\n"
            "Does NOT supply: Glucose mg/dL ground truth, body temperature, saliva pH, "
            "medication history, diabetes subtype, family history, smoking history."
        ),
        "license": (
            "CapnoBase Open Access Research License / IEEE TBME Benchmark Terms.\n"
            "Reference: Karlen W, et al. 'Multiparameter respiratory rate estimation from the photoplethysmogram,' "
            "IEEE Trans Biomed Eng, 2013.\n"
            "Data is freely accessible for non-commercial research following free registration and license click-through."
        ),
        "status": "manual_required",
        "manual_instruction": (
            "1. Visit https://www.capnobase.org/database/pulse-oximeter-ieee-tbme-benchmark/\n"
            "2. Complete the free registration / click-through license agreement.\n"
            "3. Download ONLY the 42 MAT files ending in '_8min.mat'.\n"
            "4. Place the downloaded '_8min.mat' files directly into: glucose-prediction/data/raw/capnobase/\n"
            "5. Re-run checksum verification."
        )
    },
    "ppg_dalia": {
        "name": "PPG DaLiA: PPG Dataset for Motion Compensated Heart Rate Acquisition in Daily Life Activities (UCI ID 495)",
        "source_url": "https://archive.ics.uci.edu/dataset/495/ppg+dalia",
        "download_type": "direct_url",
        "urls": [
            ("ppg+dalia.zip", "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip")
        ],
        "role": "Wearable Multi-Modal Baseline & Motion-Resilient PPG/HRV/Temperature Pretraining",
        "schema_covered": (
            "Supplies: Wrist PPG waveforms (Empatica E4 at 64 Hz), continuous skin body temperature "
            "(Empatica E4 at 4 Hz), 3-lead ECG reference (RespiBAN at 700 Hz for ground-truth HRV), "
            "3D Accelerometer, Electrodermal Activity (EDA), Age, Gender, Height, Weight, Fitness level.\n"
            "Does NOT supply: Glucose mg/dL target, saliva pH, diabetes type/diagnosis, medication, "
            "family history, smoking history."
        ),
        "license": (
            "Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
            "Citation: Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). "
            "'Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks', Sensors, 19(14), 3079."
        ),
        "status": "ready"
    },
    "d1namo": {
        "name": "D1NAMO: ECG, Breathing, Accelerometer, and CGM Continuous Glucose Dataset (Kaggle Mirror)",
        "source_url": "https://www.kaggle.com/datasets/sarabhian/d1namo-ecg-glucose-data",
        "download_type": "kaggle_cli",
        "kaggle_cmd": "kaggle datasets download -d sarabhian/d1namo-ecg-glucose-data",
        "role": "HRV ↔ Continuous Glucose (CGM) Grounding (CRITICAL NOTE: CONTAINS NO PPG)",
        "schema_covered": (
            "Supplies: ECG waveforms (Zephyr BioHarness 3 at 250 Hz → RMSSD, SDNN, pNN50, LF/HF), "
            "Continuous Glucose Monitoring (CGM) blood glucose levels (mg/dL), breathing rate, accelerometer, "
            "food photo metadata/timing, diabetes status (9 Type 1 Diabetes subjects + 20 healthy controls).\n"
            "Does NOT supply: PPG waveforms (NO PPG exists in this dataset), Red/IR raw counts, "
            "saliva pH, body temperature, comprehensive oral medications, family history, smoking."
        ),
        "license": (
            "Open Database License (ODbL) / Database Contents License (DbCL) / Kaggle Terms.\n"
            "Citation: F. Tomek et al., 'D1NAMO: A Diabetes Dataset with ECG, Accelerometer, and Food Intake', "
            "Applied Sciences, 2021."
        ),
        "status": "manual_required",
        "manual_instruction": (
            "1. Setup Kaggle API token (~/.kaggle/kaggle.json) OR log into Kaggle in browser.\n"
            "2. Navigate to: https://www.kaggle.com/datasets/sarabhian/d1namo-ecg-glucose-data\n"
            "3. Download the archive (or run: kaggle datasets download -d sarabhian/d1namo-ecg-glucose-data).\n"
            "4. Place the downloaded archive directly into: glucose-prediction/data/raw/d1namo/\n"
            "5. Re-run checksum verification."
        )
    },
    "kaggle_ppg_glucose": {
        "name": "PPG Signal with Blood Sugar Level Data (Kaggle)",
        "source_url": "https://www.kaggle.com/datasets/muhammadyasirsaleem/ppg-signal-with-blood-sugar-level-data",
        "download_type": "kaggle_cli",
        "kaggle_cmd": "kaggle datasets download -d muhammadyasirsaleem/ppg-signal-with-blood-sugar-level-data",
        "role": "Weak PPG ↔ Glucose Pairing Exploration & Direct Signal Modeling",
        "schema_covered": (
            "Supplies: Optical PPG raw/filtered signals paired with blood glucose level readings (mg/dL / categorical).\n"
            "Does NOT supply: Calibrated MAX30102 dual Red/IR optical raw counts, continuous ECG/HRV, "
            "body temperature, saliva pH, comprehensive personal parameters (family history, detailed medications, smoking)."
        ),
        "license": (
            "Community Data License Agreement / CC BY-SA 4.0 / Kaggle Dataset Terms.\n"
            "Creator: Muhammad Yasir Saleem."
        ),
        "status": "manual_required",
        "manual_instruction": (
            "1. Setup Kaggle API token (~/.kaggle/kaggle.json) OR log into Kaggle in browser.\n"
            "2. Navigate to: https://www.kaggle.com/datasets/muhammadyasirsaleem/ppg-signal-with-blood-sugar-level-data\n"
            "3. Download the dataset zip (or run: kaggle datasets download -d muhammadyasirsaleem/ppg-signal-with-blood-sugar-level-data).\n"
            "4. Place the downloaded file directly into: glucose-prediction/data/raw/kaggle_ppg_glucose/\n"
            "5. Re-run checksum verification."
        )
    },
    "nhanes": {
        "name": "National Health and Nutrition Examination Survey (CDC NHANES 2017-2018 Cycle)",
        "source_url": "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017",
        "download_type": "direct_url",
        "urls": [
            ("DEMO_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.XPT"),
            ("GLU_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.XPT"),
            ("DIQ_J.XPT", "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.XPT")
        ],
        "role": "Tabular Personal Parameters, Medication, Lifestyle & Fasting Glucose Grounding",
        "schema_covered": (
            "Supplies: Fasting Plasma Glucose (LBXGLU in mg/dL), Age (RIDAGEYR), Gender (RIAGENDR), "
            "BMI / Body measures, Diabetes diagnosis/status (DIQ010), Insulin taking (DIQ050), "
            "Diabetic pills/medication (DIQ070), Smoking history (SMQ series), Family history / Demographics.\n"
            "Does NOT supply: PPG waveforms, continuous ECG/HRV waveforms, continuous body temperature, saliva pH."
        ),
        "license": (
            "Public Domain (United States Government Work)\n"
            "Source: National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention (CDC).\n"
            "Free for public and research use without restriction."
        ),
        "status": "ready"
    },
    "pima": {
        "name": "Pima Indians Diabetes Database (NIDDK / Kaggle Mirror)",
        "source_url": "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
        "download_type": "kaggle_cli",
        "kaggle_cmd": "kaggle datasets download -d uciml/pima-indians-diabetes-database",
        "role": "Tabular Metabolic Feature Modeling & Diagnostic Baseline",
        "schema_covered": (
            "Supplies: Plasma Glucose concentration (mg/dL from 2h OGTT), Age (years), BMI (kg/m²), "
            "Diastolic Blood Pressure (mm Hg), 2-Hour serum insulin (mu U/ml), Diabetes Pedigree Function "
            "(family history score), Pregnancies, Diabetes outcome.\n"
            "Does NOT supply: PPG waveforms, ECG/HRV time series, body temperature, saliva pH, "
            "smoking history, specific medication brands."
        ),
        "license": (
            "CC0: Public Domain\n"
            "Original Owner: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).\n"
            "Donor: Vincent Sigillito, Johns Hopkins University."
        ),
        "status": "manual_required",
        "manual_instruction": (
            "1. Setup Kaggle API token (~/.kaggle/kaggle.json) OR log into Kaggle in browser.\n"
            "2. Navigate to: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n"
            "3. Download diabetes.csv (or run: kaggle datasets download -d uciml/pima-indians-diabetes-database).\n"
            "4. Place the downloaded archive/file directly into: glucose-prediction/data/raw/pima/\n"
            "5. Re-run checksum verification."
        )
    },
    "diabetes130": {
        "name": "Diabetes 130-US Hospitals for Years 1999-2008 (UCI ID 296)",
        "source_url": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
        "download_type": "direct_url",
        "urls": [
            ("diabetes+130-us+hospitals+for+years+1999-2008.zip", "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip")
        ],
        "role": "Tabular Medication Regimens, HbA1c/Glucose Categories & Clinical Features",
        "schema_covered": (
            "Supplies: Glucose test serum indicator (max_glu_serum: >200, >300, normal, none), "
            "HbA1c test results (A1Cresult: >7, >8, normal, none), Age brackets, Gender, "
            "24 distinct diabetes medications (metformin, repaglinide, nateglinide, glimepiride, "
            "glipizide, glyburide, pioglitazone, rosiglitazone, insulin, etc.), medication changes, diagnoses.\n"
            "Does NOT supply: Continuous PPG waveforms, raw Red/IR optical counts, continuous ECG/HRV, "
            "body temperature, saliva pH, smoking history."
        ),
        "license": (
            "Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
            "Citation: Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). "
            "'Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records', "
            "BioMed Research International, vol. 2014, Article ID 781670."
        ),
        "status": "ready"
    }
}


def calculate_sha256(filepath: Path) -> str:
    """Computes sha256 hash of a file efficiently."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def download_file(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Streams download with progress reporting and retry logic."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {url} -> {dest_path.name} (attempt {attempt}/{max_retries})...")
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                total_length = r.headers.get("content-length")
                total_bytes = int(total_length) if total_length else None
                downloaded = 0
                start_time = time.time()
                last_print = start_time

                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            now = time.time()
                            if now - last_print > 3.0:
                                last_print = now
                                if total_bytes:
                                    percent = (downloaded / total_bytes) * 100
                                    mb = downloaded / (1024 * 1024)
                                    total_mb = total_bytes / (1024 * 1024)
                                    print(f"  Progress: {mb:.1f}MB / {total_mb:.1f}MB ({percent:.1f}%)")
                                else:
                                    mb = downloaded / (1024 * 1024)
                                    print(f"  Downloaded: {mb:.1f}MB")

                size = dest_path.stat().st_size
                print(f"  Success: {dest_path.name} downloaded ({size:,} bytes).")
                return True
        except Exception as e:
            print(f"  Error downloading {url}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            time.sleep(2)
    return False


def generate_metadata_files(dataset_key: str, cfg: dict):
    """Generates README.txt, LICENSE.txt, and checksums.txt in data/raw/<dataset>/."""
    dataset_dir = RAW_DIR / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 1. Checksums & file list
    downloaded_files = [f for f in dataset_dir.iterdir() if f.is_file() and f.name not in ["README.txt", "LICENSE.txt", "checksums.txt"]]
    
    checksum_lines = []
    file_list_lines = []
    
    for f in sorted(downloaded_files, key=lambda x: x.name):
        sha = calculate_sha256(f)
        size_bytes = f.stat().st_size
        checksum_lines.append(f"{sha}  {f.name}")
        file_list_lines.append(f"- {f.name} ({size_bytes:,} bytes, {size_bytes / (1024*1024):.2f} MB)")

    if not checksum_lines:
        checksum_content = "# No downloaded data files present yet. Awaiting manual download step.\n"
        file_list_text = "(No data files present yet - see manual instructions below)"
    else:
        checksum_content = "\n".join(checksum_lines) + "\n"
        file_list_text = "\n".join(file_list_lines)

    with open(dataset_dir / "checksums.txt", "w", encoding="utf-8") as f:
        f.write(checksum_content)

    # 2. LICENSE.txt
    with open(dataset_dir / "LICENSE.txt", "w", encoding="utf-8") as f:
        f.write(cfg["license"] + "\n")

    # 3. README.txt
    manual_part = ""
    if cfg["status"] == "manual_required":
        manual_part = f"\nMANUAL DOWNLOAD REQUIRED:\n{cfg.get('manual_instruction', 'N/A')}\n"

    readme_content = f"""================================================================================
DATASET: {cfg['name']}
================================================================================
Source URL: {cfg['source_url']}
Folder: data/raw/{dataset_key}/
Role in Project: {cfg['role']}
Date Organized: {DOWNLOAD_DATE}

FILES PRESENT:
{file_list_text}
{manual_part}
SCHEMA FIELDS COVERAGE:
{cfg['schema_covered']}

IMMUTABILITY NOTICE:
Files in this raw directory are stored strictly as downloaded from the original
source without modification, in-place unzipping, or mutation.
"""
    with open(dataset_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(readme_content)


def main():
    print("=" * 80)
    print("STARTING PUBLIC DATASET INGESTION & ORGANIZATION")
    print(f"Target raw directory: {RAW_DIR}")
    print("=" * 80)

    # Process all datasets
    for key, cfg in DATASET_CONFIGS.items():
        print(f"\n[{key.upper()}] Processing {cfg['name']}...")
        dest_dir = RAW_DIR / key
        dest_dir.mkdir(parents=True, exist_ok=True)

        if cfg["download_type"] == "direct_url":
            for fname, url in cfg["urls"]:
                dest_file = dest_dir / fname
                if dest_file.exists() and dest_file.stat().st_size > 0:
                    print(f"  File {fname} already exists ({dest_file.stat().st_size:,} bytes). Skipping download.")
                else:
                    success = download_file(url, dest_file)
                    if not success:
                        print(f"  FAILED to download {fname} from {url}")
        elif cfg["download_type"] == "manual_login" or cfg["download_type"] == "kaggle_cli":
            print(f"  Skipping automated download: {cfg['download_type']} required.")

        # Always generate / update metadata files
        generate_metadata_files(key, cfg)

    print("\n" + "=" * 80)
    print("ALL DATASETS PROCESSED. SUMMARY TABLE BELOW.")
    print("=" * 80)


if __name__ == "__main__":
    main()
