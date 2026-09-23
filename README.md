# Maritime & Industrial AI Hackathon Repository

Welcome to the master repository for the **Maritime & Industrial AI Hackathon**. This repository hosts the complete datasets, evaluation suites, preparation pipelines, and documentation across all five real-world Problem Statements (PS1 through PS5).

---

## Root Directory Structure & Purpose

```text
Hackathon/
├── README.md                             # Global repository guide (this file)
├── DATASET_PREPARATION_GUIDE.md          # Step-by-step guide to download & recreate datasets
├── Dataset.txt                           # Master URL index of all original data sources
├── build_real_hackathon_datasets.py      # Master end-to-end Python dataset builder script
├── Hackathon-Problem_Statements.pdf      # Official problem statements brief document
│
├── Hackathon/                            # PARTICIPANT DISTRIBUTION PACKAGE
│   ├── PS1_public/                       # Predictive Maintenance (CSVs)
│   ├── PS2_public/                       # Oil Spill Detection (SAR PNGs + CSVs)
│   ├── PS3_public/                       # Illegal Fishing Detection (GeoTIFFs + CSVs)
│   ├── PS4_public/                       # Satellite Ship Detection (JPGs + CSVs)
│   └── PS5_public/                       # Acoustic Species Classification (WAVs + CSVs)
│
├── Hackathon.zip                         # Pre-packaged clean ZIP of the 'Hackathon/' folder (~111 MB)
│
├── Hackathon - Internal Reference/       # ORGANIZER EVALUATION & REFERENCE SUITE
│   ├── PS1-Predictive Maintenance/       # Evaluation code, secret solutions, pipelines
│   ├── PS2-Oil Spill Detection/          # Evaluation code, secret solutions, pipelines
│   ├── PS3-Illegal Fishing Detection/    # Evaluation code, secret solutions, pipelines
│   ├── PS4-Ship Detection from Satellite/# Evaluation code, secret solutions, pipelines
│   └── PS5-Acoustic Species Classification/# Evaluation code, secret solutions, pipelines
│
└── Hackathon - Internal Reference.zip    # Pre-packaged ZIP of the 'Hackathon - Internal Reference/' folder (~111 MB)
```

---

## Folder & File Descriptions

### 1. `Hackathon/` *(Participant Distribution Package)*
* **Purpose:** The folder shared directly with hackathon participants (e.g. via Google Drive or direct download).
* **Contents:**
  * `PS1_public/`: 8,000 train rows, 2,000 test rows (targets dropped), and `sample_submission.csv`.
  * `PS2_public/`: 500 train SAR images ($400\times 400$ PNGs), 250 unlabeled test images, `train_labels.csv`, and `sample_submission.csv`.
  * `PS3_public/`: 50 multi-band SAR train scenes (GeoTIFFs), 20 unlabeled test scenes, `train_labels.csv`, and `sample_submission.csv`.
  * `PS4_public/`: 400 train optical satellite JPGs, 100 unlabeled test JPGs, `train_labels.csv`, and `sample_submission.csv`.
  * `PS5_public/`: 400 train hydrophone audio clips (5s mono WAVs @ 16 kHz), 100 unlabeled test clips, `train_labels.csv`, and `sample_submission.csv`.
* **Security:** **Zero target leaks.** All test targets and secret solutions have been strictly removed.

### 2. `Hackathon.zip` *(Participant Download Archive)*
* **Purpose:** The compressed zip archive of the `Hackathon/` directory.
* **Size:** **~111 MB** (compressed using standard ZIP DEFLATE).
* **Usage:** Ready for direct distribution to participants so they can download the entire hackathon package in under a minute.

### 3. `Hackathon - Internal Reference/` *(Organizer Suite - Keep Private)*
* **Purpose:** The internal reference and evaluation backend reserved for hackathon organizers and the automated grading system.
* **Contents:**
  * **Secret Solutions (`PS{n}_secret/solution.csv`):** Private ground truth labels for the test sets.
  * **Evaluation Scripts (`evaluate_PS{n}.py`):** Standalone scoring scripts implementing the official competition metrics.
  * **Per-Track Preparation Scripts (`prepare_PS{n}.py`):** Self-contained scripts to validate and split each problem statement.
  * **Public Mirrors (`PS{n}_public/`):** Local mirrors of the public distribution folders for local testing and validation.

### 4. `Hackathon - Internal Reference.zip` *(Organizer Archive Backup)*
* **Purpose:** The compressed zip archive of the `Hackathon - Internal Reference/` directory.
* **Size:** **~111 MB**.
* **Usage:** Serves as a portable, self-contained backup of all scoring engines, solutions, and internal pipelines.

### 5. `DATASET_PREPARATION_GUIDE.md`
* **Purpose:** A complete, step-by-step reproducible guide documenting:
  * Kaggle API credentials & CLI download commands.
  * Exact data transformation pipelines and formulas.
  * How to run evaluations and verify baseline scores.
  * Cleanup and archive packaging instructions.

### 6. `build_real_hackathon_datasets.py`
* **Purpose:** The master executable ETL script that automatically converts raw downloads into both `Hackathon/` and `Hackathon - Internal Reference/` in a single run.

### 7. `Dataset.txt`
* **Purpose:** Master index of external links, research papers, and Kaggle download repositories for all datasets.

### 8. `Hackathon-Problem_Statements.pdf`
* **Purpose:** The original competition brief defining problem statements, rules, submission formats, and evaluation metrics.

---

## Problem Statements Overview

| PS # | Problem Statement Name | Modality & Data Source | Public Dataset Size | Evaluation Metric & Formula |
| :---: | :--- | :--- | :--- | :--- |
| **PS1** | **Predictive Maintenance** | NASA C-MAPSS Turbofan Telemetry | 8k train / 2k test rows | $\text{Score} = \text{AUC} - \frac{\text{RMSE}}{1000}$ |
| **PS2** | **Oil Spill Detection** | ESA Sentinel-1 C-band SAR Radar | 500 train / 250 test PNGs | $\text{Score} = \text{LogLoss} + \frac{\text{RMSE}}{10}$ |
| **PS3** | **Illegal Fishing Detection** | SSDD / xView3 SAR Radar Scenes | 50 train / 20 test scenes | xView3 Maritime Ranking ($\text{MR}$) |
| **PS4** | **Satellite Ship Detection** | PlanetScope Optical Satellite Imagery | 400 train / 100 test JPGs | $\text{Score} = \text{AUC} - \frac{\text{MAE}}{10}$ |
| **PS5** | **Acoustic Species Classification** | NOAA SanctSound Hydrophone Audio | 400 train / 100 test WAVs | Multi-label $\text{mAP}$ |

---

## Scoring Submissions (Organizers)

To score a participant's submission file against the secret solution, run the respective evaluation script from the repository root:

```powershell
# Score PS1 Submission
python "Hackathon - Internal Reference\PS1-Predictive Maintenance\evaluate_PS1.py" "<submission.csv>" "Hackathon - Internal Reference\PS1-Predictive Maintenance\PS1_secret\solution.csv"

# Score PS2 Submission
python "Hackathon - Internal Reference\PS2-Oil Spill Detection\evaluate_PS2.py" "<submission.csv>" "Hackathon - Internal Reference\PS2-Oil Spill Detection\PS2_secret\solution.csv"

# Score PS3 Submission
python "Hackathon - Internal Reference\PS3-Illegal Fishing Detection\evaluate_PS3.py" "<submission.csv>" "Hackathon - Internal Reference\PS3-Illegal Fishing Detection\PS3_secret\solution.csv"

# Score PS4 Submission
python "Hackathon - Internal Reference\PS4-Ship Detection from Satellite Imagery\evaluate_PS4.py" "<submission.csv>" "Hackathon - Internal Reference\PS4-Ship Detection from Satellite Imagery\PS4_secret\solution.csv"

# Score PS5 Submission
python "Hackathon - Internal Reference\PS5-Acoustic Species Classification\evaluate_PS5.py" "<submission.csv>" "Hackathon - Internal Reference\PS5-Acoustic Species Classification\PS5_secret\solution.csv"
```

Each script outputs a standardized JSON payload with validation status, sub-metrics, and the `final_score`.

---

## Git LFS Information

The large archive files (`Hackathon.zip` and `Hackathon - Internal Reference.zip`) are tracked using **Git LFS** (`.gitattributes`). When cloning this repository, ensure Git LFS is installed:

```bash
git lfs install
git clone https://github.com/kshitij-ai-wgl/Hackathon.git
git lfs pull
```
