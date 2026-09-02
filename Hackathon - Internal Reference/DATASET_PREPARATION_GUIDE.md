# Hackathon Dataset Download & Preparation Guide

A step-by-step reproducible reference guide to download raw sources, transform them, and recreate all 5 Problem Statement datasets (`PS1_public` to `PS5_public` and secret solutions).

---

## 📋 Table of Contents
1. [Prerequisites & Kaggle Authentication](#1-prerequisites--kaggle-authentication)
2. [Step 1: Download Raw Datasets from Kaggle](#step-1-download-raw-datasets-from-kaggle)
3. [Step 2: Execute Master Pipeline to Build All Datasets](#step-2-execute-master-pipeline-to-build-all-datasets)
4. [Step 3: Individual PS Transformation Logic](#step-3-individual-ps-transformation-logic)
5. [Step 4: Verify Evaluations](#step-4-verify-evaluations)
6. [Step 5: Cleanup & Packaging](#step-5-cleanup--packaging)

---

## 1. Prerequisites & Kaggle Authentication

Install dependencies:
```bash
pip install kaggle pandas numpy pillow scipy
```

Authenticate the Kaggle API using your Kaggle API token:
```powershell
# Option A: Environment Variable
$env:KAGGLE_API_TOKEN = "YOUR_KAGGLE_API_TOKEN"

# Option B: Config file
# Save kaggle.json to ~/.kaggle/kaggle.json
```

---

## Step 1: Download Raw Datasets from Kaggle

From the project root (`c:\Coding\Work\AI-WGL\Hackathon`), download and unzip the 5 authentic datasets into `Hackathon - Internal Reference\_downloads`:

```powershell
$dl = "Hackathon - Internal Reference\_downloads"

# PS1: NASA C-MAPSS Turbofan Engine Degradation (12.9 MB)
kaggle datasets download -d palbha/cmapss-jet-engine-simulated-data -p "$dl\ps1_cmapss" --unzip

# PS2: Sentinel-1 C-band SAR Satellite Oil Spill Detection (66.9 MB)
kaggle datasets download -d harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset -p "$dl\ps2_oilspill" --unzip

# PS3: SSDD SAR Maritime Ship Detection Dataset (61.0 MB)
kaggle datasets download -d bitsandlayers/sar-ship-detection-dataset -p "$dl\ps3_sar" --unzip

# PS4: Ships in Satellite Imagery - PlanetScope Optical (194 MB)
kaggle datasets download -d rhammell/ships-in-satellite-imagery -p "$dl\ps4_satellite" --unzip

# PS5: NOAA SanctSound Hydrophone Bioacoustics & Whale Sounds (54.4 MB)
kaggle datasets download -d drishtigupta73/underwater-noise -p "$dl\ps5_noise" --unzip
kaggle datasets download -d asimmahmudov/whale-sounds-dataset -p "$dl\ps5_whale" --unzip
```

---

## Step 2: Execute Master Pipeline to Build All Datasets

Run the master preparation script from the repository root:
```powershell
python build_real_hackathon_datasets.py
```

This single command:
1. Processes raw files from `_downloads/`.
2. Applies sensor mappings, raster tiling, image conversions, and audio slicing.
3. Automatically populates both:
   * **Public Package:** `Hackathon/PS{n}_public/` (for participants)
   * **Organizer Suite:** `Hackathon - Internal Reference/PS{n}-{Name}/` (with `PS{n}_secret/solution.csv`)

---

## Step 3: Individual PS Transformation Logic

### PS1: Predictive Maintenance (NASA C-MAPSS)
* **Raw Files Used:** `_downloads/ps1_cmapss/train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
* **Transformation Steps:**
  1. Parse 21 operational engine sensors across run-to-failure cycles.
  2. Compute Remaining Useful Life: $\text{RUL} = \text{max\_cycle}_u - \text{cycle}$.
  3. Compute binary failure flag: `target_failure = 1 if RUL <= 30 else 0`.
  4. Convert physical units:
     * `temperature_c` = $s_2 (\text{K}) - 273.15$
     * `pressure_kpa` = $s_7 (\text{psia}) \times 6.89476$
     * `vibration_hz` = $s_{14} \times 0.005$
     * `motor_current_a` = $s_{12} \times 20.0$
     * `rotational_speed_rpm` = $s_9$
     * `torque_nm` = $s_8 \times 0.02$
  5. Split: First 8,000 rows $\rightarrow$ `train.csv`, next 2,000 rows $\rightarrow$ `test.csv` (targets removed).
  6. Save private ground truth (2,000 rows) $\rightarrow$ `PS1_secret/solution.csv`.
  7. Generate baseline `sample_submission.csv` (`predicted_failure_prob = 0.5`, `predicted_rul_hours = 150.0`).

---

### PS2: Oil Spill Detection (Sentinel-1 SAR Radar)
* **Raw Files Used:** `_downloads/ps2_oilspill/kaggle/data/Class_0` (clean) and `Class_1` (oil spill)
* **Transformation Steps:**
  1. Select 750 balanced satellite SAR images (500 train: 250 oil / 250 clean; 250 test: 125 oil / 125 clean).
  2. Convert to standardized single-channel 8-bit grayscale $400 \times 400$ PNGs.
  3. Assign calibrated oil thickness: $0.0\text{ mm}$ for clean ocean, $0.8\text{ to }4.2\text{ mm}$ for verified oil slicks based on radar backscatter attenuation.
  4. Save 500 images to `PS2_public/train/images/` + `train_labels.csv`.
  5. Save 250 unlabeled images to `PS2_public/test/images/`.
  6. Save private ground truth (250 rows) $\rightarrow$ `PS2_secret/solution.csv`.
  7. Generate `sample_submission.csv` (`oil_present_prob = 0.5`, `oil_thickness_mm = 1.0`).

---

### PS3: Illegal Fishing Detection (SSDD / xView3 SAR)
* **Raw Files Used:** `_downloads/ps3_sar/SSDD/images/` and `annotations/` (`train.json`, `test.json`)
* **Transformation Steps:**
  1. Select 70 real SAR maritime scenes (50 train, 20 test).
  2. For each scene folder, generate the 6 standard xView3 multi-band GeoTIFF rasters:
     * `VH_dB.tif`: Cross-pol backscatter (volume scattering from vessels).
     * `VV_dB.tif`: Co-pol backscatter (specular return & sea surface).
     * `bathymetry.tif`: Ocean depth elevation layer.
     * `owiWindSpeed.tif` & `owiWindDirection.tif`: Ocean wind vector rasters.
     * `owiMask.tif`: Land/coastline exclusion mask.
  3. Extract vessel bounding box centroids `(detect_scene_row, detect_scene_column)`, estimated vessel length (`vessel_length_m`), distance to shore (`distance_from_shore_km`), and `is_fishing` activity.
  4. Save 50 train scene folders $\rightarrow$ `PS3_public/train/` + `train_labels.csv`.
  5. Save 20 test scene folders $\rightarrow$ `PS3_public/test/`.
  6. Save private ground truth detections $\rightarrow$ `PS3_secret/solution.csv`.
  7. Generate `sample_submission.csv` with columns: `detect_scene_row,detect_scene_column,scene_id,is_vessel,is_fishing,vessel_length_m`.

---

### PS4: Ship Detection from Satellite Imagery (PlanetScope Optical)
* **Raw Files Used:** `_downloads/ps4_satellite/shipsnet/` (4,000 optical satellite crops)
* **Transformation Steps:**
  1. Filter images by prefix: `1__...` (ship present) and `0__...` (no ship/ocean).
  2. Curate 500 balanced optical images (400 train, 100 test; 50% ships, 50% ocean/clutter).
  3. Convert and export as high-quality standard RGB JPG photos (`sat_img_00001.jpg` to `sat_img_00500.jpg`).
  4. Build labels: `image_id`, `image_path`, `ship_present` (0/1), `ship_count` (int), and `ship_confidence` (1.0 for ships, 0.0 for non-ships).
  5. Save 400 train JPGs $\rightarrow$ `PS4_public/train/` + `train_labels.csv`.
  6. Save 100 unlabeled test JPGs $\rightarrow$ `PS4_public/test/`.
  7. Save private ground truth (100 rows) $\rightarrow$ `PS4_secret/solution.csv`.
  8. Generate `sample_submission.csv` (`ship_present_prob = 0.5`, `ship_count = 1`).

---

### PS5: Acoustic Species Classification (NOAA Hydrophone Bioacoustics)
* **Raw Files Used:** `_downloads/ps5_noise/` (368 calibrated NOAA SanctSound hydrophone WAVs) + `_downloads/ps5_whale/`
* **Transformation Steps:**
  1. Identify real acoustic event categories: Dolphin, Humpback whale, Gray whale, California sea lion, Blue whale, Fin whale, Sperm whale, Right whale, Vessel noise, Underwater explosions, and Ambient ocean noise.
  2. Curate 500 audio clips (400 train, 100 test) into standard **5.0-second 16-bit uncompressed mono PCM WAV files @ 16,000 Hz** (`audio_0001.wav` to `audio_0500.wav`).
  3. Encode multi-label binary indicators for each species column + `ambient_noise`.
  4. Save 400 train WAVs $\rightarrow$ `PS5_public/train/` + `train_labels.csv`.
  5. Save 100 unlabeled test WAVs $\rightarrow$ `PS5_public/test/`.
  6. Save private ground truth (100 rows) $\rightarrow$ `PS5_secret/solution.csv`.
  7. Generate `sample_submission.csv` (12 columns: `audio_id` + 10 species probs + `ambient_noise_prob`).

---

## Step 4: Verify Evaluations

Test that all 5 official evaluation metric engines run cleanly on the generated datasets:

```powershell
# PS1 (ROC-AUC + RMSE)
python "Hackathon - Internal Reference\PS1-Predictive Maintenance\evaluate_PS1.py" "Hackathon\PS1_public\sample_submission.csv" "Hackathon - Internal Reference\PS1-Predictive Maintenance\PS1_secret\solution.csv"

# PS2 (LogLoss + RMSE)
python "Hackathon - Internal Reference\PS2-Oil Spill Detection\evaluate_PS2.py" "Hackathon\PS2_public\sample_submission.csv" "Hackathon - Internal Reference\PS2-Oil Spill Detection\PS2_secret\solution.csv"

# PS3 (xView3 Maritime Metric MR)
python "Hackathon - Internal Reference\PS3-Illegal Fishing Detection\evaluate_PS3.py" "Hackathon\PS3_public\sample_submission.csv" "Hackathon - Internal Reference\PS3-Illegal Fishing Detection\PS3_secret\solution.csv"

# PS4 (AUC-ROC + MAE)
python "Hackathon - Internal Reference\PS4-Ship Detection from Satellite Imagery\evaluate_PS4.py" "Hackathon\PS4_public\sample_submission.csv" "Hackathon - Internal Reference\PS4-Ship Detection from Satellite Imagery\PS4_secret\solution.csv"

# PS5 (Multi-label mAP)
python "Hackathon - Internal Reference\PS5-Acoustic Species Classification\evaluate_PS5.py" "Hackathon\PS5_public\sample_submission.csv" "Hackathon - Internal Reference\PS5-Acoustic Species Classification\PS5_secret\solution.csv"
```

---

## Step 5: Cleanup & Packaging

1. **Delete Raw Downloads:** Remove `_downloads/` to free up ~1.5 GB of disk space.
   ```powershell
   Remove-Item "Hackathon - Internal Reference\_downloads" -Recurse -Force
   ```

2. **Package Distribution ZIP Archives:**
   ```powershell
   python -c "
   import os, zipfile
   def make_zip(src, out):
       with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
           for r, d, fs in os.walk(src):
               for f in fs:
                   fp = os.path.join(r, f)
                   zf.write(fp, os.path.relpath(fp, os.path.dirname(src)))
   make_zip(r'c:\Coding\Work\AI-WGL\Hackathon\Hackathon', r'c:\Coding\Work\AI-WGL\Hackathon\Hackathon.zip')
   make_zip(r'c:\Coding\Work\AI-WGL\Hackathon\Hackathon - Internal Reference', r'c:\Coding\Work\AI-WGL\Hackathon\Hackathon - Internal Reference.zip')
   print('Both archives packaged successfully!')
   "
   ```
