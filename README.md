# Sign Language Recognition System (LSTM + MediaPipe)

## Overview
This project is a real-time sign language recognition system using:
- MediaPipe for hand landmark extraction
- LSTM model for sequence classification
- Audio output for detected signs

The system detects hand gestures from the camera and plays a corresponding voice.

---

## Project Pipeline

### 1. Data Collection
- Recorded videos for 7 signs
- Each sign has multiple samples (videos)

---

### 2. Feature Extraction (`extract_all.py`)
- Extracts **both hands landmarks (21 points × 3)**
- Total features per frame = **126**
- Saves sequences as `.npy`

---

### 3. Data Preparation (`prepare_data_multi.py`)
- Padding / truncation to fixed length (30 frames)
- Data augmentation:
  - Noise
  - Shift
- Output:
  - `X_multi.npy`
  - `y_multi.npy`

---

### 4. Model Training (`train.py`)
- Model: LSTM
- Input shape: `(30, 126)`
- Includes:
  - Normalization (StandardScaler)
  - Dropout
  - Batch Normalization
- Saves:
  - `best_sign_model.h5`
  - `scaler.save`

---

### 5. Real-Time Inference (`camera_inference.py`)
- Opens webcam
- Extracts hand landmarks
- Predicts sign using trained model
- Displays text
- Plays audio for detected sign

---

## Classes (Signs)

- alsalam_kum
- kaif_halik
- kam_alsaah
- momkin_tosaedni
- sabah_alkhair
- win_alhamam
- win_altawaree

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt