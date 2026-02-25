# 🧠 Multi-Modal Biometric Stress Detection

## 📌 Project Overview
This project is a Deep Learning-based application that detects human stress in real-time by analyzing physiological signals. Instead of relying on a single data point, this model utilizes **Sensor Fusion**, combining heart activity (ECG) and skin conductance (EDA/Sweat) to make highly accurate predictions.

## 🗄️ The Dataset (WESAD)
The model is trained on the **WESAD (Wearable Stress and Affect Detection)** dataset. 
* **Raw Data:** Chest-worn sensor data sampled at a massive 700Hz.
* **Preprocessing:** To optimize for computational efficiency without losing signal integrity, the data was downsampled to 100Hz and Min-Max normalized.
* **Windowing:** The model processes time-series sequences in 1-second windows (100 data points per sequence) to capture the *trend* of the biological response rather than isolated moments.

## 🏗️ Neural Network Architecture
The brain of this project is a custom **Dual-Head LSTM (Long Short-Term Memory)** network.
1. **EDA Branch:** Learns the slow-rising trends of galvanic skin response.
2. **ECG Branch:** Learns the rapid, rhythmic patterns of heart rate variability.
3. **Fusion Layer:** The outputs of both LSTMs are concatenated (fused) and passed through Dense layers to output a final binary classification.
   * `0` = Relaxed / Baseline
   * `1` = Stressed

## 🚀 How to Run the Project
1. **Install Dependencies:** `pip install tensorflow pandas numpy scikit-learn streamlit matplotlib`
2. **Data Prep:** Run `python src/data_prep.py` to extract and process WESAD `.pkl` files.
3. **Train Model:** Run `python src/train_fusion.py` to train the Dual-LSTM.
4. **Launch Dashboard:** Run `streamlit run src/napp.py` to open the real-time visualization app in your browser.


--------------------------------------------------------------------------------------------------------------------------------------
folder structure
## 📁 Project Structure

Avinyaproj/
├── Data/                   # Directory for raw/processed datasets (contents ignored by Git)
│   └── .gitkeep            # Preserves the empty folder structure
├── docs/                   # Project documentation and research notes
├── models/                 # Saved AI model checkpoints
│   └── fusion_model.h5     # Trained Dual-LSTM Sensor Fusion model
├── src/                    # Main source code
│   └── napp.py             # Streamlit real-time monitoring dashboard
├── .gitignore              # Git rules blocking heavy data files (>100MB)
└── README.md               # Project overview and setup instructions

incase adding raw and processed datasets 
Data/RAW/..files.pkl
Data/Processed/..files.npy
