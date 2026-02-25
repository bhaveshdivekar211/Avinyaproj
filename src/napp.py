import streamlit as st
import numpy as np
import pandas as pd
import time
import os
from collections import deque
from tensorflow.keras.models import load_model

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'fusion_model.h5')

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Biometric AI", page_icon="🧠", layout="wide")

st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <h1 style='color: #4A90E2;'>🧠 Multi-Modal Stress Detection System</h1>
        <p style='color: #888;'>Real-time physiological monitoring using Dual-LSTM Sensor Fusion</p>
    </div>
    <hr>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)

model = load_ai_model()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    if model is None:
        st.error("❌ Model not found! Train the model first.")
        st.stop()
    else:
        st.success("✅ AI Model Loaded")
    
    st.markdown("---")
    # Using a button instead of a toggle to trigger the stream
    start_stream = st.button("▶️ START LIVE FEED (Simulation)", use_container_width=True)
    st.caption("To stop the stream early, click 'Stop' in the top right corner of the screen.")

# --- CREATE UI PLACEHOLDERS ---
# We create empty boxes here, and we will inject data into them in the loop
metric_col1, metric_col2, metric_col3 = st.columns(3)
state_metric = metric_col1.empty()
ecg_metric = metric_col2.empty()
eda_metric = metric_col3.empty()

progress_container = st.empty()
st.markdown("<br>", unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("### 🫀 Live ECG (Heart Activity)")
    ecg_chart = st.empty()
with chart_col2:
    st.markdown("### 💧 Live EDA (Skin Conductance)")
    eda_chart = st.empty()

# --- THE LIVE STREAM ENGINE ---
if start_stream:
    # Setup fresh buffers
    ecg_buf = deque([0.0]*200, maxlen=200)
    eda_buf = deque([0.0]*200, maxlen=200)
    idx = 0
    
    # Run a continuous loop for 1000 frames (approx 5-10 minutes of live demo)
    for step in range(1000):
        try:
            # 1. Generate new data points
            new_ecg = np.sin(np.linspace(idx, idx + 2, 20)) + np.random.normal(0, 0.05, 20)
            new_eda = np.linspace(0, 0.5, 20) + np.random.normal(0, 0.01, 20)
            
            ecg_buf.extend(new_ecg)
            eda_buf.extend(new_eda)
            idx += 2

            # 2. Update Charts IN PLACE (No page refresh!)
            df_ecg = pd.DataFrame({'ECG (mV)': list(ecg_buf)})
            df_eda = pd.DataFrame({'EDA (μS)': list(eda_buf)})
            
            ecg_chart.line_chart(df_ecg, color="#FF4B4B", height=250)
            eda_chart.line_chart(df_eda, color="#00C4EB", height=250)

            # 3. Predict using the AI Model
            ecg_win = np.array(list(ecg_buf)[-100:]).reshape(1, 100, 1)
            eda_win = np.array(list(eda_buf)[-100:]).reshape(1, 100, 1)
            
            pred = model.predict([eda_win, ecg_win], verbose=0)[0][0]

            # 4. Update Metrics IN PLACE
            if pred > 0.5:
                state_metric.error(f"🚨 ALERT: STRESS DETECTED ({pred:.1%})")
                progress_container.progress(float(pred), text="Stress Probability (High)")
            else:
                state_metric.success(f"✅ STATUS: RELAXED ({(1-pred):.1%} Confidence)")
                progress_container.progress(float(pred), text="Stress Probability (Low)")

            ecg_metric.metric("Heart Rate", f"{int(70 + (pred * 25) + np.random.normal(0, 2))} bpm")
            eda_metric.metric("Skin Conductance", f"{np.mean(new_eda):.2f} μS")

            # 5. Control the speed of the stream
            time.sleep(0.2)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            break

else:
    # What it looks like before you click start
    state_metric.info("⏸️ SYSTEM IDLE. Click 'Start Live Feed' to begin.")
    ecg_metric.metric("Heart Rate (Proxy)", "-- bpm")
    eda_metric.metric("Skin Conductance", "-- μS")