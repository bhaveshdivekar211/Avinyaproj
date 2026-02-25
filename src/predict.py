import numpy as np
import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fusion_model.h5')

def make_prediction(new_eda, new_ecg):
    model = load_model(MODEL_PATH)
    
    # Ensure exactly 100 timesteps
    new_eda = np.array(new_eda).flatten()[-100:]
    new_ecg = np.array(new_ecg).flatten()[-100:]
    
    input_eda = new_eda.reshape(1, 100, 1)
    input_ecg = new_ecg.reshape(1, 100, 1)
    
    prediction = model.predict([input_eda, input_ecg])
    result = "Stress" if prediction[0][0] > 0.5 else "Relaxed"
    print(f"Prediction: {result} (Confidence: {prediction[0][0]:.2f})")

if __name__ == "__main__":
    sample_eda = np.random.rand(100)
    sample_ecg = np.random.rand(100)
    make_prediction(sample_eda, sample_ecg)