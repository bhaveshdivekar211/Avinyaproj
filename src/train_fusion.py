import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# === FORCE GPU ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # change to your GPU index if needed
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"✅ FORCED GPU: {gpus[0]}")
else:
    print("❌ No GPU found. Running on CPU.")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X_EDA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'X_eda.npy')
X_ECG_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'X_ecg.npy')
Y_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'y_train.npy')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'fusion_model.h5')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'models', 'checkpoint.weights.h5')

def train_fusion():
    if not (os.path.exists(X_EDA_PATH) and os.path.exists(X_ECG_PATH)):
        print("ERROR: Processed data not found. Run data_prep.py first!")
        return

    X_eda = np.load(X_EDA_PATH)
    X_ecg = np.load(X_ECG_PATH)
    y = np.load(Y_PATH)
    
    print(f"Data Loaded. Shapes: EDA {X_eda.shape}, ECG {X_ecg.shape}")

    X_eda_train, X_eda_test, X_ecg_train, X_ecg_test, y_train, y_test = train_test_split(
        X_eda, X_ecg, y, test_size=0.2, random_state=42
    )

    eda_input = Input(shape=(X_eda.shape[1], 1))
    eda_lstm = LSTM(64, activation='tanh')(eda_input)
    eda_drop = Dropout(0.2)(eda_lstm)
    
    ecg_input = Input(shape=(X_ecg.shape[1], 1))
    ecg_lstm = LSTM(64, activation='tanh')(ecg_input)
    ecg_drop = Dropout(0.2)(ecg_lstm)
    
    fused = Concatenate()([eda_drop, ecg_drop])
    dense = Dense(32, activation='relu')(fused)
    output = Dense(1, activation='sigmoid')(dense)
    
    model = Model(inputs=[eda_input, ecg_input], outputs=output)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    if os.path.exists(CHECKPOINT_PATH):
        model.load_weights(CHECKPOINT_PATH)
        print("✅ Resumed from checkpoint")
    
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    
    print("Starting Training...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, monitor='val_loss',
                                 mode='min', verbose=1, save_weights_only=True)
    
    model.fit([X_eda_train, X_ecg_train], y_train, epochs=50, batch_size=64,
              validation_split=0.2, callbacks=[early_stop, checkpoint])
    
    test_loss, test_acc = model.evaluate([X_eda_test, X_ecg_test], y_test)
    print(f"Test Accuracy: {test_acc:.2f}")
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model successfully saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_fusion()