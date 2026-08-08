import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from src.encryption import encrypt_data, decrypt_data

# -------------------------------------------------
# Project Paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "autoencoder.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.pkl")

# -------------------------------------------------
# Load Model, Scaler & Threshold
# -------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

threshold = joblib.load(THRESHOLD_PATH)

# -------------------------------------------------
# Feature Names
# -------------------------------------------------

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# -------------------------------------------------
# Prediction Function
# -------------------------------------------------

def predict_anomaly(
    air_temp,
    process_temp,
    rotational_speed,
    torque,
    tool_wear
):

    # ---------------------------------------------
    # Convert Sensor Values to String
    # ---------------------------------------------

    sensor_string = (
        f"{air_temp},"
        f"{process_temp},"
        f"{rotational_speed},"
        f"{torque},"
        f"{tool_wear}"
    )

    # ---------------------------------------------
    # AES Encryption
    # ---------------------------------------------

    encrypted_data, iv = encrypt_data(sensor_string)

    # ---------------------------------------------
    # AES Decryption
    # ---------------------------------------------

    decrypted_data = decrypt_data(encrypted_data, iv)

    # ---------------------------------------------
    # Convert Back to Float Values
    # ---------------------------------------------

    values = list(
        map(
            float,
            decrypted_data.split(",")
        )
    )

    # ---------------------------------------------
    # Create DataFrame
    # ---------------------------------------------

    sample = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    # ---------------------------------------------
    # Scale Features
    # ---------------------------------------------

    sample_scaled = scaler.transform(sample)

    sample_scaled = sample_scaled.astype(np.float32)

    # ---------------------------------------------
    # Direct Model Inference
    # ---------------------------------------------

    reconstructed = model(
        sample_scaled,
        training=False
    ).numpy()

    # ---------------------------------------------
    # Reconstruction Error
    # ---------------------------------------------

    mse = np.mean(
        np.square(
            sample_scaled - reconstructed
        )
    )

    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------

    if mse > threshold:
        status = "Anomaly"
    else:
        status = "Normal"

    # ---------------------------------------------
    # Return Results
    # ---------------------------------------------

    return (
        status,
        mse,
        encrypted_data,
        iv,
        decrypted_data,
        threshold
    )