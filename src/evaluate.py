import os
import joblib
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# Paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "autoencoder.keras")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.pkl")
TEST_PATH = os.path.join(BASE_DIR, "models", "test_data.pkl")

# -------------------------------------------------
# Load Files
# -------------------------------------------------

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Loading threshold...")

threshold = joblib.load(THRESHOLD_PATH)

print("Loading test dataset...")

X_test = joblib.load(TEST_PATH)

print(f"Test Samples : {len(X_test)}")

# -------------------------------------------------
# Prediction
# -------------------------------------------------

print("Running prediction...")

reconstructed = model.predict(
    X_test,
    batch_size=256,
    verbose=1
)

print("Prediction completed.")

# -------------------------------------------------
# Reconstruction Error
# -------------------------------------------------

errors = np.mean(
    np.square(X_test - reconstructed),
    axis=1
)

# -------------------------------------------------
# Predictions
# -------------------------------------------------

y_pred = (errors > threshold).astype(int)

print("\nThreshold :", threshold)

print("\nMean Error :", np.mean(errors))

print("Max Error :", np.max(errors))

print("Min Error :", np.min(errors))

print("\nPredicted Normal :", np.sum(y_pred == 0))

print("Predicted Anomaly :", np.sum(y_pred == 1))

# -------------------------------------------------
# If Ground Truth Exists
# -------------------------------------------------

print("\nConfusion Matrix")

print(confusion_matrix(
    np.zeros(len(y_pred)),
    y_pred
))

print("\nClassification Report")

print(classification_report(
    np.zeros(len(y_pred)),
    y_pred,
    zero_division=0
))

print("\nEvaluation Completed Successfully")