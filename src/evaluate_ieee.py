import os
import joblib
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# Project Paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "autoencoder.keras"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "models",
    "threshold.pkl"
)

X_TEST_PATH = os.path.join(
    BASE_DIR,
    "models",
    "X_test.pkl"
)

Y_TEST_PATH = os.path.join(
    BASE_DIR,
    "models",
    "y_test.pkl"
)

# -------------------------------------------------
# Load Files
# -------------------------------------------------

print("Loading Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Loading Threshold...")

threshold = joblib.load(THRESHOLD_PATH)

print("Loading Test Dataset...")

X_test = joblib.load(X_TEST_PATH)

y_test = joblib.load(Y_TEST_PATH)

print("\nRunning IEEE Evaluation...\n")

# -------------------------------------------------
# Prediction
# -------------------------------------------------

reconstructed = model(
    X_test,
    training=False
).numpy()

errors = np.mean(
    np.square(
        X_test - reconstructed
    ),
    axis=1
)

y_pred = (
    errors > threshold
).astype(int)

# -------------------------------------------------
# Metrics
# -------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

# -------------------------------------------------
# Results
# -------------------------------------------------

print("=" * 50)
print("IEEE EVALUATION RESULTS")
print("=" * 50)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(
    y_test,
    y_pred
))

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\n===================================")
print("IEEE Evaluation Completed Successfully")
print("===================================")