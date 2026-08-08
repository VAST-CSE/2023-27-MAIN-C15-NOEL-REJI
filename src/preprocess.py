import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# Paths
# -------------------------------------------------

DATA_PATH = "data/ai4i2020.csv"

MODEL_DIR = "models"

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# -------------------------------------------------
# Features & Target
# -------------------------------------------------

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

# -------------------------------------------------
# Load Data
# -------------------------------------------------

def load_data():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]

    y = df[TARGET]

    # ---------------------------------------------
    # Train/Test Split
    # ---------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ---------------------------------------------
    # Feature Scaling
    # ---------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    # ---------------------------------------------
    # Save Everything
    # ---------------------------------------------

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(scaler, SCALER_PATH)

    joblib.dump(X_test, os.path.join(MODEL_DIR, "X_test.pkl"))

    joblib.dump(y_test, os.path.join(MODEL_DIR, "y_test.pkl"))

    print("Dataset loaded successfully")

    print("Training Samples :", len(X_train))

    print("Testing Samples  :", len(X_test))

    return X_train.astype("float32")