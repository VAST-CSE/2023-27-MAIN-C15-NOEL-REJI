import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = pd.read_csv("data/ai4i2020.csv")

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"

X = df[FEATURES]
y = df[TARGET]

# -------------------------------------------------
# Train Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# Scaling
# -------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# -------------------------------------------------
# Autoencoder
# -------------------------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(5,)),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(16, activation="relu"),

    tf.keras.layers.Dense(8, activation="relu"),

    tf.keras.layers.Dense(16, activation="relu"),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(5)

])

model.compile(
    optimizer="adam",
    loss="mse"
)

print("Training IEEE Model...")

model.fit(

    X_train,
    X_train,

    epochs=10,

    batch_size=256,

    verbose=1

)

# -------------------------------------------------
# Threshold
# -------------------------------------------------

reconstruction = model.predict(
    X_train,
    verbose=0
)

errors = np.mean(
    np.square(X_train - reconstruction),
    axis=1
)

threshold = np.percentile(errors,99)

# -------------------------------------------------
# Save
# -------------------------------------------------

os.makedirs("models_ieee",exist_ok=True)

model.save("models_ieee/autoencoder.keras")

joblib.dump(scaler,"models_ieee/scaler.pkl")

joblib.dump(threshold,"models_ieee/threshold.pkl")

joblib.dump(X_test,"models_ieee/X_test.pkl")

joblib.dump(y_test,"models_ieee/y_test.pkl")

print("\nIEEE Training Completed Successfully")