import os
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from preprocess import load_data

# -------------------------------------------------
# Create Results Folder
# -------------------------------------------------

os.makedirs("results", exist_ok=True)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

print("Loading Dataset...")

X = load_data().astype(np.float32)

print("Dataset Shape :", X.shape)

# -------------------------------------------------
# Load Model
# -------------------------------------------------

print("Loading Autoencoder...")

model = tf.keras.models.load_model(
    "models/autoencoder.keras"
)

# -------------------------------------------------
# Load Threshold
# -------------------------------------------------

threshold = joblib.load(
    "models/threshold.pkl"
)

print("Threshold :", threshold)

# -------------------------------------------------
# Reconstruction
# -------------------------------------------------

print("Calculating Reconstruction Errors...")

reconstructed = model.predict(
    X,
    batch_size=256,
    verbose=1
)

errors = np.mean(
    np.square(
        X - reconstructed
    ),
    axis=1
)

# -------------------------------------------------
# Reconstruction Error Histogram
# -------------------------------------------------

plt.figure(figsize=(8,5))

plt.hist(
    errors,
    bins=50
)

plt.axvline(
    threshold,
    linestyle="--",
    linewidth=2,
    label="Threshold"
)

plt.title("Reconstruction Error Distribution")

plt.xlabel("Reconstruction Error")

plt.ylabel("Frequency")

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/reconstruction_error_histogram.png"
)

plt.close()

print("✓ Reconstruction Error Histogram Saved")

# -------------------------------------------------
# Reconstruction Error Line Plot
# -------------------------------------------------

plt.figure(figsize=(10,5))

plt.plot(
    errors,
    linewidth=1
)

plt.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label="Threshold"
)

plt.title("Reconstruction Error")

plt.xlabel("Sample Index")

plt.ylabel("MSE")

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/reconstruction_error_line.png"
)

plt.close()

print("✓ Reconstruction Error Line Plot Saved")

# -------------------------------------------------
# Feature Distribution
# -------------------------------------------------

feature_names = [
    "Air Temp",
    "Process Temp",
    "RPM",
    "Torque",
    "Tool Wear"
]

for i in range(5):

    plt.figure(figsize=(6,4))

    plt.hist(
        X[:, i],
        bins=30
    )

    plt.title(
        feature_names[i]
    )

    plt.tight_layout()

    filename = feature_names[i].replace(" ", "_")

    plt.savefig(
        f"results/{filename}.png"
    )

    plt.close()

print("✓ Feature Distribution Graphs Saved")

print("\n===================================")
print("ALL IEEE FIGURES GENERATED")
print("===================================")