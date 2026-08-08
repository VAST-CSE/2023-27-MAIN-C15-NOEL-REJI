import os
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from preprocess import load_data

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

print("=" * 50)
print("LOADING DATASET")
print("=" * 50)

X = load_data().astype(np.float32)

print("Dataset Shape :", X.shape)

# -------------------------------------------------
# Build Autoencoder
# -------------------------------------------------

print("\nBuilding Autoencoder...\n")

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

print("Model Created Successfully")

# -------------------------------------------------
# Train Model
# -------------------------------------------------

print("\nStarting Training...\n")

history = model.fit(
    X,
    X,
    epochs=10,
    batch_size=256,
    shuffle=True,
    verbose=1
)

print("\nTraining Finished Successfully")

# -------------------------------------------------
# Save Training Loss Graph
# -------------------------------------------------

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    linewidth=2,
    label="Training Loss"
)

plt.title("Autoencoder Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss (MSE)")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig("results/training_loss.png")

plt.close()

print("Training Loss Graph Saved")

# -------------------------------------------------
# Calculate Threshold
# -------------------------------------------------

print("\nCalculating Threshold...\n")

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

threshold = np.percentile(
    errors,
    99
)

print(f"\nDynamic Threshold : {threshold:.6f}")

# -------------------------------------------------
# Save Files
# -------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/autoencoder.keras"
)

joblib.dump(
    threshold,
    "models/threshold.pkl"
)

# -------------------------------------------------
# Completed
# -------------------------------------------------

print("\n" + "=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 50)

print("✓ Autoencoder Model Saved")
print("✓ Scaler Saved (from preprocess.py)")
print("✓ Threshold Saved")
print("✓ Training Loss Graph Saved")
print("✓ Ready for Prediction")