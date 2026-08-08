import tensorflow as tf
import numpy as np
import time

print("Loading model...")
model = tf.keras.models.load_model("models/autoencoder.keras")

print("Creating test data...")
x = np.random.rand(100, 5)

print("Starting timer...")
start = time.time()

print("Running prediction...")
y = model.predict(
    x,
    batch_size=32,
    verbose=1
)

end = time.time()

print("Prediction Finished!")
print(f"Prediction Time: {end - start:.2f} seconds")
print("Output Shape:", y.shape)