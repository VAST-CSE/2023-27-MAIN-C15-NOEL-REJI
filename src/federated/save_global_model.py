import os
import tensorflow as tf

from src.federated.model import create_model

MODEL_PATH = "models/federated_autoencoder.keras"


def save_global_model(weights):

    model = create_model()

    model.set_weights(weights)

    os.makedirs("models", exist_ok=True)

    model.save(MODEL_PATH)

    print("=" * 60)
    print("Federated Global Model Saved Successfully")
    print(MODEL_PATH)
    print("=" * 60)