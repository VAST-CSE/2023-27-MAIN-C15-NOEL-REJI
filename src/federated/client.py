import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

import argparse
import flwr as fl

from src.federated.model import create_model
from src.federated.preprocess import load_factory_data


# -----------------------------------
# Read Factory Name
# -----------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--factory",
    required=True
)

args = parser.parse_args()

factory = args.factory

# -----------------------------------
# Load Factory Dataset
# -----------------------------------

X = load_factory_data(factory)

model = create_model()


class AutoencoderClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):

        model.set_weights(parameters)

        model.fit(
            X,
            X,
            epochs=1,
            batch_size=32,
            verbose=0
        )

        return (
            model.get_weights(),
            len(X),
            {}
        )

    def evaluate(self, parameters, config):

        model.set_weights(parameters)

        loss = model.evaluate(
            X,
            X,
            verbose=0
        )

        return (
            float(loss),
            len(X),
            {"loss": float(loss)}
        )


fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=AutoencoderClient()
)