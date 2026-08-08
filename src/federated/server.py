import os
import sys

# -------------------------------------------------
# Add Project Root to Python Path
# -------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import flwr as fl
from src.federated.strategy import get_strategy


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print("=" * 60)
    print("🚀 Federated Learning Server Started")
    print("=" * 60)
    print("Server Address : 127.0.0.1:8080")
    print("Waiting for 3 clients...")
    print("=" * 60)

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(
            num_rounds=5
        ),
        strategy=get_strategy()
    )


if __name__ == "__main__":
    main()