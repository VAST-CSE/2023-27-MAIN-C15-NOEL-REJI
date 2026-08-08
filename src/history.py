import csv
import os
from datetime import datetime

HISTORY_FILE = "logs/prediction_history.csv"


def save_prediction(
    air_temp,
    process_temp,
    rotational_speed,
    torque,
    tool_wear,
    prediction,
    error
):
    os.makedirs("logs", exist_ok=True)

    file_exists = os.path.isfile(HISTORY_FILE)

    with open(HISTORY_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Air Temperature",
                "Process Temperature",
                "Rotational Speed",
                "Torque",
                "Tool Wear",
                "Prediction",
                "Reconstruction Error"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            air_temp,
            process_temp,
            rotational_speed,
            torque,
            tool_wear,
            prediction,
            float(error)
        ])