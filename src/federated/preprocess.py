import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


def load_factory_data(factory):

    path = f"data/{factory}.csv"

    df = pd.read_csv(path)

    X = df[FEATURES]

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    return X