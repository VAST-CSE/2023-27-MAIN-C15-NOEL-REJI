import numpy as np
import shap

FEATURE_NAMES = [
    "Air Temperature",
    "Process Temperature",
    "Rotational Speed",
    "Torque",
    "Tool Wear"
]


def explain_prediction(model, scaler, sample):

    sample_scaled = scaler.transform(sample)

    background = np.zeros((1, sample.shape[1]))

    explainer = shap.KernelExplainer(
        model.predict,
        background
    )

    shap_values = explainer.shap_values(
        sample_scaled,
        nsamples=100
    )

    values = np.abs(shap_values[0])

    contributions = []

    for feature, value in zip(FEATURE_NAMES, values):

        contributions.append(
            (feature, float(value))
        )

    contributions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return contributions