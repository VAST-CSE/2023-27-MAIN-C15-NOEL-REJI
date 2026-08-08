import tensorflow as tf


def create_model():

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

    return model