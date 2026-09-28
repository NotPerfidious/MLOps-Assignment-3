import tensorflow as tf
import yaml
import numpy as np
import os

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

dense_units = params["train"]["dense_units"]
dropout_rate = params["train"]["dropout_rate"]
learning_rate = params["train"]["learning_rate"]
epochs = params["train"]["epochs"]
batch_size = params["train"]["batch_size"]

train_data = np.load("data/processed/train.npz")
val_data = np.load("data/processed/val.npz")


X_train = train_data["X_train"]
y_train = train_data["y_train"]

X_val = val_data["X_val"]
y_val = val_data["y_val"]

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(units=dense_units, activation="relu"),
        tf.keras.layers.Dropout(rate=dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

os.makedirs("models", exist_ok=True)

csv_logger = tf.keras.callbacks.CSVLogger("models/history.csv")

history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=[csv_logger],
)


model.save("models/model.h5")
