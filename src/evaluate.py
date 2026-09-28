import tensorflow as tf
import json
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

test_data = np.load("data/processed/test.npz")

X_test = test_data["X_test"]
y_test = test_data["y_test"]

metrics = dict()


model = tf.keras.models.load_model("models/model.h5")

loss_and_accuracy = model.evaluate(X_test, y_test)

metrics["loss"] = float(loss_and_accuracy[0])
metrics["accuracy"] = float(loss_and_accuracy[1])

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

metrics["precision"] = float(precision_score(y_test, y_pred, average="macro"))
metrics["recall"] = float(recall_score(y_test, y_pred, average="macro"))
metrics["f1-score"] = float(f1_score(y_test, y_pred, average="macro"))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
plt.title("Fashion-MNIST Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close(fig)

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)
