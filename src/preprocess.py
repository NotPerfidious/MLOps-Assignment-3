from sklearn.model_selection import train_test_split
import numpy as np
import os
import yaml

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)


test_size = params["preprocess"]["test_size"]
seed = params["preprocess"]["seed"]

data = np.load("./data/raw/fashion_mnist.npz")

X_temp = data["X_train"]
y_temp = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]


X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=test_size, random_state=seed
)

# Global Z-score standardization
mean = np.mean(X_train)
std = np.std(X_train)
e = 1e-7

X_train = (X_train.astype(np.float32) - mean) / (std + e)
X_val = (X_val.astype(np.float32) - mean) / (std + e)
X_test = (X_test.astype(np.float32) - mean) / (std + e)


os.makedirs("data/processed", exist_ok=True)

np.savez("data/processed/train.npz", X_train=X_train, y_train=y_train)

np.savez("data/processed/val.npz", X_val=X_val, y_val=y_val)

np.savez("data/processed/test.npz", X_test=X_test, y_test=y_test)
