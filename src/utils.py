import os
import dill
import numpy as np
from sklearn.metrics import r2_score
from src.exception import CustomException
import sys


def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        dill.dump(obj, f)


def load_object(file_path):
    with open(file_path, "rb") as f:
        return dill.load(f)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    report = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report[name] = r2_score(y_test, y_pred)
    return report
