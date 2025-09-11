import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def make_classification_data():
    X = np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8], [0.15, 0.25], [0.85, 0.75]])
    y = np.array(["a", "a", "b", "b", "a", "b"])
    X_prop = X[:4]
    y_prop = y[:4]
    X_cal = X[4:5]
    y_cal = y[4:5]
    X_test = X[5:6]
    y_test = y[5:6]
    return X_prop, y_prop, X_cal, y_cal, X_test, y_test


def make_regression_data():
    X = np.linspace(0, 1, 8)[:, None]
    y = (X.ravel() * 2.0) + 0.1
    X_prop = X[:5]
    y_prop = y[:5]
    X_cal = X[5:6]
    y_cal = y[5:6]
    X_test = X[6:8]
    y_test = y[6:8]
    return X_prop, y_prop, X_cal, y_cal, X_test, y_test


class SimpleClassifier:
    """Small deterministic classifier used by several tests."""
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, **kwargs):
        return None

    def predict_proba(self, X):
        probs = np.zeros((len(X), 2))
        probs[:, 0] = 0.3
        probs[:, 1] = 0.7
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class SimpleRegressor:
    def fit(self, X, y, **kwargs):
        return None

    def predict(self, X):
        return np.zeros(len(X))
