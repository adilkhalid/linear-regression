import numpy as np


class LinearRegression:
    def __init__(self):
        self.W = None

    def fit(self, X, y: np.ndarray):
        """
        X: n x d
        """
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean


    def predict(self, X):
        n = X.shape[0]
        X = np.hstack([X, np.ones((n, 1))])
        return X @ self.W
