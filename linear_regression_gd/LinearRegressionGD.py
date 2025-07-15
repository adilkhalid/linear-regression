import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from data.dataloader import CaliforniaHousingDataLoader, TrainTestSplit


class LinearRegressionGD:
    def __init__(self, regul=0):
        self.regul = regul
        self.W = None

    def fit(self, X, y, lr=0.0001, num_iter=1000):
        # Input validation
        if len(X) != len(y) or len(X) == 0:
            raise ValueError("X and y must have the same length and cannot be empty")

        # Add bias term to X -> [1 X]
        X = np.hstack([np.ones((len(X), 1)), X])

        # Initialize W to zeros
        self.W = np.zeros(X.shape[1])

        # Use gradient descent to minimize cost function
        for i in range(num_iter):
            # Calculate predicted values
            y_pred = np.dot(X, self.W)

            # Calculate cost function
            cost = np.sum((y_pred - y) ** 2) + self.regul * np.sum(self.W ** 2)

            # Calculate gradients
            # 2 * X^T*(y_pred-y_true)
            gradients = 2 * np.dot(X.T, (y_pred - y)) # + 2 * self.regul * self.W
            # np.clip(gradients, -1e5, 1e5, out=gradients)
            # Update W
            self.W = self.W - lr * gradients

            if i % 1000 == 0:
                print(f"Iteration {i}, Cost: {cost}, W Norm: {np.linalg.norm(self.W)}")
                print(cost)

    def predict(self, X):
        # Add bias term to X
        X = np.hstack([np.ones((len(X), 1)), X])

        # Calculate predicted values
        y_pred = np.dot(X, self.W)
        return y_pred


if __name__ == "__main__":
    features = ["MedInc", "AveRooms", "AveOccup"]
    loader = CaliforniaHousingDataLoader(feature_names=features, normalize=True)
    data_split: TrainTestSplit = loader.load_and_split_data()

    model = LinearRegressionGD(regul=0.1)
    model.fit(data_split.x_train, data_split.y_train, lr=1e-5, num_iter=10000)

    y_pred = model.predict(data_split.x_test)

    mse = mean_squared_error(data_split.y_test, y_pred)
    r2 = r2_score(data_split.y_test, y_pred)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test R² Score: {r2:.4f}")

    # Save model weights and normalization info
    joblib.dump({
        "weights": model.W,
        "mean": loader.scalar_mean,
        "std": loader.scalar_std,
        "features": features
    }, "./linear_model_gd.pkl")
