import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from data.dataloader import CaliforniaHousingDataLoader, TrainTestSplit


class LinearRegression:
    def __init__(self):
        self.W = None  # W[0] is intercept, W[1:] are weights

    def fit(self, X, y):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        # y = w0 + w1*x1
        X_bias = np.hstack([np.ones((n, 1)), X])  # Add bias term

        # W = (X^T*X)^-1 * (X^T * y)
        self.W = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        X_bias = np.hstack([np.ones((n, 1)), X])
        return X_bias @ self.W

    def visualize(self, data_split: TrainTestSplit):
        if data_split.x_test.ndim > 1 and data_split.x_test.shape[1] > 1:
            print("Visualization skipped for multivariable regression.")
            return

        y_pred = self.predict(data_split.x_test)
        sorted_indices = np.argsort(data_split.x_test.flatten())
        x_test_sorted = data_split.x_test[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.scatter(data_split.x_test, data_split.y_test, alpha=0.4, label='Actual Data')
        plt.plot(x_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression Line')
        plt.xlabel('Feature (normalized)')
        plt.ylabel('Target ($100k)')
        plt.title('Linear Regression Prediction Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    loader = CaliforniaHousingDataLoader(
        feature_names=["MedInc", "AveRooms", "AveOccup"],  # example multivariate case
        normalize=True
    )
    data_split = loader.load_and_split_data()

    model = LinearRegression()
    model.fit(data_split.x_train, data_split.y_train)

    y_pred = model.predict(data_split.x_test)
    # XWYY = 0
    # W = XYY
    mse = mean_squared_error(data_split.y_test, y_pred)
    r2 = r2_score(data_split.y_test, y_pred)

    print("Model weights (bias + coefficients):", model.W)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R² Score: {r2:.4f}")

    joblib.dump({
        "weights": model.W,
        "mean": loader.scalar_mean,
        "std": loader.scalar_std
    }, "./linear_model.pkl")

    model.visualize(data_split)
