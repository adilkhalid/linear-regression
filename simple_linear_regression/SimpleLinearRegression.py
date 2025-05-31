import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

from data.dataloader import CaliforniaHousingDataLoader, TrainTestSplit
import joblib

# Visualize predictions
import matplotlib.pyplot as plt


class SimpleLinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean

    def predict(self, X):
        return self.slope * X + self.intercept

    def visualize(self, data_split: TrainTestSplit):
        sorted_indices = np.argsort(data_split.x_test)
        x_test_sorted = data_split.x_test[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.scatter(data_split.x_test, data_split.y_test, alpha=0.4, label='Actual Data')
        plt.plot(x_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Regression Line')
        plt.xlabel('Median Income (normalized)')
        plt.ylabel('Median House Value ($100k)')
        plt.title('Simple Linear Regression Prediction Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    loader = CaliforniaHousingDataLoader(feature_names=["MedInc"], normalize=True)
    trainTestSplit: TrainTestSplit = loader.load_and_split_data()

    linearRegressionModel = SimpleLinearRegression()
    linearRegressionModel.fit(trainTestSplit.x_train, trainTestSplit.y_train)

    y_pred = linearRegressionModel.predict(trainTestSplit.x_test)

    mse = mean_squared_error(trainTestSplit.y_test, y_pred)
    r2 = r2_score(trainTestSplit.y_test, y_pred)

    print(f"Slope: {linearRegressionModel.slope:.4f}")
    print(f"Intercept: {linearRegressionModel.intercept:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R² Score: {r2:.4f}")

    # Save model
    joblib.dump({
        "slope": linearRegressionModel.slope,
        "intercept": linearRegressionModel.intercept,
        "mean": loader.scalar_mean,  # if applicable
        "std": loader.scalar_std  # if applicable
    }, "./simple_linear_model.pkl")

    linearRegressionModel.visualize(trainTestSplit)
