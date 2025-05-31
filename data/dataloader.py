from dataclasses import dataclass
from typing import Optional, Tuple

from sklearn.datasets import fetch_california_housing
import numpy as np

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


@dataclass
class TrainTestSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class CaliforniaHousingDataLoader:
    def __init__(self,
                 feature_names=None,
                 normalize=False,
                 test_size: float = 0.2):
        """
        Parameters:
        - feature_names (list of str or None):
            If None, load all features. Otherwise, load only the selected named features.
        """
        self.feature_names = feature_names
        self.test_size = test_size
        self.normalize = normalize

        self.scalar_mean = None
        self.scalar_std = None

    def load_data(self):
        """
        Loads California housing data.

        Returns:
        - X: np.ndarray of shape (n_samples,) or (n_samples, n_features)
        - y: np.ndarray of shape (n_samples,)
        """
        data = fetch_california_housing()
        all_feature_names = data.feature_names  # List of strings
        X: np.ndarray = data.data
        y: np.ndarray = data.target

        if self.feature_names is not None:
            name_to_index = {name: i for i, name in enumerate(all_feature_names)}
            indices = [name_to_index[name] for name in self.feature_names]
            X = X[:, indices]
            if len(indices) == 1:
                X = X.flatten()

        return X, y

    def load_and_split_data(self) -> TrainTestSplit:
        # Split data into train and test sets
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        if self.normalize:
            if X_train.ndim == 1:
                self.scalar_mean,  self.scalar_std = X_train.mean(), X_train.std()
            else:
                self.scalar_mean = X_train.mean(axis=0)
                self.scalar_std = X_train.std(axis=0)
            X_train = (X_train - self.scalar_mean) / self.scalar_std
            X_test = (X_test - self.scalar_mean) / self.scalar_std

        return TrainTestSplit(X_train, X_test, y_train, y_test)
