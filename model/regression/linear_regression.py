import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate  # 學習率，決定每次更新參數的步長
        self.n_iterations = n_iterations  # 迭代次數，控制訓練循環的次數
        self.weights = None  # 模型參數 \theta，初始化為空，會在 fit() 裡賦值
        self.bias = None  # 偏置項 b，初始化為空

    def _ensure_2d(self, X):
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def fit(self, X, y):
        X = self._ensure_2d(np.asarray(X))
        y = np.asarray(y).reshape(-1)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            # 前向計算 (預測值)
            y_pred = X @ self.weights + self.bias
            err = y_pred - y

            # 計算梯度
            dw = (X.T @ err) / n_samples
            db = err.mean()

            # 更新規則 (梯度下降)
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

    def predict(self, X):
        X = self._ensure_2d(np.asarray(X))
        return X @ self.weights + self.bias

    # R^2 = 1 - SS_res / SS_tot
    def score(self, X, y):
        X = self._ensure_2d(np.asarray(X))
        y = np.asarray(y).reshape(-1)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def mse(self, X, y):
        X = self._ensure_2d(np.asarray(X))
        y = np.asarray(y).reshape(-1)
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)
