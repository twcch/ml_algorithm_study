import numpy as np
import matplotlib.pyplot as plt
from model.regression.linear_regression import LinearRegression

# ===== Demo Data =====
# y = 4x + 3 + noise
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 * X[:, 0] + 3 + np.random.randn(100)

# reshape y
y = y.reshape(-1,)

# ===== 訓練 =====
model = LinearRegression(learning_rate=0.001, n_iterations=100000)
model.fit(X, y)

# ===== 預測 =====
y_pred = model.predict(X)

# ===== 結果視覺化 =====
plt.scatter(X, y, color="red", label="True data")
plt.plot(X, y_pred, color="blue", linewidth=2, label="Prediction")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Demo")
plt.show()

print(f"學到的權重: {model.weights}, 偏置: {model.bias}")