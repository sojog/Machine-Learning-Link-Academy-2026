import numpy as np
from sklearn.datasets import make_regression

# ==========================
# Generarea datelor
# ==========================
X, y = make_regression(
    n_samples=10,
    n_features=2,
    noise=5,
    random_state=42
)

# y trebuie să fie coloană
y = y.reshape(-1, 1)

# ==========================
# Inițializarea parametrilor
# ==========================
np.random.seed(42)

input_size = 2
hidden_size = 4
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

learning_rate = 0.01
epochs = 5000

# ==========================
# Funcții de activare
# ==========================

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# ==========================
# Antrenare
# ==========================
for epoch in range(epochs):

    # Forward
    z1 = X @ W1 + b1
    a1 = relu(z1)

    z2 = a1 @ W2 + b2
    y_pred = z2  # activare liniară pentru regresie

    # Mean Squared Error
    loss = np.mean((y_pred - y) ** 2)

    # Backpropagation

    dLoss = 2 * (y_pred - y) / len(y)

    dW2 = a1.T @ dLoss
    db2 = np.sum(dLoss, axis=0, keepdims=True)

    da1 = dLoss @ W2.T
    dz1 = da1 * relu_derivative(z1)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Gradient Descent
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 500 == 0:
        print(f"Epoca {epoch}, Loss = {loss:.4f}")

# ==========================
# Predicții finale
# ==========================

print("\nValori reale vs. predicții\n")

for real, pred in zip(y, y_pred):
    print(f"Real: {real[0]:8.2f} | Predicție: {pred[0]:8.2f}")