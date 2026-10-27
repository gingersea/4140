"""
AIST4010 Assignment 2 - Question 6
Adam optimizer implemented in NumPy, applied to the A1 logistic-regression
problem, compared with SGD (learning rate = 1e-4).

- Adam uses the default hyperparameters discussed in class:
  lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8.
- SGD uses lr = 1e-4 (as required by the assignment).
- Same data, same initialization (W = 0, b = 0), same seed: a fair comparison.
- For SGD the single-sample update of A1 Q4(c) is used (true stochastic GD).
- For Adam the mini-batch version (batch size 32) is used, as is standard in
  practice; the per-step magnitude stays comparable to the default lr=1e-3.

Outputs: A2_Q6_loss.png (training loss curves), printed final losses/accuracies.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data generation (identical to A1 Q4)
# ---------------------------------------------------------------------------
def generate_data(num_features=100, num_samples=1000):
    rng = np.random.RandomState(42)
    w_true = rng.normal(0.0, 1.0, size=num_features)
    b_true = rng.normal(0.0, 0.5)
    n_per_class = num_samples // 2
    labels = np.empty(num_samples, dtype=np.int64)
    labels[:n_per_class] = 0
    labels[n_per_class:] = 1
    data = rng.normal(0.0, 1.0, size=(num_samples, num_features))
    direction = w_true / (np.linalg.norm(w_true) + 1e-12)
    margin = 1.0
    data[:n_per_class] -= margin * direction
    data[n_per_class:] += margin * direction
    noise_std = 0.5
    data += rng.normal(0.0, noise_std, size=data.shape)
    flip_frac = 0.03
    n_flip = int(num_samples * flip_frac)
    flip_idx = rng.choice(num_samples, size=n_flip, replace=False)
    labels[flip_idx] = 1 - labels[flip_idx]
    return data, labels


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y, p):
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def accuracy(X, y, W, b):
    return np.mean((sigmoid(X @ W + b) >= 0.5).astype(np.int64) == y)


# ---------------------------------------------------------------------------
# SGD (single-sample update, same as A1 Q4(c)), lr = 1e-4
# ---------------------------------------------------------------------------
def train_sgd(X_train, y_train, X_test, y_test, W, b, epochs, lr):
    n = X_train.shape[0]
    train_loss_history = np.zeros(epochs)
    test_loss_history = np.zeros(epochs)
    for epoch in range(epochs):
        train_loss_history[epoch] = binary_cross_entropy(
            y_train, sigmoid(X_train @ W + b))
        test_loss_history[epoch] = binary_cross_entropy(
            y_test, sigmoid(X_test @ W + b))
        perm = np.random.permutation(n)
        for i in perm:
            x_i, y_i = X_train[i], y_train[i]
            p_i = sigmoid(np.dot(x_i, W) + b)
            W -= lr * (p_i - y_i) * x_i
            b -= lr * (p_i - y_i)
    return train_loss_history, test_loss_history


# ---------------------------------------------------------------------------
# Adam implemented in NumPy (default hyperparameters from class)
# ---------------------------------------------------------------------------
class Adam:
    """Adam optimizer (NumPy).

    Default hyperparameters (as discussed in class):
        lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8.
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0          # time step
        self.m = None       # 1st moment (W)
        self.v = None       # 2nd moment (W)
        self.m_b = None     # 1st moment (b)
        self.v_b = None     # 2nd moment (b)

    def step(self, W, b, grad_w, grad_b):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(W)
            self.v = np.zeros_like(W)
            self.m_b = np.zeros_like(b)
            self.v_b = np.zeros_like(b)

        # biased first/second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_w
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_w ** 2
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * grad_b ** 2

        # bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

        # parameter update
        W -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
        return W, b


def train_adam(X_train, y_train, X_test, y_test, W, b, epochs,
               lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, batch_size=32):
    """Adam with mini-batch gradient descent (standard practice)."""
    opt = Adam(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
    n = X_train.shape[0]
    train_loss_history = np.zeros(epochs)
    test_loss_history = np.zeros(epochs)

    for epoch in range(epochs):
        train_loss_history[epoch] = binary_cross_entropy(
            y_train, sigmoid(X_train @ W + b))
        test_loss_history[epoch] = binary_cross_entropy(
            y_test, sigmoid(X_test @ W + b))

        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            X_b, y_b = X_train[idx], y_train[idx]
            p_b = sigmoid(X_b @ W + b)
            grad_w = X_b.T @ (p_b - y_b) / len(idx)
            grad_b = np.mean(p_b - y_b)
            W, b = opt.step(W, b, grad_w, grad_b)

    return train_loss_history, test_loss_history


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    epochs = 100

    X_train, y_train = generate_data(100, 2000)
    X_test, y_test = generate_data(100, 100)

    # --- SGD, lr = 1e-4 (required) ---
    W = np.zeros(X_train.shape[1])
    b = np.zeros(1)
    sgd_tr, sgd_te = train_sgd(X_train, y_train, X_test, y_test, W, b,
                               epochs, lr=1e-4)
    sgd_acc = accuracy(X_test, y_test, W, b)

    # --- Adam, default hyperparameters ---
    W = np.zeros(X_train.shape[1])
    b = np.zeros(1)
    adam_tr, adam_te = train_adam(X_train, y_train, X_test, y_test, W, b,
                                  epochs, lr=1e-3, beta1=0.9, beta2=0.999,
                                  eps=1e-8, batch_size=32)
    adam_acc = accuracy(X_test, y_test, W, b)

    print(f"[SGD  lr=1e-4] final train loss = {sgd_tr[-1]:.4f}  "
          f"test loss = {sgd_te[-1]:.4f}  test acc = {sgd_acc:.4f}")
    print(f"[Adam default] final train loss = {adam_tr[-1]:.4f}  "
          f"test loss = {adam_te[-1]:.4f}  test acc = {adam_acc:.4f}")

    # --- plot: convergence speed on the training loss ---
    plt.figure(figsize=(7, 4.5))
    ep = np.arange(1, epochs + 1)
    plt.plot(ep, sgd_tr, label="SGD (lr=1e-4)", linewidth=1.5)
    plt.plot(ep, adam_tr, label="Adam (lr=1e-3, $\\beta_1$=0.9, $\\beta_2$=0.999)",
             linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Training cross-entropy loss")
    plt.title("Convergence comparison: Adam vs SGD (logistic regression, A1 data)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("A2_Q6_loss.png", dpi=150)
    print("Saved A2_Q6_loss.png")
