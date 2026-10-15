# Question 4
# Import libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import torch

## First, write your name and SID here!
name = "JIANG Huanyun"
SID = "1155233165"

## a. Data Generation
"""Generation function"""
def generate_data(num_features=100, num_samples=1000):
    """
    please fix the seed before generating the data
    data: (num_samples, num_features) 2D np.array
    labels: (num_samples, ) 1D np.array
    """
    # Fix the seed inside the function so that every call is reproducible
    rng = np.random.RandomState(42)

    # A ground-truth separating hyperplane
    w_true = rng.normal(0.0, 1.0, size=num_features)
    b_true = rng.normal(0.0, 0.5)

    # Half of the samples from each class
    n_per_class = num_samples // 2
    labels = np.empty(num_samples, dtype=np.int64)
    labels[:n_per_class] = 0
    labels[n_per_class:] = 1

    # Inputs drawn from a unit Gaussian, then shifted along the normal
    # direction so that the two classes lie on opposite sides of the plane.
    data = rng.normal(0.0, 1.0, size=(num_samples, num_features))
    direction = w_true / (np.linalg.norm(w_true) + 1e-12)
    margin = 1.0
    data[:n_per_class] -= margin * direction
    data[n_per_class:] += margin * direction

    # Label-flipping noise and Gaussian feature noise make the data not
    # perfectly linearly separable.
    noise_std = 0.5
    data += rng.normal(0.0, noise_std, size=data.shape)

    flip_frac = 0.03
    n_flip = int(num_samples * flip_frac)
    flip_idx = rng.choice(num_samples, size=n_flip, replace=False)
    labels[flip_idx] = 1 - labels[flip_idx]

    return data, labels

## 4.b Logistic Regression and GD
"""Notice that you may need to define additional function for LR and GD"""

def sigmoid(z):
    """Numerically stable sigmoid: z: (n, ) -> (n, )"""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def binary_cross_entropy(y, p):
    """Average binary cross-entropy loss, p = P(y=1|x)"""
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def predict_prob(X, W, b):
    """P(y=1|x) for all samples: X (n, d), W (d, ), b (1, )"""
    return sigmoid(X @ W + b)

"""Training of GD"""
def train_gd(X_train, y_train, X_test, y_test, W, b, epochs, lr):
    # Train logistic regression using gradient descent
    # X_train: (num_samples, num_features) 2D np.array
    # y_train: (num_samples, ) 1D np.array
    # X_test: (num_samples, num_features) 2D np.array
    # y_test: (num_samples, ) 1D np.array
    # W: (num_features, ) 1D np.array
    # b: (1, ) 1D np.array
    # lr: learning rate set to be 0.001
    # epochs: number of epochs set to be 100

    # train_loss_history: (epochs, ) 1D np.array
    # test_loss_history: (epochs, ) 1D np.array

    n = X_train.shape[0]
    train_loss_history = np.zeros(epochs)
    test_loss_history = np.zeros(epochs)

    for epoch in range(epochs):
        p_train = predict_prob(X_train, W, b)
        train_loss_history[epoch] = binary_cross_entropy(y_train, p_train)
        test_loss_history[epoch] = binary_cross_entropy(y_test, predict_prob(X_test, W, b))

        # Full-batch gradients of the average cross-entropy
        grad_w = X_train.T @ (p_train - y_train) / n
        grad_b = np.mean(p_train - y_train)
        W -= lr * grad_w
        b -= lr * grad_b

    return train_loss_history, test_loss_history

"""calculate the test accuracy"""
def cal_accuracy(X, y, W, b):
    # Calculate the accuracy of logistic regression
    # X: (num_samples, num_features) 2D np.array
    # y: (num_samples, ) 1D np.array
    # W: (num_features, ) 1D np.array
    # b: (1, ) 1D np.array
    # accuracy: float
    y_pred = (predict_prob(X, W, b) >= 0.5).astype(np.int64)
    accuracy = np.mean(y_pred == y)
    return accuracy

## 4.c SGD
"""Training function of SGD"""
def train_sgd(X_train, y_train, X_test, y_test, W, b, epochs, lr):
    # Train logistic regression using stochastic gradient descent
    # X_train: (num_samples, num_features) 2D np.array
    # y_train: (num_samples, ) 1D np.array
    # X_test: (num_samples, num_features) 2D np.array
    # y_test: (num_samples, ) 1D np.array
    # W: (num_features, ) 1D np.array
    # b: (1, ) 1D np.array
    # lr: learning rate set to be 0.001
    # epochs: number of epochs set to be 100

    # train_loss_history: (epochs, ) 1D np.array
    # test_loss_history: (epochs, ) 1D np.array

    n = X_train.shape[0]
    train_loss_history = np.zeros(epochs)
    test_loss_history = np.zeros(epochs)

    # One pass over the (shuffled) training set per epoch, one sample at a time
    for epoch in range(epochs):
        # Record the full-batch losses at the start of each epoch for a fair
        # comparison with GD.
        train_loss_history[epoch] = binary_cross_entropy(y_train, predict_prob(X_train, W, b))
        test_loss_history[epoch] = binary_cross_entropy(y_test, predict_prob(X_test, W, b))

        perm = np.random.permutation(n)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        for i in range(n):
            x_i = X_shuffled[i]
            y_i = y_shuffled[i]
            p_i = sigmoid(np.dot(x_i, W) + b)
            grad_w = (p_i - y_i) * x_i
            grad_b = p_i - y_i
            W -= lr * grad_w
            b -= lr * grad_b

    return train_loss_history, test_loss_history

## 4.d Plot the losses
"""Plot the training loss and test loss"""
def plot_curves(train_loss_history, test_loss_history, name):
    # Plot the training loss and test loss
    # train_loss_history: (epochs, ) 1D np.array
    # test_loss_history: (epochs, ) 1D np.array

    # save the figure as 'loss.png' and insert into your report
    plt.figure(figsize=(6, 4))
    epochs = np.arange(1, len(train_loss_history) + 1)
    plt.plot(epochs, train_loss_history, label="Train loss")
    plt.plot(epochs, test_loss_history, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"Logistic regression ({name})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{name}_loss.png", dpi=150)
    plt.close()
    return

## 4.e PyTorch version
"""Define the LR module"""
class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, 1)
        self.sigmoid = torch.nn.Sigmoid()
        # Start from W = 0, b = 0 so the PyTorch run is directly comparable
        # with the NumPy GD/SGD runs, which also start from zeros.
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        predictions = self.sigmoid(self.linear(x))
        return predictions

"""Training function of PyTorch version"""
def train_torch(X_train, y_train, X_test, y_test, model, criterion, optimizer, epochs):
    # Train logistic regression using stochastic gradient descent
    # X_train: (num_samples, num_features) 2D np.array
    # y_train: (num_samples, ) 1D np.array
    # X_test: (num_samples, num_features) 2D np.array
    # y_test: (num_samples, ) 1D np.array
    # model: LogisticRegression
    # criterion: torch.nn.BCELoss
    # optimizer: torch.optim.SGD
    # epochs: number of epochs set to be 100
    # lr: learning rate set to be 0.001

    # train_loss_history: (epochs, ) 1D np.array
    # test_loss_history: (epochs, ) 1D np.array

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loss_history = np.zeros(epochs)
    test_loss_history = np.zeros(epochs)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
        train_loss_history[epoch] = loss.item()

        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test_t), y_test_t)
            test_loss_history[epoch] = test_loss.item()

    return train_loss_history, test_loss_history

"""calculate the test accuracy"""
def cal_accuracy_torch(X, y, model):
    # Calculate the accuracy of logistic regression
    # X: (num_samples, num_features) 2D np.array
    # y: (num_samples, ) 1D np.array
    # model: torch
    # accuracy: float
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y_pred = (model(X_t).squeeze() >= 0.5).long()
    accuracy = torch.mean((y_pred == y_t).float()).item()
    return accuracy

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    epochs = 100
    lr = 0.001

    ## a. Data Generation
    """Generate training set (1000 samples for each class)"""
    X_train, y_train = generate_data(100, 2000)

    """Generate testing set (50 samples for each class)"""
    X_test, y_test = generate_data(100, 100)

    ## 4.b Logistic Regression and GD
    """Training"""
    W = np.zeros(X_train.shape[1])
    b = np.zeros(1)
    gd_train_loss_history, gd_test_loss_history = train_gd(X_train, y_train, X_test, y_test, W, b, epochs, lr)

    """calculate the test accuracy"""
    gd_accuracy = cal_accuracy(X_test, y_test, W, b)
    print(f"[GD]  test accuracy = {gd_accuracy:.4f}")

    ## 4.c SGD
    """Training"""
    W = np.zeros(X_train.shape[1])
    b = np.zeros(1)
    sgd_train_loss_history, sgd_test_loss_history = train_sgd(X_train, y_train, X_test, y_test, W, b, epochs, lr)

    """calculate the test accuracy"""
    sgd_accuracy = cal_accuracy(X_test, y_test, W, b)
    print(f"[SGD] test accuracy = {sgd_accuracy:.4f}")

    ## 4.d Plot the losses
    """Plot the training loss and test loss of GD"""
    plot_curves(gd_train_loss_history, gd_test_loss_history, 'gd')

    """Plot the training loss and test loss of SGD"""
    plot_curves(sgd_train_loss_history, sgd_test_loss_history, 'sgd')

    ## 4.e PyTorch version
    model = LogisticRegression(X_train.shape[1])
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    torch_train_loss_history, torch_test_loss_history = train_torch(X_train, y_train, X_test, y_test, model, criterion, optimizer, epochs)

    """Plot the training loss and test loss of PyTorch version"""
    plot_curves(torch_train_loss_history, torch_test_loss_history, 'torch')
    """calculate the test accuracy"""
    torch_accuracy = cal_accuracy_torch(X_test, y_test, model)
    print(f"[Torch] test accuracy = {torch_accuracy:.4f}")
