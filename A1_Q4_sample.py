# Question 4
# Import libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import torch

## First, write your name and SID here!
name = ""
SID = ""

## a. Data Generation
"""Generation function"""
def generate_data(num_features=100, num_samples=1000):
    """
    please fix the seed before generating the data
    data: (num_samples, num_features) 2D np.array
    labels: (num_samples, ) 1D np.array
    """
    data = ...
    labels = ...

    return data, labels

## 4.b Logistic Regression and GD
"""Notice that you may need to define additional function for LR and GD"""
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

    train_loss_history = ...
    test_loss_history = ...

    return train_loss_history, test_loss_history

"""calculate the test accuracy"""
def cal_accuracy(X, y, W, b):
    # Calculate the accuracy of logistic regression
    # X: (num_samples, num_features) 2D np.array
    # y: (num_samples, ) 1D np.array
    # W: (num_features, ) 1D np.array
    # b: (1, ) 1D np.array
    # accuracy: float
    accuracy = ...
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

    train_loss_history = ...
    test_loss_history = ...

    return train_loss_history, test_loss_history

## 4.d Plot the losses
"""Plot the training loss and test loss"""
def plot_curves(train_loss_history, test_loss_history, name):
    # Plot the training loss and test loss
    # train_loss_history: (epochs, ) 1D np.array
    # test_loss_history: (epochs, ) 1D np.array

    # save the figure as 'loss.png' and insert into your report
    return

## 4.e PyTorch version
"""Define the LR module"""
class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = ...
        self.sigmoid = ...

    def forward(self, x):
        predictions = ...
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

    train_loss_history = ...
    test_loss_history = ...

    return train_loss_history, test_loss_history

"""calculate the test accuracy"""
def cal_accuracy_torch(X, y, model):
    # Calculate the accuracy of logistic regression
    # X: (num_samples, num_features) 2D np.array
    # y: (num_samples, ) 1D np.array
    # model: torch
    # accuracy: float
    accuracy = ...
    return accuracy

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    epochs = 100
    lr = 0.001

    ## a. Data Generation
    """Generate training set (1000 samples for each class)"""
    X_train, y_train = generate_data(...,...)

    """Generate training set (50 samples for each class)"""
    X_test, y_test = generate_data(...,...)

    ## 4.b Logistic Regression and GD
    """Training"""
    W = ...
    b = ...
    gd_train_loss_history, gd_test_loss_history = train_gd(X_train, y_train, X_test, y_test, W, b, epochs, lr)

    """calculate the test accuracy"""
    gd_accuracy = cal_accuracy(X_test, y_test, W, b)

    ## 4.c SGD
    """Training"""
    W = ...
    b = ...
    sgd_train_loss_history, sgd_test_loss_history = train_sgd(X_train, y_train, X_test, y_test, W, b, epochs, lr)

    """calculate the test accuracy"""
    sgd_accuracy = cal_accuracy(X_test, y_test, W, b)

    ## 4.d Plot the losses
    """Plot the training loss and test loss of GD"""
    plot_curves(gd_train_loss_history, gd_test_loss_history, 'gd')

    """Plot the training loss and test loss of SGD"""
    plot_curves(sgd_train_loss_history, sgd_test_loss_history, 'sgd')

    ## 4.e PyTorch version
    model = ...
    criterion = ...
    optimizer = ...
    torch_train_loss_history, torch_test_loss_history = train_torch(X_train, y_train, X_test, y_test, model, criterion, optimizer, epochs)

    """Plot the training loss and test loss of PyTorch version"""
    plot_curves(torch_train_loss_history, torch_test_loss_history, 'torch')
    """calculate the test accuracy"""
    torch_accuracy = cal_accuracy_torch(X_test, y_test, model)