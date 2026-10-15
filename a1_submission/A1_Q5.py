"""
AIST4010 Assignment 1 - Question 5
Apply logistic regression, SVM, fully-connected neural network and CNN
on the Iris dataset. Compare their performance and discuss why a deep
neural network does not show its power on this dataset.

Reproducible: 5-fold stratified cross-validation on the 105 labelled
samples, same folds for all four models. Report mean ± std accuracy and
a 45/105 holdout test accuracy.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = "a0"
# Resolve the data directory robustly: prefer a sibling a0/ folder, otherwise
# the one next to this script in the repo.
from pathlib import Path as _Path
if not _Path(BASE_DIR).is_dir() and _Path(__file__).parent.joinpath("..", BASE_DIR).is_dir():
    BASE_DIR = str(_Path(__file__).parent.joinpath("..", BASE_DIR).resolve())
elif not _Path(BASE_DIR).is_dir() and _Path(__file__).parent.joinpath(BASE_DIR).is_dir():
    BASE_DIR = str(_Path(__file__).parent.joinpath(BASE_DIR).resolve())
RANDOM_STATE = 0
N_FOLDS = 5
EPOCHS = 300
LR = 0.05
BATCH = 16


def load_iris():
    X = pd.read_csv(f"{BASE_DIR}/X_train.csv", header=None).to_numpy(dtype=np.float32)
    y = pd.read_csv(f"{BASE_DIR}/y_train.csv", header=None).to_numpy().ravel()
    return X, y


# ----------------------------------------------------------------------
# PyTorch models
# ----------------------------------------------------------------------
class FCN(nn.Module):
    """Fully-connected network: 4 -> 64 -> 32 -> 3 (two hidden layers)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """1-D CNN: (1, 4) -> conv3x4x16 -> conv3x16x32 -> flatten -> 3."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),   # (B, 16, 4)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),  # (B, 32, 4)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                      # (B, 32, 1)
            nn.Flatten(),
        )
        self.head = nn.Linear(32, 3)

    def forward(self, x):
        if x.dim() == 2:          # (B, 4) -> (B, 1, 4)
            x = x.unsqueeze(1)
        return self.head(self.features(x))


def fit_torch(model, Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    model = model.__class__()  # fresh weights every fold
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xtr_t = torch.tensor(Xtr_s)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva_s)
    yva_t = torch.tensor(yva, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=BATCH, shuffle=True)
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(Xva_t).argmax(1) == yva_t).float().mean().item()
    return acc


def run_cv():
    X, y = load_iris()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X, y))

    results = {"logreg": [], "svm": [], "fcn": [], "cnn": []}

    # sklearn models (standardized, multinomial / OvR multiclass)
    for tr, va in folds:
        logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
        logreg.fit(X[tr], y[tr])
        results["logreg"].append(logreg.score(X[va], y[va]))

        svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0))
        svm.fit(X[tr], y[tr])
        results["svm"].append(svm.score(X[va], y[va]))

    for seed, (tr, va) in enumerate(folds):
        results["fcn"].append(fit_torch(FCN(), X[tr], y[tr], X[va], y[va], seed))
        results["cnn"].append(fit_torch(CNN(), X[tr], y[tr], X[va], y[va], seed))

    print("5-fold stratified CV accuracy (mean ± std):")
    summary = {}
    for name, accs in results.items():
        mean, std = float(np.mean(accs)), float(np.std(accs))
        summary[name] = (mean, std, accs)
        print(f"  {name:8s}  {mean:.4f} ± {std:.4f}   {accs}")
    return X, y, folds, summary


def holdout_eval(X, y, folds):
    """Use fold 0 as a deterministic 21-sample holdout test set."""
    tr, va = folds[0]
    rows = []
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
    logreg.fit(X[tr], y[tr])
    rows.append(("Logistic Regression", logreg.score(X[va], y[va])))

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0))
    svm.fit(X[tr], y[tr])
    rows.append(("SVM (RBF)", svm.score(X[va], y[va])))

    rows.append(("FCN (4-64-32-3)", fit_torch(FCN(), X[tr], y[tr], X[va], y[va], 0)))
    rows.append(("CNN (1-D)", fit_torch(CNN(), X[tr], y[tr], X[va], y[va], 0)))
    print("\nHoldout (fold 0, 21 samples):")
    for name, acc in rows:
        print(f"  {name:20s} {acc:.4f}")
    return rows


if __name__ == "__main__":
    X, y, folds, summary = run_cv()
    holdout_eval(X, y, folds)
