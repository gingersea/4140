"""AIST4010 Fall 2026 — Assignment A0: Iris classification baseline.

Trains a classifier on a0/X_train.csv + a0/y_train.csv and writes
a0/submission.csv in the Kaggle sample_submission format (Id,Predicted).

Usage:
    python a0/solution.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

A0_DIR = Path(__file__).resolve().parent
FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Read the A0 train/test splits and the sample submission."""
    x_train = pd.read_csv(A0_DIR / "X_train.csv", header=None, names=FEATURE_COLS)
    y_train = pd.read_csv(A0_DIR / "y_train.csv", header=None, names=["target"])["target"]
    x_test = pd.read_csv(A0_DIR / "X_test.csv", header=None, names=FEATURE_COLS)
    sample = pd.read_csv(A0_DIR / "sample_submission.csv")
    return x_train, y_train, x_test, sample


def build_model():
    """Standardise features, then fit multinomial logistic regression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=0),
    )


def main() -> None:
    x_train, y_train, x_test, sample = load_data()
    print(f"train rows={len(x_train)}  test rows={len(x_test)}  classes={sorted(y_train.unique())}")

    model = build_model()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")
    print(f"5-fold CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    submission = pd.DataFrame({"Id": sample["Id"], "Predicted": preds.astype(int)})
    out_path = A0_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"wrote {out_path}  rows={len(submission)}  "
          f"label counts={submission['Predicted'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
