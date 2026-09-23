# AIST4010 Fall 2026 — Assignment A0

Kaggle competition: [aist-4010-fall-2026-a-0](https://www.kaggle.com/competitions/aist-4010-fall-2026-a-0/overview)

## Task

Classify iris plants into one of 3 species from 4 numeric measurements
(sepal length/width, petal length/width in cm). This is Fisher's classic
Iris dataset — see [`iris_information.txt`](iris_information.txt).

- Train: 105 labelled samples (`X_train.csv`, `y_train.csv`)
- Test: 45 samples (`X_test.csv`)
- Labels: `0`, `1`, `2` (setosa / versicolour / virginica)

## Data files

| File | Description |
|---|---|
| `X_train.csv` | 105 training rows, 4 features, no header |
| `y_train.csv` | 105 training labels (`0`/`1`/`2`), no header |
| `X_test.csv` | 45 test rows, 4 features, no header |
| `sample_submission.csv` | Required submission format: `Id,Predicted` |
| `iris_information.txt` | UCI Iris dataset documentation |

## Approach

`solution.py` builds a scikit-learn pipeline:

1. `StandardScaler` — standardise the 4 features.
2. `LogisticRegression(max_iter=1000)` — multinomial logistic regression.

No tuning is needed for this near-linearly-separable problem.

## Result

| Metric | Value |
|---|---|
| 5-fold CV accuracy | **0.9524 ± 0.0426** |
| Predicted class counts (test) | {0: 14, 1: 18, 2: 13} |

## Reproduce

```bash
python a0/solution.py
```

Outputs `a0/submission.csv` (45 rows, columns `Id,Predicted`, Ids 0–44),
matching `sample_submission.csv`.
