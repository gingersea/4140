from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
FEATURE_COLS = ("sepal_length", "sepal_width", "petal_length", "petal_width")
TARGET_COL = "target"
ID_COL = "Id"
PRED_COL = "Predicted"

CV_SPLITS = 5
RANDOM_STATE = 0
SUBMISSION_PATH = BASE_DIR / "submission.csv"


@dataclass(frozen=True)
class Dataset:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    sample_submission: pd.DataFrame


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, **kwargs)


def load_data() -> Dataset:
    """Load train/test splits and the sample submission from BASE_DIR."""
    x_train = _read_csv(BASE_DIR / "X_train.csv", header=None, names=FEATURE_COLS)
    y_train = _read_csv(BASE_DIR / "y_train.csv", header=None, names=[TARGET_COL])[TARGET_COL]
    x_test = _read_csv(BASE_DIR / "X_test.csv", header=None, names=FEATURE_COLS)
    sample = _read_csv(BASE_DIR / "sample_submission.csv")

    if len(x_train) != len(y_train):
        raise ValueError(f"Train/test row mismatch: {len(x_train)} vs {len(y_train)}")
    if len(x_test) != len(sample):
        raise ValueError(f"Test/sample row mismatch: {len(x_test)} vs {len(sample)}")

    return Dataset(x_train, y_train, x_test, sample)


def build_model() -> Pipeline:
    """Standardise features, then fit multinomial logistic regression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )


def cross_validate(model: Pipeline, dataset: Dataset) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model, dataset.x_train, dataset.y_train, cv=cv, scoring="accuracy"
    )
    return float(scores.mean()), float(scores.std())


def train_and_predict(model: Pipeline, dataset: Dataset) -> pd.Series:
    model.fit(dataset.x_train, dataset.y_train)
    return pd.Series(model.predict(dataset.x_test), name=PRED_COL)


def save_submission(predictions: pd.Series, sample: pd.DataFrame, path: Path) -> None:
    submission = pd.DataFrame({ID_COL: sample[ID_COL], PRED_COL: predictions.astype(int)})
    submission.to_csv(path, index=False)
    logger.info("wrote %s rows=%d counts=%s", path, len(submission),
                submission[PRED_COL].value_counts().to_dict())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dataset = load_data()
    logger.info(
        "train rows=%d  test rows=%d  classes=%s",
        len(dataset.x_train), len(dataset.x_test),
        sorted(dataset.y_train.unique()),
    )

    model = build_model()
    mean, std = cross_validate(model, dataset)
    logger.info("%d-fold CV accuracy: %.4f +/- %.4f", CV_SPLITS, mean, std)

    predictions = train_and_predict(model, dataset)
    save_submission(predictions, dataset.sample_submission, SUBMISSION_PATH)


if __name__ == "__main__":
    main()
