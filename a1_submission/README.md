# AIST4010 Assignment 1 — Submission

## Files

| File | Purpose |
|---|---|
| `A1_report_compact.pdf` | Written report: cover page + Q1–Q6 (submit this as .pdf) |
| `A1_Q4.py` | Q4 code: data generation + NumPy GD/SGD + PyTorch logistic regression |
| `A1_Q5.py` | Q5 code: Iris comparison (LR / SVM / FCN / CNN, 5-fold CV) |
| `gd_loss.png`, `sgd_loss.png`, `torch_loss.png` | Q4(d)/(e) loss curves (also embedded in the PDF) |
| `q3_roc.png` | Q3 ROC curve (also embedded in the PDF) |

## How to run

```bash
python3 A1_Q4.py   # -> prints accuracies, saves gd/sgd/torch_loss.png
python3 A1_Q5.py   # -> prints 5-fold CV accuracies for the 4 models
```

Requires: numpy, matplotlib, torch, scikit-learn.

## Notes

- Fill in Name and Student ID on the cover page of the PDF.
- Q4 results are fully reproducible (seed fixed at 42):
  GD 0.84, SGD 0.84, PyTorch 0.84 test accuracy.
- Q3 AUC = 7/12 (verified with sklearn).
