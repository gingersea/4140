# AIST4010 Assignment 2 — Submission

Name: JIANG Huanyun | SID: 1155233165

## Files

| File | Description |
|------|-------------|
| `A2_report.pdf` / `A2_report.tex` | **The assignment solution (English, to submit)** — Q1–Q7 complete |
| `A2_study_guide.pdf` / `A2_study_guide.md` | Study guide (Chinese + English): background knowledge and explanation for every question |
| `A2_Q5_lenet.py` | Q5: LeNet-5 on Fashion-MNIST (PyTorch). Runs the baseline + all experiments a–g + best model. Reproducible (fixed seed). |
| `A2_Q5_results.json` | Q5: all results (accuracy, per-epoch train loss & test acc curves) |
| `A2_Q5_all_results.csv` | Q5: same results in tabular form |
| `A2_Q6_adam.py` | Q6: Adam optimizer implemented in NumPy, applied to the A1 logistic regression; comparison with SGD (lr=1e-4) |
| `A2_Q6_loss.png` | Q6: training loss convergence plot (Adam vs SGD) |
| `A2_Q5_loss_baseline.png` | Q5: baseline LeNet-5 training loss curve |
| `A2_Q5_acc_curves.png` | Q5: test accuracy curves of the key variants |
| `fmnist_data/` | Fashion-MNIST dataset (torchvision standard folder) |

## How to reproduce

```bash
# Q5 (LeNet-5 experiments; downloads Fashion-MNIST on first run)
python3 A2_Q5_lenet.py

# Q6 (Adam vs SGD)
python3 A2_Q6_adam.py
```

Both scripts print results and save figures/results files.

## Key results

- Q5 baseline LeNet-5 test accuracy: **80.75%**
- Q5 best model (MaxPool + doubled conv channels + ReLU): **84.48%** (+3.73%)
- Q6: Adam (default hyperparameters) converges much faster than SGD (lr=1e-4);
  final train loss 0.4219 (Adam) vs 0.4224 (SGD).
