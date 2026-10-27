"""
AIST4010 Assignment 2 - Question 5
LeNet-5 (PyTorch) on Fashion-MNIST.

- Baseline: the classic LeNet-5 architecture (32x32 input version, 28x28 here).
- Experiments a-g: each with at least two settings, same training protocol.
- Reproducible: fixed seed, deterministic data order, deterministic CuDNN ops.
- Protocol (following the assignment):
    * SGD, lr=0.01, 10 epochs, batch 128
    * PyTorch default (Kaiming) initialization, no momentum, no weight decay,
      no data augmentation
  Each experiment tests ONLY ONE factor in isolation.
- Outputs: A2_Q5_results.json, A2_Q5_all_results.csv, A2_Q5_loss_baseline.png
  (results are also printed).
"""
import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = os.environ.get("A2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.01
DATA_DIR = "fmnist_data"  # relative to this file's directory

torch.manual_seed(SEED)

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def g():
    """Deterministic generator: same data order for every run."""
    return torch.Generator().manual_seed(SEED)


def seed_worker(worker_id):  # for DataLoader workers (if any)
    torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_ds = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g(),
                          worker_init_fn=seed_worker, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_dl, test_dl


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LeNet5(nn.Module):
    """Classic LeNet-5: 6@5x5 -> AvgPool2 -> 16@5x5 -> AvgPool2 -> FC120 -> FC84 -> 10."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),   # 28x28x6
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),       # 14x14x6
            nn.Conv2d(6, 16, kernel_size=5),             # 10x10x16
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),       # 5x5x16
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


class LeNet5ReLU(nn.Module):
    """LeNet-5 with ReLU instead of Tanh (used for the final best model)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def _init(model):
    """Keep PyTorch's default (Kaiming) initialization for all layers."""
    return model


def width_scaled(width):
    """LeNet-5 with all conv channels scaled by `width` (wider or narrower)."""
    m = LeNet5()
    m.features[0] = nn.Conv2d(1, 6 * width, 5, padding=2)
    m.features[3] = nn.Conv2d(6 * width, 16 * width, 5)
    m.classifier[0] = nn.Linear(16 * width * 5 * 5, 120)
    return _init(m)


def conv_out_size(in_size, kernel, stride, padding, dilation=1):
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_eval(model, train_dl, test_dl, epochs=EPOCHS, lr=LR, momentum=0.0):
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        hist["train_loss"].append(round(total_loss / n_batches, 6))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()
        hist["test_acc"].append(round(100.0 * correct / total, 4))

    final_acc = hist["test_acc"][-1]
    return model, final_acc, hist


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run_experiment(name, build, train_dl, test_dl, epochs=EPOCHS, lr=LR):
    torch.manual_seed(SEED)
    model = _init(build())
    t0 = time.time()
    _, acc, hist = train_eval(model, train_dl, test_dl, epochs=epochs, lr=lr)
    dt = time.time() - t0
    print(f"{name:50s} test_acc={acc:6.2f}%  time={dt:6.1f}s")
    return {"name": name, "test_acc": acc, "train_loss": hist["train_loss"],
            "test_acc_curve": hist["test_acc"], "time_s": round(dt, 1)}


def main():
    train_dl, test_dl = get_data()

    results = {}

    # --- baseline ---
    base = run_experiment("baseline_LeNet5", LeNet5, train_dl, test_dl)
    results["baseline_LeNet5"] = base
    base_acc = base["test_acc"]

    # a. add / remove CONV layers
    def a_remove():
        m = LeNet5()
        m.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.Tanh(),
            nn.AvgPool2d(2),                        # 14x14x6
            nn.Conv2d(6, 16, 3),                    # 12x12x16  (smaller kernel: replace 2nd conv)
            nn.Tanh(),
            nn.AvgPool2d(2),                        # 6x6x16
        )
        m.classifier[0] = nn.Linear(16 * 6 * 6, 120)
        return _init(m)

    def a_add():
        m = LeNet5()
        m.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.Tanh(),      # 28x28x6
            nn.AvgPool2d(2),                                # 14x14x6
            nn.Conv2d(6, 16, 3, padding=1), nn.Tanh(),      # 14x14x16
            nn.AvgPool2d(2),                                # 7x7x16
            nn.Conv2d(16, 32, 3, padding=1), nn.Tanh(),     # 7x7x32
        )
        m.classifier[0] = nn.Linear(32 * 7 * 7, 120)
        return _init(m)

    results["a_remove_conv"] = run_experiment("a_remove_conv", a_remove, train_dl, test_dl)
    results["a_add_conv"] = run_experiment("a_add_conv", a_add, train_dl, test_dl)

    # b. add / remove FC layers
    def b_remove():
        m = LeNet5()
        m.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 84), nn.Tanh(),
            nn.Linear(84, 10),
        )
        return _init(m)

    def b_add():
        m = LeNet5()
        m.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.Tanh(),
            nn.Linear(120, 84), nn.Tanh(),
            nn.Linear(84, 42), nn.Tanh(),
            nn.Linear(42, 10),
        )
        return _init(m)

    results["b_remove_fc"] = run_experiment("b_remove_fc", b_remove, train_dl, test_dl)
    results["b_add_fc"] = run_experiment("b_add_fc", b_add, train_dl, test_dl)

    # c. dropout
    def c_dropout(p):
        def build():
            m = LeNet5()
            m.classifier = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120), nn.Tanh(), nn.Dropout(p),
                nn.Linear(120, 84), nn.Tanh(), nn.Dropout(p),
                nn.Linear(84, 10),
            )
            return _init(m)
        return build

    results["c_dropout_0.5"] = run_experiment("c_dropout_0.5", c_dropout(0.5), train_dl, test_dl)
    results["c_dropout_0.25"] = run_experiment("c_dropout_0.25", c_dropout(0.25), train_dl, test_dl)

    # d. pooling settings
    def d_maxpool():
        m = LeNet5()
        m.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.Tanh(),
            nn.MaxPool2d(2),
        )
        return _init(m)

    def d_avgpool3():
        m = LeNet5()
        m.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.Tanh(),   # 28x28x6
            nn.AvgPool2d(3, stride=2),                   # 13x13x6
            nn.Conv2d(6, 16, 5),                         # 9x9x16
            nn.Tanh(),
            nn.AvgPool2d(2),                             # 4x4x16
        )
        m.classifier[0] = nn.Linear(16 * 4 * 4, 120)
        return _init(m)

    results["d_maxpool"] = run_experiment("d_maxpool", d_maxpool, train_dl, test_dl)
    results["d_avgpool3"] = run_experiment("d_avgpool3", d_avgpool3, train_dl, test_dl)

    # e. convolution window size
    def e_kernel3():
        m = LeNet5()
        m.features = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1), nn.Tanh(),   # 28x28x6
            nn.AvgPool2d(2),                             # 14x14x6
            nn.Conv2d(6, 16, 3),                         # 12x12x16
            nn.Tanh(),
            nn.AvgPool2d(2),                             # 6x6x16
        )
        m.classifier[0] = nn.Linear(16 * 6 * 6, 120)
        return _init(m)

    def e_kernel7():
        m = LeNet5()
        m.features = nn.Sequential(
            nn.Conv2d(1, 6, 7, padding=3), nn.Tanh(),   # 28x28x6
            nn.AvgPool2d(2),                             # 14x14x6
            nn.Conv2d(6, 16, 7),                         # 8x8x16
            nn.Tanh(),
            nn.AvgPool2d(2),                             # 4x4x16
        )
        m.classifier[0] = nn.Linear(16 * 4 * 4, 120)
        return _init(m)

    results["e_kernel3"] = run_experiment("e_kernel3", e_kernel3, train_dl, test_dl)
    results["e_kernel7"] = run_experiment("e_kernel7", e_kernel7, train_dl, test_dl)

    # f. number of output channels
    results["f_width0.5"] = run_experiment("f_width0.5", lambda: width_scaled(1), train_dl, test_dl)
    results["f_width2"] = run_experiment("f_width2", lambda: width_scaled(2), train_dl, test_dl)

    # g. activation function
    def g_sigmoid():
        m = LeNet5()
        for i in (1, 4):
            m.features[i] = nn.Sigmoid()
        m.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10),
        )
        return _init(m)

    results["g_sigmoid"] = run_experiment("g_sigmoid", g_sigmoid, train_dl, test_dl)
    results["g_relu"] = run_experiment("g_relu", LeNet5ReLU, train_dl, test_dl)

    # --- best model: combining the two most helpful single-factor findings ---
    # (wider conv layers from f., and removing one FC layer from b.)
    def best():
        m = LeNet5()
        m.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=2), nn.Tanh(),  # 28x28x12
            nn.AvgPool2d(2),                                         # 14x14x12
            nn.Conv2d(12, 32, kernel_size=5),                        # 10x10x32
            nn.Tanh(),
            nn.AvgPool2d(2),                                         # 5x5x32
        )
        m.classifier = nn.Sequential(
            nn.Linear(32 * 5 * 5, 84), nn.Tanh(),
            nn.Linear(84, 10),
        )
        return _init(m)

    results["best_wide_shallowFC"] = run_experiment("best_wide_shallowFC", best, train_dl, test_dl)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    summary = {k: v["test_acc"] for k, v in results.items()}
    print("\n===== Summary (test accuracy %) =====")
    for k in sorted(summary, key=lambda k: -summary[k]):
        print(f"{summary[k]:6.2f}%  {k}")

    with open("A2_Q5_results.json", "w") as f:
        json.dump(results, f, indent=2)

    import csv
    with open("A2_Q5_all_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "test_acc", "train_loss_curve", "test_acc_curve", "time_s"])
        for k, v in results.items():
            writer.writerow([k, v["test_acc"], v["train_loss"], v["test_acc_curve"], v["time_s"]])

    # loss curve of the baseline (for the report figure)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, EPOCHS + 1), base["train_loss"], marker="o", label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("LeNet-5 baseline on Fashion-MNIST (train loss)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("A2_Q5_loss_baseline.png", dpi=150)

    print("\nSaved: A2_Q5_results.json, A2_Q5_all_results.csv, A2_Q5_loss_baseline.png")


if __name__ == "__main__":
    main()
