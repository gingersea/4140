"""Train four next-step predictors and roll each one forward by 200 steps."""

import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from utils_q4 import Attention, GRU, LogisticRegression, MyDataset, RNN, data_generator


OUT = Path(__file__).resolve().parent / "results"
SEED = 42
WIDTH = 32
WINDOW = 10
HORIZON = 200
MODELS = {"RNN": RNN, "GRU": GRU, "Attention": Attention, "Logistic": LogisticRegression}
COLORS = {"RNN": "#125781", "GRU": "#b85224", "Attention": "#14775f", "Logistic": "#754c86"}


def fit(name, series, epochs, lr, batch_size):
    torch.manual_seed(SEED)
    model = MODELS[name](WINDOW if name == "Logistic" else WIDTH).to(series.device)
    inputs, targets = MyDataset(series, WINDOW).tensors
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = []
    for _ in range(epochs):
        order = torch.randperm(len(targets), device=series.device)
        total = torch.zeros((), device=series.device)
        for batch in order.split(batch_size):
            optimizer.zero_grad()
            loss = criterion(model(inputs[batch]), targets[batch])
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(batch)
        history.append((total / len(targets)).item())
    return model.eval(), history


@torch.inference_mode()
def forecast(model, seed_window):
    window = seed_window.reshape(1, WINDOW, 1)
    predictions = []
    for _ in range(HORIZON):
        value = model(window)
        predictions.append(value)
        window = torch.cat((window[:, 1:], value.reshape(1, 1, 1)), dim=1)
    return torch.cat(predictions)


def tune_rnn(series):
    # Keep the last 200 training observations out of each trial fit.
    prefix, validation = series[:-HORIZON], series[-HORIZON:]
    trials = []
    for epochs, lr, batch_size in [(30, 0.003, 128), (100, 0.001, 64), (200, 0.001, 64)]:
        model, _ = fit("RNN", prefix, epochs, lr, batch_size)
        error = forecast(model, prefix[-WINDOW:]) - validation
        trials.append(dict(epochs=epochs, lr=lr, batch_size=batch_size,
                           validation_rmse=error.square().mean().sqrt().item()))
    return trials


def timed_forecast(model, seed_window):
    forecast(model, seed_window)  # Warm up before timing batch-size-one inference.
    durations = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = perf_counter()
        prediction = forecast(model, seed_window)
        torch.cuda.synchronize()
        durations.append(1000 * (perf_counter() - start))
    return prediction.cpu().numpy(), median(durations)


def plot_results(x, y, future_x, histories, forecasts):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.45), layout="constrained")
    ax, loss_ax = axes
    ax.scatter(x, y, s=2, color="#125781", linewidths=0)
    ax.set(xlabel="x", ylabel="Observed y", title="Noisy sine samples", xlim=(0, 100))
    for name, history in histories.items():
        loss_ax.semilogy(range(1, len(history) + 1), history, label=name, color=COLORS[name], lw=1.35)
    loss_ax.set(xlabel="Epoch", ylabel="Training MSE", title="One-step training loss")
    loss_ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.savefig(OUT / "data_and_loss.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 3.5), sharex=True, sharey=True, layout="constrained")
    for ax, (name, values) in zip(axes.flat, forecasts.items()):
        ax.plot(future_x, values, color=COLORS[name], label="Prediction", lw=1.65)
        ax.plot(future_x, np.sin(future_x), color="#383838", ls="--", lw=1, label="True sine")
        ax.set(title=name, xlim=(100, 120), xticks=range(100, 121, 5), yticks=[-1, 0, 1])
    axes[0, 0].legend(loc="upper right", ncol=2, fontsize=7, frameon=True)
    for ax in axes[-1]:
        ax.set_xlabel("Future x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    fig.savefig(OUT / "forecasts.png", dpi=220)
    plt.close(fig)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Q4 requires a CUDA-capable GPU.")
    OUT.mkdir(exist_ok=True)
    x, y = data_generator(0.02, SEED)
    series = torch.from_numpy(y).cuda()
    future_x = np.arange(len(y), len(y) + HORIZON) / 10
    trials = tune_rnn(series)
    best = min(trials, key=lambda trial: trial["validation_rmse"])
    print("RNN validation trials:", trials, flush=True)
    metrics = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
               "seed": SEED, "rnn_trials": trials, "models": {}}
    histories, forecasts = {}, {}
    for name in MODELS:
        epochs, lr, batch = 100, 0.003, 64
        if name == "RNN":
            epochs, lr, batch = best["epochs"], best["lr"], best["batch_size"]
        model, histories[name] = fit(name, series, epochs, lr, batch)
        forecasts[name], elapsed = timed_forecast(model, series[-WINDOW:])
        rmse = np.sqrt(np.mean((forecasts[name] - np.sin(future_x)) ** 2))
        metrics["models"][name] = dict(epochs=epochs, lr=lr, batch_size=batch,
                                       final_mse=histories[name][-1], forecast_ms=elapsed,
                                       forecast_rmse=float(rmse))
        print(name, metrics["models"][name], flush=True)
    plot_results(x, y, future_x, histories, forecasts)
    with (OUT / "forecasts.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["x", "true_sine", *forecasts])
        writer.writerows(zip(future_x, np.sin(future_x), *forecasts.values()))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
