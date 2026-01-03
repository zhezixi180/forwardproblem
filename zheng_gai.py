import os
import json
import time
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import deepxde as dde
import torch


plt.switch_backend("Agg")
ub = 200
rb = 20

BASE_OUTDIR = os.getenv("DDE_OUTDIR", "results_sweep")
DEFAULT_ITERS = int(os.getenv("DDE_ITERS", "50000"))
USE_LBFGS = os.getenv("DDE_USE_LBFGS", "1") == "1"
LR = float(os.getenv("DDE_LR", "0.001"))
GLOBAL_SEED = int(os.getenv("DDE_SEED", "1234"))


def func(t, r):
    x, y = r
    dx_t = 1 / ub * rb * (2.0 * ub * x - 0.04 * ub * x * ub * y)
    dy_t = 1 / ub * rb * (0.02 * ub * x * ub * y - 1.06 * ub * y)
    return dx_t, dy_t


def gen_truedata(n=100):
    t = np.linspace(0, 1, n)
    sol = integrate.solve_ivp(func, (0, 1), (100 / ub, 15 / ub), t_eval=t)
    x_true, y_true = sol.y
    return t.reshape(-1, 1), x_true.reshape(-1, 1), y_true.reshape(-1, 1)


def ode_system(x, y):
    r = y[:, 0:1]
    p = y[:, 1:2]
    dr_t = dde.grad.jacobian(y, x, i=0)
    dp_t = dde.grad.jacobian(y, x, i=1)
    return [
        dr_t - 1 / ub * rb * (2.0 * ub * r - 0.04 * ub * r * ub * p),
        dp_t - 1 / ub * rb * (0.02 * r * ub * p * ub - 1.06 * p * ub),
    ]


def input_transform(t):
    return torch.cat(
        [
            t,
            torch.sin(t),
            torch.sin(2 * t),
            torch.sin(3 * t),
            torch.sin(4 * t),
            torch.sin(5 * t),
            torch.sin(6 * t),
        ],
        dim=1,
    )


def output_transform(t, y):
    y1 = y[:, 0:1]
    y2 = y[:, 1:2]
    return torch.cat(
        [y1 * torch.tanh(t) + 100 / ub, y2 * torch.tanh(t) + 15 / ub],
        dim=1,
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def summarize_loss(losshistory):
    train = np.array(losshistory.loss_train, dtype=float)
    test = np.array(losshistory.loss_test, dtype=float) if losshistory.loss_test else None

    train_sum = train.sum(axis=1) if train.ndim == 2 else train
    test_sum = test.sum(axis=1) if (test is not None and test.ndim == 2) else test

    steps = getattr(losshistory, "steps", None)
    if steps is None or len(steps) != len(train_sum):
        steps = np.arange(len(train_sum))

    return np.array(steps), train_sum, (np.array(test_sum) if test_sum is not None else None)


def save_loss_plot(losshistory, outdir):
    steps, train_sum, test_sum = summarize_loss(losshistory)

    fig = plt.figure()
    plt.semilogy(steps, train_sum, label="Train loss")
    if test_sum is not None:
        plt.semilogy(steps, test_sum, label="Test loss")
    plt.xlabel("# Steps")
    plt.ylabel("Loss (sum)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "loss.png"), dpi=200)
    plt.close(fig)

    np.save(os.path.join(outdir, "loss_steps.npy"), steps)
    np.save(os.path.join(outdir, "loss_train_sum.npy"), train_sum)
    if test_sum is not None:
        np.save(os.path.join(outdir, "loss_test_sum.npy"), test_sum)


def save_pred_plot(t, x_true, y_true, x_pred, y_pred, outdir):
    fig = plt.figure()
    plt.plot(t, x_true, color="black", label="x_true")
    plt.plot(t, y_true, color="blue", label="y_true")
    plt.plot(t, x_pred, color="red", linestyle="dashed", label="x_pred")
    plt.plot(t, y_pred, color="orange", linestyle="dashed", label="y_pred")
    plt.xlabel("t")
    plt.ylabel("population (normalized)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "pred_vs_true.png"), dpi=200)
    plt.close(fig)


def save_phase_plot(x_true, y_true, x_pred, y_pred, outdir):
    fig = plt.figure()
    plt.plot(x_true, y_true, label="true")
    plt.plot(x_pred, y_pred, linestyle="dashed", label="pred")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "phase.png"), dpi=200)
    plt.close(fig)


def compute_metrics(x_true, y_true, x_pred, y_pred):
    mse_x = float(np.mean((x_pred - x_true) ** 2))
    mse_y = float(np.mean((y_pred - y_true) ** 2))
    mse_all = float(np.mean(np.concatenate([(x_pred - x_true), (y_pred - y_true)], axis=1) ** 2))
    maxae_x = float(np.max(np.abs(x_pred - x_true)))
    maxae_y = float(np.max(np.abs(y_pred - y_true)))
    return {
        "mse_x": mse_x,
        "mse_y": mse_y,
        "mse_all": mse_all,
        "maxae_x": maxae_x,
        "maxae_y": maxae_y,
    }


def run_one_experiment(cfg: dict):
    outdir = os.path.join(BASE_OUTDIR, cfg["name"])
    ensure_dir(outdir)

    dde.config.set_random_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    geom = dde.geometry.TimeDomain(0, 1.0)
    train_distribution = cfg.get("train_distribution", "pseudo")
    data = dde.data.PDE(
        geom,
        ode_system,
        [],
        cfg["num_domain"],
        cfg["num_boundary"],
        num_test=cfg["num_test"],
        train_distribution=train_distribution,
    )

    net = dde.nn.FNN(cfg["layer_size"], cfg["activation"], cfg["initializer"])
    net.apply_feature_transform(input_transform)
    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    t0 = time.time()
    model.compile("adam", lr=cfg["lr"])
    losshistory, train_state = model.train(iterations=cfg["iterations"])
    if cfg["use_lbfgs"]:
        model.compile("L-BFGS")
        losshistory, train_state = model.train()
    t1 = time.time()

    save_loss_plot(losshistory, outdir)

    t, x_true, y_true = gen_truedata(n=200)
    pred = model.predict(t)
    x_pred = pred[:, 0:1]
    y_pred = pred[:, 1:2]

    save_pred_plot(t, x_true, y_true, x_pred, y_pred, outdir)
    save_phase_plot(x_true, y_true, x_pred, y_pred, outdir)

    metrics = compute_metrics(x_true, y_true, x_pred, y_pred)
    metrics["train_seconds"] = float(t1 - t0)

    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def make_experiments():
    archs = [
        {"arch_name": "FNN_64x6", "layer_size": [7] + [64] * 6 + [2], "activation": "tanh"},
        {"arch_name": "FNN_128x3", "layer_size": [7] + [128] * 3 + [2], "activation": "tanh"},
        {"arch_name": "FNN_32x10", "layer_size": [7] + [32] * 10 + [2], "activation": "tanh"},
        {"arch_name": "FNN_64x6_silu", "layer_size": [7] + [64] * 6 + [2], "activation": "silu"},
    ]

    samplings = [
        {"num_domain": 500, "num_boundary": 2},
        {"num_domain": 3000, "num_boundary": 2},
        {"num_domain": 10000, "num_boundary": 2},
        {"num_domain": 3000, "num_boundary": 10},
    ]

    exps = []
    for a in archs:
        for s in samplings:
            name = f'{a["arch_name"]}__Nd{s["num_domain"]}__Nb{s["num_boundary"]}'
            exps.append(
                {
                    "name": name,
                    "layer_size": a["layer_size"],
                    "activation": a["activation"],
                    "initializer": "Glorot normal",
                    "num_domain": s["num_domain"],
                    "num_boundary": s["num_boundary"],
                    "num_test": 3000,
                    "iterations": DEFAULT_ITERS,
                    "lr": LR,
                    "use_lbfgs": USE_LBFGS,
                    "seed": GLOBAL_SEED,
                    "train_distribution": "pseudo",
                }
            )
    return exps


def write_summary_csv(rows, outpath):
    ensure_dir(os.path.dirname(outpath))
    header = [
        "name",
        "mse_x",
        "mse_y",
        "mse_all",
        "maxae_x",
        "maxae_y",
        "train_seconds",
    ]
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def main():
    ensure_dir(BASE_OUTDIR)
    exps = make_experiments()

    all_rows = []
    for cfg in exps:
        print(f"\n[RUN] {cfg['name']}")
        metrics = run_one_experiment(cfg)
        row = {"name": cfg["name"], **metrics}
        all_rows.append(row)
        print("[DONE]", row)

    write_summary_csv(all_rows, os.path.join(BASE_OUTDIR, "summary.csv"))
    print(f"\nAll done. Summary saved to: {os.path.join(BASE_OUTDIR, 'summary.csv')}")
    print(f"Each experiment has its own folder under: {BASE_OUTDIR}")


if __name__ == "__main__":
    main()
