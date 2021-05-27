import numpy as np
import matplotlib.pyplot as plt
from scaling_model import FinalResNet
from utils import load_pickle, save_pickle
from os import walk
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

import seaborn as sns

sns.set(
    font="Helvetica",
    rc={
        "axes.axisbelow": False,
        "axes.edgecolor": "lightgrey",
        "axes.facecolor": "None",
        "axes.grid": False,
        "axes.labelcolor": "dimgrey",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.facecolor": "white",
        "lines.solid_capstyle": "round",
        "patch.edgecolor": "w",
        "patch.force_edgecolor": True,
        "text.color": "dimgrey",
        "xtick.bottom": False,
        "xtick.color": "dimgrey",
        "xtick.direction": "out",
        "xtick.top": False,
        "ytick.color": "dimgrey",
        "ytick.direction": "out",
        "ytick.left": False,
        "ytick.right": False,
    },
)


def load_delta(path, activation):
    depths = []
    weights = {}
    _, folders, _ = next(walk(path))
    for f in folders:
        if f != "regression" and f != ".ipynb_checkpoints":
            depth = f[6:]
            depth = int(depth)
            depths.append(depth)
            with open(path + f + "/weights/C.p", "rb") as file:
                delta = torch.load(file)
                if activation == "relu":
                    weights[depth] = np.array([np.sign(d.data.numpy()) for d in delta])
                else:
                    weights[depth] = np.array([d.data.numpy() for d in delta])

    return np.sort(depths), weights


def load_dataset(path, dim):
    with open(path + "train_set.p", "rb") as file:
        train_set = torch.load(file)

    return train_set[:, :dim], train_set[:, dim:]


def detrended_weight():
    path = "./scaling/dataset-ODE/act-relu/delta-multi/2021-01-07-15-07/"
    depth = 6319  # 9100
    index = (7, 7)
    # Load entries
    A = torch.load(path + f"depth_{depth}/weights/A.p")
    A = [a.data.numpy() for a in A]
    deltas = torch.load(path + f"depth_{depth}/weights/C.p")
    deltas = [d.data.numpy() for d in deltas]

    AD = [np.power(depth, 0.5) * np.abs(d) * a[index] for d, a in zip(deltas, A)]

    detrended = [0.0 for k in range(depth + 1)]
    maxi = 0.0
    window = 100

    for k in range(1, depth):
        if k < window:
            detrended[k] = AD[k] - sum(AD[0 : 2 * k]) / (2 * k)
        elif window <= k < depth - window:
            detrended[k] = AD[k] - sum(AD[k - window : k + window]) / (2 * window)
        else:
            detrended[k] = AD[k] - sum(AD[2 * k - depth : depth]) / (2 * (depth - k))

    plt.plot(np.arange(depth + 1), detrended, "ko", markersize=0.5)
    plt.show()
    plt.clf()


def coupling(
    path,
    activation,
    tensor,
    delta_type,
    beta,
    n_times=500,
    offset=5,
    index=(0, 0),
    plot=False,
):
    dim = 10

    depths = []
    weights = {}
    _, folders, _ = next(walk(path))
    for f in folders:
        if f != "regression" and f != ".ipynb_checkpoints":
            depth = f[6:]
            depth = int(depth)
            depths.append(depth)
            _, _, files = next(walk(path + f + "/weights"))
            with open(path + f + "/weights/" + tensor + ".p", "rb") as file:
                AorB = torch.load(file)
            with open(path + f + "/weights/C.p", "rb") as file:
                delta = torch.load(file)

            if len(delta) == 1 and delta_type == "shared" and activation == "tanh":
                weights[depth] = np.array([aorb.data.numpy() for aorb in AorB])
            elif delta_type == "multi" and activation == "relu":
                weights[depth] = np.array(
                    [
                        np.abs(d.data.numpy()) * aorb.data.numpy()
                        for d, aorb in zip(delta, AorB)
                    ]
                )
            else:
                raise ValueError("This case is not treated, please check it.")

    depths = np.sort(depths)
    depths = depths[depths > n_times]
    times = np.arange(n_times + 1)
    rescaled_cumsum = {t: [] for t in times}  # tensor T x N_depths x d x d

    for depth in depths:
        W = np.zeros((dim, dim)) if tensor == "A" else np.zeros(dim)
        rescaled_cumsum[0].append(W)
        i = 1
        for k in range(depth):
            W = W + weights[depth][k]
            if k + 1 <= depth * times[i] / n_times < k + 2:
                rescaled_cumsum[times[i]].append(np.power(depth, beta - 1) * W)
                i += 1

    integral_bar = [np.mean(rescaled_cumsum[t][-8:-4], axis=0) for t in times]
    print("Integral bar shape: ", np.array(integral_bar).shape)
    weights_bar = {}

    for depth in depths:
        weights_bar[depth] = (
            np.zeros((depth, dim, dim)) if tensor == "A" else np.zeros((depth, dim))
        )
        for k in range(depth):
            idx = int(n_times * k * 1.0 / depth)
            weights_bar[depth][k] = (
                np.power(depth, -beta)
                * n_times
                * (integral_bar[idx + 1] - integral_bar[idx])
            )

    if plot:
        L = depths[-7]
        ts = np.array([x[index] for x in weights[L]])
        ts_bar = np.array([x[index] for x in weights_bar[L]])
        ax = plt.gca()
        locs = (
            {"A": "upper right", "b": "upper left"}
            if activation == "tanh"
            else {"A": "lower left", "b": "upper left"}
        )
        label_trend = r"$\overline{A}$" if tensor == "A" else r"$\overline{b}$"
        plt.plot(
            np.arange(L),
            np.power(L, beta) * ts_bar,
            color="#0D8700",
            label=r"Trend part " + label_trend,
            lw=2,
        )
        label_noise = r"$W^A$" if tensor == "A" else r"$W^b$"
        plt.plot(
            np.arange(L),
            np.power(L, beta - 1) * np.cumsum(ts - ts_bar),
            color="#7A0087",
            label=f"Noise part " + label_noise,
            lw=2,
        )
        plt.legend(loc=locs[tensor], fontsize=16)
        plt.savefig(
            f"figs/act-{activation}-delta-{delta_type}-{index}-decomposition-{tensor}.png"
        )
        plt.clf()

    return weights, weights_bar


def accuracy_trained_weights(
    depths, params, params_denoised, train_set, labels, delta_type, activation
):
    loss = {}
    loss_denoised = {}
    criterion = nn.MSELoss()
    for depth in depths:
        net = FinalResNet(
            dim=10,
            num_layers=depth,
            delta_type=delta_type,
            initial_sd=0.0,
            activation=activation,
            reg_a_type=None,
            reg_b_type=None,
            reg_c_type=None,
            lamb=0.0,
            dim_out=None,
        )
        net.init_from_data(
            params["A"][depth], params["b"][depth], params["delta"][depth]
        )
        loss[depth] = criterion(net(train_set), labels)

        net_denoised = FinalResNet(
            dim=10,
            num_layers=depth,
            delta_type=delta_type,
            initial_sd=0.0,
            activation=activation,
            reg_a_type=None,
            reg_b_type=None,
            reg_c_type=None,
            lamb=0.0,
            dim_out=None,
        )
        net_denoised.init_from_data(
            params_denoised["A"][depth],
            params_denoised["b"][depth],
            params_denoised["delta"][depth],
        )
        loss_denoised[depth] = criterion(net_denoised(train_set), labels)

    return loss, loss_denoised


if __name__ == "__main__":

    path = "..."  # add path of tanh
    delta_type = "shared"
    activation = "tanh"
    dim = 10
    n_times = 500
    offset = 2
    beta = 0.2
    index_A = (9, 7)
    index_b = 5

    ### Uncomment this for relu (and comment the block above). ###
    # path = "..." # add path of relu
    # delta_type = "multi"
    # activation = "relu"
    # dim = 10
    # n_times = 500
    # offset = 2
    # beta = 0.5
    # index_A = (7, 7)
    # index_b = 6

    depths, delta = load_delta(path, activation)
    depths = depths[depths > n_times]
    inputs, labels = load_dataset(path, dim)

    A, A_denoised = coupling(
        path=path,
        activation=activation,
        tensor="A",
        delta_type=delta_type,
        beta=beta,
        n_times=n_times,
        offset=offset,
        index=index_A,
        plot=True,
    )

    b, b_denoised = coupling(
        path=path,
        activation=activation,
        tensor="b",
        delta_type=delta_type,
        beta=beta,
        n_times=n_times,
        offset=offset,
        index=index_b,
        plot=True,
    )

    params = {"A": A, "b": b, "delta": delta}
    params_denoised = {"A": A_denoised, "b": b_denoised, "delta": delta}

    loss, loss_denoised = accuracy_trained_weights(
        depths=depths,
        params=params,
        params_denoised=params_denoised,
        train_set=inputs,
        labels=labels,
        delta_type=delta_type,
        activation=activation,
    )

    save_pickle("./figs/loss.p", loss)
    save_pickle("./figs/loss_denoised.p", loss_denoised)

    loss = load_pickle("./figs/loss.p")
    loss_denoised = load_pickle("./figs/loss_denoised.p")

    ax = plt.gca()
    depths = depths[:-4]
    xmin = int(0.9 * min(depths))
    xmax = int(1.1 * max(depths))
    plt.plot(
        depths,
        [loss[depth] for depth in depths],
        c="k",
        label="Training loss",
        lw=3,
        ls="--",
    )
    plt.plot(
        depths,
        [loss_denoised[depth] for depth in depths],
        c="#0D8700",
        label="With denoised weights",
        lw=3,
        ls="--",
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(7 * 10 ** (-4), 4 * 10 ** (-2))
    plt.fill_between(x=[xmin, xmax], y1=10 ** (-2), color="#fefef2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend(loc="upper right", fontsize=16)
    plt.savefig(f"figs/act-{activation}-delta-{delta_type}-loss_denoised.png")
    plt.clf()
