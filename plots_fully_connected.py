import numpy as np
import matplotlib.pyplot as plt
from regression import run_regression, regression_delta_times_AorB
import torch
from sklearn.linear_model import LinearRegression
import seaborn as sns
from utils import custom_round

sns.set(
    font="DejaVu Sans",
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


def scaling_tanh_shared_delta(
    path_toy, path_mnist, save=False, tensor_names=["A.p", "C.p"]
):
    colors = ["red", "red"]
    linestyles = ["dotted", "solid"]
    locs = {"A.p": "center right", "b.p": "upper right", "C.p": "lower left"}

    values = {}
    depths, values_toy = run_regression(path_toy, tensor_names, ["Maximum"])
    values["Synthetic"] = values_toy["Maximum"]

    depths, values_mnist = run_regression(path_mnist, tensor_names, ["Maximum"])
    values["MNIST"] = values_mnist["Maximum"]

    for j, name in enumerate(tensor_names):
        fig = plt.figure()
        ax = plt.gca()
        for i, dataset in enumerate(["Synthetic", "MNIST"]):
            y = values[dataset][name]
            lr = LinearRegression().fit(np.log(depths).reshape(-1, 1), np.log(y))
            slope = lr.coef_
            beta = custom_round(float(-slope), 1)
            label = (
                r"Max norm: $\alpha={}$".format(beta)
                if name == "C.p"
                else r"Max norm: $\beta={}$".format(beta)
            )
            plt.loglog(
                depths, y, c=colors[j], lw=3, linestyle=linestyles[i], label=label
            )

        plt.legend(loc=locs[name], fontsize=18)
        if save:
            plt.savefig("./figs/scaling_tanh_shared_delta_" + name[0] + ".png")
        else:
            plt.show()
        plt.clf()


def hypothesis_tanh_shared_delta(path_toy, path_mnist, save=False, tensor_name=["A.p"]):
    stats = ["Increment", "Root SSQ"]
    linestyles = ["dotted", "solid"]
    colors = ["orange", "magenta"]

    values = {}
    depths, values_toy = run_regression(path_toy, tensor_name, stats, factor=0.2)
    values["Synthetic"] = values_toy

    depths, values_mnist = run_regression(path_mnist, tensor_name, stats)
    values["MNIST"] = values_mnist

    fig = plt.figure()
    ax = plt.gca()

    for i, dataset in enumerate(["Synthetic", "MNIST"]):
        for j, name in enumerate(stats):
            y = values[dataset][name][tensor_name[0]]
            lr = LinearRegression().fit(np.log(depths).reshape(-1, 1), np.log(y))
            slope = lr.coef_
            beta = custom_round(float(slope), 1)
            plt.loglog(
                depths,
                y,
                c=colors[j],
                lw=3,
                linestyle=linestyles[i],
                label=name + r": ${}$".format(beta),
            )

    if tensor_name[0] == "A.p":
        ax.set_ylim(10 ** (-2), 5 * 10 ** 3)
    if tensor_name[0] == "b.p":
        ax.set_ylim(10 ** (-3), 5 * 10 ** 2)

    plt.legend(loc="upper right", ncol=2, fontsize=14)
    if save:
        plt.savefig("./figs/hypothesis_tanh_shared_delta_" + tensor_name[0][0] + ".png")
    else:
        plt.show()
    plt.clf()


def scaling_relu_scalar_delta(path_toy, path_mnist, save=False, tensor_name="A"):
    linestyles = ["dotted", "solid"]
    color = "red"

    values = {}
    depths, values_toy = regression_delta_times_AorB(path_toy, ["Maximum"], tensor_name)
    values["Synthetic"] = values_toy

    depths, values_mnist = regression_delta_times_AorB(
        path_mnist, ["Maximum"], tensor_name
    )
    values["MNIST"] = values_mnist

    fig = plt.figure()
    ax = plt.gca()

    for i, dataset in enumerate(["Synthetic", "MNIST"]):
        y = values[dataset]["Maximum"]
        lr = LinearRegression().fit(np.log(depths).reshape(-1, 1), np.log(y))
        slope = lr.coef_
        intercept = lr.intercept_
        beta = custom_round(float(-slope), 1)
        plt.loglog(
            depths,
            y,
            c=color,
            lw=3,
            linestyle=linestyles[i],
            label="Max norm: " + r"$\beta={}$".format(beta),
        )

    plt.legend(loc="upper right", fontsize=16)
    if save:
        plt.savefig("./figs/scaling_relu_scalar_delta_" + tensor_name + ".png")
    else:
        plt.show()
    plt.clf()


def individual_delta_relu_scaling_delta(path_toy, depth, save=False):
    delta = torch.load(path_toy + f"/depth_{depth}/weights/C.p")

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(np.arange(9100), [x.data.numpy() for x in delta], lw=0.08, c="b")

    if save:
        plt.savefig("./figs/individual_delta_relu_scaling_delta.png")
    else:
        plt.show()
    plt.clf()


def hypothesis_relu_delta_scalar(path_toy, path_mnist, save=False, tensor_name="A"):
    stats = ["Increment", "Root SSQ"]
    linestyles = ["dotted", "solid"]
    colors = ["orange", "magenta"]

    values = {}
    depths, values_toy = regression_delta_times_AorB(
        path_toy, stats, tensor_name, factor=0.5
    )
    values["Synthetic"] = values_toy

    depths, values_mnist = regression_delta_times_AorB(
        path_mnist, stats, tensor_name, factor=0.4
    )
    values["MNIST"] = values_mnist

    fig = plt.figure()
    ax = plt.gca()

    for i, dataset in enumerate(["Synthetic", "MNIST"]):
        for j, name in enumerate(stats):
            y = values[dataset][name]
            lr = LinearRegression().fit(np.log(depths).reshape(-1, 1), np.log(y))
            slope = lr.coef_
            intercept = lr.intercept_
            beta = custom_round(float(slope), 1)
            plt.loglog(
                depths,
                y,
                c=colors[j],
                lw=3,
                linestyle=linestyles[i],
                label=name + r": ${}$".format(beta),
            )

    if tensor_name == "A":
        ax.set_ylim(1, 5 * 10 ** 2)
    if tensor_name == "b":
        ax.set_ylim(10 ** (-1), 5 * 10 ** 1)

    plt.legend(loc="upper right", ncol=2, fontsize=14)
    if save:
        plt.savefig("./figs/hypothesis_relu_delta_scalar_" + tensor_name + ".png")
    else:
        plt.show()
    plt.clf()


def scaling_relu_matrix_shared(path_toy, path_mnist, save=False, tensor_name="A"):
    linestyles = ["dotted", "solid"]
    color = "red"

    values = {}
    depths, values_toy = regression_delta_times_AorB(path_toy, ["Maximum"], tensor_name)
    values["Synthetic"] = values_toy

    depths, values_mnist = regression_delta_times_AorB(
        path_mnist, ["Maximum"], tensor_name
    )
    values["MNIST"] = values_mnist

    depths = depths[:-3]

    fig = plt.figure()
    ax = plt.gca()

    for i, dataset in enumerate(["Synthetic", "MNIST"]):
        y = values[dataset]["Maximum"][:-3]
        lr = LinearRegression().fit(np.log(depths).reshape(-1, 1), np.log(y))
        slope = lr.coef_
        beta = custom_round(float(slope), 1)
        plt.loglog(
            depths,
            y,
            c=color,
            lw=3,
            linestyle=linestyles[i],
            label="Max norm: " + r"$\beta={}$".format(beta),
        )

    plt.legend(loc="lower left", fontsize=14)
    if save:
        plt.savefig("./figs/scaling_relu_matrix_shared_" + tensor_name + ".png")
    else:
        plt.show()
    plt.clf()


if __name__ == "__main__":
    tensor_name = "A.p"

    path_toy = "..."  #  Add path of synthetic dataset experiments
    path_mnist = "..."  #  Add path of synthetic dataset experiments

    hypothesis_tanh_shared_delta(
        path_toy, path_mnist, save=True, tensor_name=[tensor_name]
    )

    hypothesis_relu_delta_scalar(
        path_toy, path_mnist, save=True, tensor_name=tensor_name[0]
    )

    scaling_relu_matrix_shared(
        path_toy, path_mnist, save=True, tensor_name=tensor_name[0]
    )

    depth = 9100  # depth from synthetic dataset exp
    individual_delta_relu_scaling_delta(path_toy, depth=depth, save=True)

    scaling_relu_scalar_delta(
        path_toy, path_mnist, save=True, tensor_name=tensor_name[0]
    )

    scaling_tanh_shared_delta(
        path_toy, path_mnist, save=True, tensor_names=["A.p", "C.p"]
    )
