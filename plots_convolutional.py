import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Any
from sklearn.linear_model import LinearRegression
from utils import custom_round

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
sns.set_context(
    "notebook", rc={"font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18}
)

# load
PATH = "./scaling/"
_depths = [8, 11, 12, 14, 16, 20, 24, 28, 33, 42, 50, 65, 80, 100, 121]
depths = [f"{depth:03d}" for depth in _depths]
sv = [
    torch.load(PATH + f"dataset-cifar/act-relu/depth_" + d + "/spectral/sv_dict.pth")
    for d in depths
]


def plot_max_norms(
    tensor_lists: List[List[torch.tensor]], _depths, path, beta=True, offset: int = 0
):
    fig = plt.figure()
    ax = plt.gca()

    ax.scatter(
        sum([[x] * len(t_l) for x, t_l in zip(_depths, tensor_lists)], []),
        sum([[np.max(x) for x in tensor_list] for tensor_list in tensor_lists], []),
        s=1,
        c="blue",
    )

    max_values = [
        np.max([np.max(x) for x in tensor_list]) for tensor_list in tensor_lists
    ]
    lr_max = LinearRegression().fit(
        np.log(_depths[offset:]).reshape(-1, 1),
        np.log(np.array(max_values)[offset:]).reshape(-1, 1),
    )
    beta_max = custom_round(-float(lr_max.coef_), 1)
    label = (
        r"Max norm: $\beta={}$".format(beta_max)
        if beta
        else r"Max norm: $\alpha={}$".format(beta_max)
    )
    ax.plot(
        [x for x in _depths],
        max_values,
        linewidth=3,
        label=label,
        c="red",
        linestyle="solid",
    )

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([0, 10])
    plt.legend(loc="lower left", fontsize=16)
    plt.show(path)


#############################################################


# load
PATH = "./scaling/"
_depths = [8, 11, 12, 14, 16, 20, 24, 28, 33, 42, 50, 65, 80, 100, 121]
depths = [f"{depth:03d}" for depth in _depths]
sv = [
    torch.load(PATH + f"dataset-cifar/act-relu/depth_" + d + "/spectral/sv_dict.pth")
    for d in depths
]

diffs_sv = [
    torch.load(PATH + f"dataset-cifar/act-relu/depth_" + d + "/weights/diffs_sv.pth")
    for d in depths
]


def plot_rssq_and_increments(
    tensor_lists: List[List[torch.tensor]],
    tensor_diff_lists: List[List[torch.tensor]],
    _depths,
    beta,
    path,
    offset: int = 0,
):
    # we take the max before doing any op because we have the 10 largest singular values!

    fig = plt.figure()
    ax = plt.gca()

    rssq_values = [
        np.sqrt(np.sum([np.max(x) ** 2 for x in tensor_list]))
        for tensor_list in tensor_lists
    ]
    lr_rssq = LinearRegression().fit(
        np.log(_depths[offset:]).reshape(-1, 1),
        np.log(np.array(rssq_values)[offset:]).reshape(-1, 1),
    )
    beta_rssq = custom_round(-float(lr_rssq.coef_), 1)
    ax.plot(
        [x for x in _depths],
        rssq_values,
        linewidth=3,
        label=r"Root SSQ: ${}$".format(beta_rssq),
        c="magenta",
        linestyle="solid",
    )
    increment_values = [
        (_depths[i] ** beta) * np.max([np.max(x) for x in tensor_list])
        for i, tensor_list in enumerate(tensor_diff_lists)
    ]
    lr_increment = LinearRegression().fit(
        np.log(_depths[offset:]).reshape(-1, 1),
        np.log(np.array(increment_values)[offset:]).reshape(-1, 1),
    )
    beta_increment = custom_round(-float(lr_increment.coef_), 1)
    ax.plot(
        [x for x in _depths],
        increment_values,
        linewidth=3,
        label=r"Increment: ${}$".format(beta_increment),
        c="orange",
        linestyle="solid",
    )

    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.legend(loc="right", fontsize=16)
    plt.savefig(path)
