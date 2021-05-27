import numpy as np
import torch
from os import walk
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from utils import save_pickle


TENSOR_NAMES = ["A.p", "b.p", "C.p"]
STATISTICS = ["QuadraticVariation", "Sobolev", "L2", "Spectral", "Maximum"]
COLORS = ["b", "r", "g", "m"]

STATS_NAME = {
    "SqrtSobolev": r"$||A||_{H^1}$",
    "SqrtSSQ": r"$||A||_F$",
    "Maximum": r"$||A||_{F, \infty}$",
}


def compute_statistics(tensor_list, statistics, factor=0.0):
    """ For a given parameter at a fixed depth, compute the quantities
    specified in 'STATISTICS'
    :param tensor_list: tensor of size (L, shape)
    :return: dictionary containing all the names and the values of the statistics
    """
    dico = {}

    for stats in statistics:

        if stats == "Sobolev" or stats == "SqrtSobolev":
            res = 0.0
            temp = tensor_list[0]
            for param in tensor_list[1:]:
                res += torch.norm(param - temp, p="fro").data.numpy() ** 2
                temp = param
            dico[stats] = np.sqrt(len(tensor_list) * res)

        elif stats == "Holder" or stats == "Increment":
            L = len(tensor_list)
            temp = 0.0
            maxi = 0.0
            for k in range(1, L):
                temp = torch.norm(
                    tensor_list[k] - tensor_list[k - 1], p="fro"
                ).data.numpy()
                if temp > maxi:
                    maxi = temp
            dico[stats] = np.power(L, factor) * maxi

        elif stats == "QuadraticVariation":
            res = 0.0
            temp = tensor_list[0]
            for param in tensor_list[1:]:
                res += torch.norm(param - temp, p="fro").data.numpy() ** 2
                temp = param
            dico[stats] = res

        elif stats in ["L2", "SSQ", "SqrtSSQ", "Root SSQ"]:
            res = 0.0
            for param in tensor_list:
                res += torch.norm(param, p="fro").data.numpy() ** 2
            if stats == "L2":
                dico[stats] = np.sqrt(res / len(tensor_list))
            if stats == "SSQ" or "SqrtSSQ" or "Root SSQ":
                dico[stats] = np.sqrt(res)

        elif stats == "Spectral":
            dico[stats] = np.max(
                [torch.norm(param, p=2).data.numpy() for param in tensor_list]
            )

        elif stats == "Maximum":
            dico[stats] = np.max(
                [torch.norm(param, p="fro").data.numpy() for param in tensor_list]
            )

        elif stats == "cumsum":
            dico[stats] = torch.norm(
                sum([param for param in tensor_list]), p="fro"
            ).data.numpy()

        elif stats == "detrended-cumsum":
            L = len(tensor_list)
            W = [0.0 for k in range(L + 1)]
            for k in range(L):
                W[k + 1] = W[k] + tensor_list[k]

            if L < 10:
                dico[stats] = max(
                    [
                        torch.norm(W[k] - k / L * W[L]).data.numpy()
                        for k in range(1, L + 1)
                    ]
                )
            else:
                temp = 0.0
                maxi = 0.0
                window = max(3, int(0.10 * L))

                for k in range(1, L):
                    if k < window:
                        temp = W[k] - sum(W[0 : 2 * k]) / (2 * k)
                    elif window <= k < L - window:
                        temp = W[k] - sum(W[k - window : k + window]) / (2 * window)
                    else:
                        temp = W[k] - sum(W[2 * k - L : L]) / (2 * (L - k))

                    temp = torch.norm(temp, p="fro").data.numpy()
                    if temp > maxi:
                        maxi = temp

                dico[stats] = maxi

        else:
            raise NotImplementedError(f"This statistic: {stats} is not implemented")

    return dico


def get_folders(experiment_path):
    _, folders, _ = next(walk(experiment_path))
    res = []
    for f in folders:
        _, sub_folders, _ = next(walk(experiment_path + f))
        for g in sub_folders:
            res.append(experiment_path + f + "/" + g + "/")

    return res


def run_regression(path, tensor_names, statistics, factor=0.0):
    _, folders, _ = next(walk(path))
    values = {stat: {name: [] for name in tensor_names} for stat in statistics}
    depths = []
    for f in folders:
        if f != "regression":
            depth = f[6:]
            depth = int(depth)
            depths.append(depth)
            _, _, files = next(walk(path + f + "/weights"))
            for name in tensor_names:
                if name in files:
                    with open(path + f + "/weights/" + name, "rb") as file:
                        t = torch.load(file)
                        temp = compute_statistics(t, statistics, factor)
                        for stat in statistics:
                            values[stat][name].append(temp[stat])

    idx = np.argsort(depths)
    for name in tensor_names:
        for stat in statistics:
            values[stat][name] = np.array(values[stat][name])[idx]

    return np.sort(depths), values


def shared_delta(path):
    _, folders, _ = next(walk(path))
    delta = []
    depths = []
    for f in folders:
        if f != "regression":
            depth = f[6:]
            depth = int(depth)
            depths.append(depth)
            with open(path + f + "/weights/C.p", "rb") as file:
                t = torch.load(file)
                delta.append(t.detach().numpy()[0])

    idx = np.argsort(depths)
    delta = np.array(delta)[idx]
    return delta


def A_or_b_times_C(path, label, delta_type, names):
    _, folders, _ = next(walk(path))
    values = {name: [] for name in names}
    depths = []
    for f in folders:
        if f != "regression":
            depth = f[-4:]
            depth = int(depth[1:] if depth[0] == "_" else depth)
            depths.append(depth)
            with open(path + f + "/weights/" + label + ".p", "rb") as file:
                ab = torch.load(file)
            with open(path + f + "/weights/C.p", "rb") as file:
                c = torch.load(file)

            if delta_type == "multi":
                t = [ab[i] * torch.abs(c[i]) for i in range(len(ab))]

            if delta_type == "shared":
                t = [ab[i] * torch.abs(c[0]) for i in range(len(ab))]

            if delta_type != "none":
                temp = compute_statistics(t, names)
                for name in names:
                    values[name].append(temp[name])

    idx = np.argsort(depths)
    for name in names:
        values[name] = np.array(values[name])[idx]

    return values


def regression_delta_times_AorB(path, statistics, tensor="A", factor=0.0):
    _, folders, _ = next(walk(path))
    values = {stat: [] for stat in statistics}
    depths = []
    for f in folders:
        if f != "regression":
            depth = f[6:]
            depth = int(depth)
            depths.append(depth)
            with open(path + f + "/weights/" + tensor + ".p", "rb") as file:
                AorB = torch.load(file)
            with open(path + f + "/weights/C.p", "rb") as file:
                delta = torch.load(file)

            t = [torch.norm(d) * aorb for aorb, d in zip(AorB, delta)]
            temp = compute_statistics(t, statistics, factor)
            for stat in statistics:
                values[stat].append(temp[stat])

    idx = np.argsort(depths)
    for stat in statistics:
        values[stat] = np.array(values[stat])[idx]

    return np.sort(depths), values


def plot_final_regression(
    depths,
    values,
    tensor_names,
    statistics,
    stats_name,
    colors,
    path,
    save=True,
    reg=True,
    loc="upper right",
    fontsize="small",
    filename="",
):

    if save:
        save_pickle(path + "depths", depths)

    for name in tensor_names:
        for i, stat in enumerate(statistics):
            y = values[stat][name]

            if name == "A.p" and stat == "detrended-cumsum" and filename == "ass2":
                idx = np.argmin(depths < 200)
                lr = LinearRegression().fit(
                    np.log(depths[idx:]).reshape(-1, 1), np.log(y[idx:])
                )

            else:
                lr = LinearRegression().fit(np.log(depths).reshape(-1, 1), np.log(y))

            slope = lr.coef_
            intercept = lr.intercept_
            plt.loglog(
                depths,
                y,
                c=colors[i],
                lw=1,
                label=stats_name[stat]
                + (" : slope = %.1f" % np.round(slope, 1) if reg else ""),
            )
            regy = np.exp(intercept + slope * np.log(depths))
            if reg:
                plt.loglog(depths, regy, "k--", ls="--", lw=0.5)
            if save:
                save_pickle(path + stat + "_" + name, y)

        plt.xlabel("Depth")
        plt.legend(loc=loc, fontsize=fontsize)
        if save:
            plt.savefig(
                path + f"{filename}_{name}ng"
            )  # genius # elon-musk # QI=200 # newton
        else:
            plt.show()
        plt.clf()


def make_regression(net, path):
    colors = ["b", "r", "g", "m"]
    tensor_names = ["A.p", "b.p"] + (
        ["C.p"] if net.delta_type in ["matrix", "multi"] else []
    )
    statistics = ["SqrtSobolev", "SqrtSSQ", "Maximum"]
    stats_name = {
        "SqrtSobolev": r"$||A||_{H^1}$",
        "SqrtSSQ": r"$||A||_F$",
        "Maximum": r"$||A||_{F, \infty}$",
    }
    depths, values = run_regression(path, tensor_names, statistics)
    plot_final_regression(
        depths,
        values,
        tensor_names,
        statistics,
        stats_name,
        colors,
        path + "regression/",
        True,
    )

    if net.delta_type == "shared":
        _, folders, _ = next(walk(path))
        deltas = []
        depths = []
        for f in folders:
            if f != "regression":
                depth = f[6:]
                depth = int(depth)
                depths.append(depth)
                deltas.append(torch.load(path + f + "/weights/C.p").detach().numpy()[0])

        idx = np.argsort(depths)
        depths = np.sort(depths)
        deltas = np.abs(np.array(deltas)[idx])

        lr = LinearRegression().fit(np.log(depths).reshape(-1, 1), np.log(deltas))
        slope = lr.coef_
        intercept = lr.intercept_
        plt.loglog(
            depths,
            deltas,
            c="b",
            label="Shared $\\delta^{(L)}$" + ": slope = %.1f" % np.round(slope, 1),
            lw=1,
        )
        regy = np.exp(intercept + slope * np.log(depths))

        plt.loglog(depths, regy, "k--", lw=0.5)
        plt.plot()
        plt.xlabel("Depth $L$")
        plt.legend(loc="lower left", fontsize="small")
        plt.savefig(path + "/regression_shared_delta.png")
        plt.clf()
