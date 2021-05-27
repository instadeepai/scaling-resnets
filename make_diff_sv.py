import os

import numpy as np
import torch
from torch.serialization import save

from cifar_utils import flat, singular_values
from cifar_model import *

D = [8, 11, 12, 14, 16, 20, 24, 28, 33, 42, 50, 65, 80, 100, 121]
_D = [f"{depth:03d}" for depth in D]


# load deltas
deltas_list = [
    torch.load(
        "./scaling/dataset-cifar/act-relu/depth_" + _d + "/weights/deltas.pth",
        map_location=torch.device("cpu"),
    )
    for _d in _D
]

A_list = [
    torch.load(
        "./scaling/dataset-cifar/act-relu/depth_" + _d + "/weights/A.pth",
        map_location=torch.device("cpu"),
    )
    for _d in _D
]

# untrained nets
nets = [eval("Net" + str(_d))() for _d in D]

# sample image
img = torch.zeros((2, 3, 32, 32))

# dummy forward pass to store shapes
_ = [net(img).detach().numpy() for net in nets]

# get input shapes for all nets
all_input_shape_As = [flat(flat(net.input_shape_As)) for net in nets]
all_input_shape_deltas = [flat(net.input_shape_deltas) for net in nets]

A_dict = {d: A for d, A in zip(D, A_list)}
deltas_dict = {d: deltas for d, deltas in zip(D, deltas_list)}
A_shapes_dict = {d: shape for d, shape in zip(D, all_input_shape_As)}
deltas_shapes_dict = {d: shape for d, shape in zip(D, all_input_shape_deltas)}

# compute differences
def get_diffs(weights, shapes):
    diffs_list = [
        {"diff": b - a, "shape": s}
        for a, b, s in zip(weights[:-1], weights[1:], shapes)
        if a.shape == b.shape
    ]
    return {
        "diffs": [x["diff"] for x in diffs_list],
        "shapes": [x["shape"] for x in diffs_list],
    }


# compute top 10 singular values
def compute_sv_on_diffs(diffs, input_shapes):
    return [
        np.sort(singular_values(diff, shape).flatten())[-10:]
        for diff, shape in zip(diffs, input_shapes)
    ]


def save_sv_on_diffs(diffs_sv, path: str):
    if not os.path.exists(path + "weights/"):
        os.makedirs(path + "weights/")
    print("Saving singular values of diffs..")
    torch.save(diffs_sv, path + "weights/diffs_sv.pth")


def _path_with_depth(path: str, _d: str):
    return "./" + path + "/dataset-cifar/act-relu/depth_" + _d + "/"


if __name__ == "__main__":
    for d in D:
        print(f"\nStarting depth {d}..")
        _d = f"{d:03d}"
        PATH = "..."  # add exp directory
        path = _path_with_depth(PATH, _d)

        diffs_A = get_diffs(weights=A_dict[d], shapes=A_shapes_dict[d])
        diffs_deltas = get_diffs(weights=deltas_dict[d], shapes=deltas_shapes_dict[d])

        sv_A = compute_sv_on_diffs(diffs_A["diffs"], diffs_A["shapes"])
        sv_deltas = compute_sv_on_diffs(diffs_deltas["diffs"], diffs_deltas["shapes"])

        save_sv_on_diffs(diffs_sv={"A": sv_A, "deltas": sv_deltas}, path=path)
        print(f"Finished depth {d}..")
