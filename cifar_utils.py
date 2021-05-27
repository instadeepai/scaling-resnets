"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
    
Inspired from https://github.com/kuangliu/pytorch-cifar
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen("stty size", "r").read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def cifar_data(path="./data"):
    # standard processing
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    # no data augmentation for the test
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    return trainloader, testloader


def singular_values(kernel, input_shape):
    k = kernel.detach().cpu().numpy().transpose(2, 3, 0, 1)
    transforms = np.fft.fft2(k, input_shape, axes=[0, 1])
    return np.linalg.svd(transforms, compute_uv=False)


def flat(_list):
    return [x for sublist in _list for x in sublist]


def flat_net_items(net):
    return {
        "input_shape_deltas": flat(net.input_shape_deltas),
        "input_shape_As": flat(flat(net.input_shape_As)),
        "input_shape_SCs": flat(
            [sc for sc in flat(net.input_shape_SCs) if len(sc) > 0]
        ),
        "deltas": flat(net.deltas),
        "A": flat(flat(net.A)),
        "SC": flat([sc for sc in flat(net.SC) if len(sc) > 0]),
    }


def compute_net_sv(items_dict):
    # get the max 10 singular values
    def sv(k: str, s: str):
        return [
            np.sort(singular_values(kernel, shape).flatten())[-10:]
            for kernel, shape in zip(items_dict[k], items_dict[s])
        ]

    return {
        "sv_A": sv("A", "input_shape_As"),
        "sv_deltas": sv("deltas", "input_shape_deltas"),
        "sv_SC": sv("SC", "input_shape_SCs"),
    }


def cifar_experiment_results(net, path):
    items = flat_net_items(net.module)
    sv_dict = compute_net_sv(items)

    if not os.path.exists(path + "weights/"):
        os.makedirs(path + "weights/")
    print("Saving weights..")
    torch.save(items["A"], path + "weights/A.pth")
    torch.save(items["deltas"], path + "weights/deltas.pth")
    torch.save(items["SC"], path + "weights/SC.pth")

    print("Saving spectral norms..")
    if not os.path.exists(path + "spectral/"):
        os.makedirs(path + "spectral/")

    torch.save(sv_dict, path + "spectral/sv_dict.pth")
