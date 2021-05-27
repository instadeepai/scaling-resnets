"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import argparse

from cifar_utils import (
    progress_bar,
    cifar_data,
    cifar_experiment_results,
    flat_net_items,
)
from cifar_model import *

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
args = parser.parse_args()

###############

# cuda recommended for conv experiments
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"==> Running on device {device}.")
NUM_TRAIN_EPOCHS = 200

# default path where everything is stored
PATH = "./scaling/dataset-cifar/act-relu/"

# Data
print("==> Preparing data.")
trainloader, testloader = cifar_data()

_depths = [8, 11, 12, 14, 16, 18, 20, 24, 28, 33, 42, 50, 65, 80, 100, 121]


def get_resnet(depth: int = 18):
    assert depth in _depths, f"Depths allowed: {_depths}, unknown value: {depth}."
    return eval(f"Net{depth}")()


def train_and_test(
    net,
    epoch,
    trainloader,
    testloader,
    criterion,
    optimizer,
    scheduler,
    path_with_depth,
):
    print("\nEpoch: %d" % epoch)

    # Train
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * correct / total

        progress_bar(
            batch_idx,
            len(trainloader),
            "(Train) Loss: %.3f | (Train) Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), train_acc, correct, total),
        )

    # Test
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            test_acc = 100.0 * correct / total

            progress_bar(
                batch_idx,
                len(testloader),
                "(Test) Loss: %.3f | (Test) Acc: %.3f%% (%d/%d)"
                % (test_loss / (batch_idx + 1), test_acc, correct, total),
            )

    if epoch == NUM_TRAIN_EPOCHS - 1:
        if not os.path.exists(path_with_depth + f"epoch_{epoch:03d}/"):
            os.makedirs(path_with_depth + f"epoch_{epoch:03d}/")

        print("Saving train and test accuracies..")
        torch.save(
            {"train_acc": train_acc, "test_acc": test_acc, "epoch": epoch},
            path_with_depth + f"epoch_{epoch:03d}/acc.pth",
        )
        print("Saving net items..")
        torch.save(
            flat_net_items(net.module), path_with_depth + f"epoch_{epoch:03d}/items.pth"
        )

    # Scheduler
    scheduler.step()


DEPTHS = _depths

if __name__ == "__main__":

    for depth in DEPTHS:
        path_with_depth = PATH + f"depth_{depth:03d}/"
        if not os.path.exists(path_with_depth):
            os.makedirs(path_with_depth)
        print(f"\n====> All elements will be saved to {path_with_depth}.")

        print(f"\n====> Building model.")
        net = get_resnet(depth)
        net = net.to(device)
        if device == "cuda":
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        print(f"\n====> Created a network of effective depth {depth}.")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        for epoch in range(NUM_TRAIN_EPOCHS):
            train_and_test(
                net,
                epoch,
                trainloader,
                testloader,
                criterion,
                optimizer,
                scheduler,
                path_with_depth,
            )

        cifar_experiment_results(net=net, path=path_with_depth)
