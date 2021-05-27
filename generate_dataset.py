import numpy as np

# DL packages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import Compose, ToTensor, Normalize


def generate_data_ODE(d, f, g, noise, train_size, test_size, n_steps=100, scale=True):
    """ 
    Generate samples (x, y), where if z'(t) = tanh(f(t)*z(t) + g(t)), then 
    y = z(1) + x + e, x ~ U(-1, 1) and e is an isotropic noise
    """
    input_ = torch.FloatTensor(train_size + test_size, d).uniform_(-1, 1)
    temp = input_
    for t in range(n_steps):
        temp = temp + np.power(n_steps, -0.5) * torch.tanh(
            f(5 * t * np.pi / n_steps) * temp + g(5 * t * np.pi / n_steps)
        )

    output_ = temp + noise * torch.randn(train_size + test_size, d)

    if scale:
        input_ /= np.linalg.norm(input_, axis=1, keepdims=True)
        output_ /= np.linalg.norm(output_, axis=1, keepdims=True)

    train_set = torch.cat([input_[:train_size], output_[:train_size]], dim=1)
    test_set = torch.cat([input_[train_size:], output_[train_size:]], dim=1)

    return train_set, test_set


def generate_mnist_resnet(batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=data_transform
    )
    mnist_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=data_transform
    )

    conv1 = nn.Conv2d(kernel_size=5, stride=2, in_channels=1, out_channels=1)
    conv2 = nn.Conv2d(kernel_size=3, stride=2, in_channels=1, out_channels=1)

    train_tensor = conv2(
        conv1(mnist_trainset.train_data.unsqueeze(dim=1).float())
    ).view(mnist_trainset.train_data.shape[0], -1)
    train_tensor /= torch.norm(train_tensor, dim=1, keepdim=True)
    train_labels = torch.zeros_like(train_tensor)
    train_labels[np.arange(train_labels.shape[0]), mnist_trainset.train_labels] = 1.0
    train_set = torch.cat([train_tensor, train_labels], dim=1)

    test_tensor = conv2(conv1(mnist_testset.test_data.unsqueeze(dim=1).float())).view(
        mnist_testset.test_data.shape[0], -1
    )
    test_tensor /= torch.norm(test_tensor, dim=1, keepdim=True)

    test_labels = torch.zeros_like(test_tensor)
    test_labels[np.arange(test_labels.shape[0]), mnist_testset.test_labels] = 1.0
    test_set = torch.cat([test_tensor, test_labels], dim=1)

    return train_set.detach(), test_set.detach()
