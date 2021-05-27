"""ResNet in PyTorch.

Inspired from https://github.com/kuangliu/pytorch-cifar.
Adapted to our needs.
Original reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )
        else:
            self.shortcut = nn.Sequential()

        self._delta = self.conv2.weight

        self._A = [self.conv1.weight]
        self._SC = [self.shortcut[0].weight] if len(self.shortcut) > 0 else []

    def forward(self, x):
        self._input_shape_A = [list(x.shape)[2:]]
        out = F.relu(self.conv1(x))
        self._input_shape_delta = list(out.shape)[2:]
        out = self.conv2(out)  #  C_k here

        self._input_shape_SC = []
        if len(self.shortcut) > 0:
            self._input_shape_SC.append(list(x.shape)[2:])
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        in_planes = 64
        self.in_planes = 64
        self.factor = 2

        self.conv1 = nn.Conv2d(
            3, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            block, in_planes * self.factor, num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, in_planes * self.factor ** 2, num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, in_planes * self.factor ** 3, num_blocks[3], stride=2
        )
        self.linear = nn.Linear(
            in_planes * self.factor ** 3 * block.expansion, num_classes
        )

        self.deltas = [
            [block._delta for block in layer]
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]
        ]

        self.A = [
            [block._A for block in layer]
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]
        ]

        self.SC = [
            [block._SC for block in layer]
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]
        ]

        self.input_shape_deltas = []
        self.input_shape_As = []
        self.input_shape_SCs = []

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        self.input_shape_deltas = [
            [block._input_shape_delta for block in layer]
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]
        ]

        self.input_shape_As = [
            [block._input_shape_A for block in layer]
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]
        ]

        self.input_shape_SCs = [
            [block._input_shape_SC for block in layer]
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]
        ]
        return out

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


def Net8():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def Net11():
    return ResNet(BasicBlock, [2, 3, 4, 2])


def Net12():
    return ResNet(BasicBlock, [2, 3, 5, 2])


def Net14():
    return ResNet(BasicBlock, [3, 3, 5, 3])


def Net16():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def Net18():
    return ResNet(BasicBlock, [3, 4, 8, 3])


def Net20():
    return ResNet(BasicBlock, [3, 4, 10, 3])


def Net24():
    return ResNet(BasicBlock, [3, 4, 14, 3])


def Net28():
    return ResNet(BasicBlock, [3, 4, 18, 3])


def Net33():
    return ResNet(BasicBlock, [3, 4, 23, 3])


def Net42():
    return ResNet(BasicBlock, [3, 8, 28, 3])


def Net50():
    return ResNet(BasicBlock, [3, 8, 36, 3])


def Net65():
    return ResNet(BasicBlock, [3, 8, 51, 3])


def Net80():
    return ResNet(BasicBlock, [4, 12, 60, 4])


def Net100():
    return ResNet(BasicBlock, [4, 18, 74, 4])


def Net121():
    return ResNet(BasicBlock, [4, 31, 82, 4])
