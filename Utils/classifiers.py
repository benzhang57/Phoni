import torch
from torch import nn


class Cifar10_Classifier(nn.Module):
    def __init__(self, num_step=5):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            #nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer6 = View(-1, 256 * 4 * 4 * 16)
        self.layer7 = nn.Sequential(
            nn.Linear(4096 * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.layer8 = nn.Sequential(
            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer9 = nn.Sequential(
            nn.Linear(64, 10),
        )
        self.layers = []
        self.layers.append(self.layer0)
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)
        self.layers.append(self.layer3)
        self.layers.append(self.layer4)
        self.layers.append(self.layer5)
        self.layers.append(self.layer6)
        self.layers.append(self.layer7)
        self.layers.append(self.layer8)
        self.layers.append(self.layer9)
        self.layers = self.layers[num_step+1:10]

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

class Mnist_Classifier(nn.Module):
    def __init__(self, num_step=0):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            # nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            # nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            # nn.ReLU(),
        )
        self.layer3 = View(-1, 4 * 28 * 28)
        self.layer4 = nn.Sequential(
            nn.Linear(4 * 28 * 28, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            #nn.Softmax(),
        )
        self.layers = []
        self.layers.append(self.layer0)
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)
        self.layers.append(self.layer3)
        self.layers.append(self.layer4)
        self.layers.append(self.layer5)
        self.layers = self.layers[num_step+1:6]

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

class View(torch.nn.Module):

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)