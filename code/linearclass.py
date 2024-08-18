import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
# 一次取64张图片
dataloader = DataLoader(dataset, batch_size=64)


class Mydata(nn.Module):
    def __init__(self):
        super(Mydata, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x


# [4, 3, 32, 32]->[1, 1, 1, 12288]通过线性层将12288->10
mydata = Mydata()
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)  # [4, 3, 32, 32]
    output = torch.reshape(imgs, (1, 1, 1, -1))
    # output = torch.flatten(imgs)
    print(output.shape)  # [1, 1, 1, 12288]
    output = mydata(output)
    print(output.shape)  # [1, 1, 1, 10]
