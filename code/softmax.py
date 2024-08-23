import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
from d2l import torch as d2l
import os


d2l.use_svg_display()

trans = transforms.ToTensor()
batch_size = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# PyTorch不会隐式地调整输⼊的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整⽹络输⼊的形状
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 10)
                    )


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)
# 这个方法会对网络 net 中的每一层调用 init_weights 函数。
# 如果某一层是 nn.Linear，就会用前面定义的正态分布来初始化其权重。
loss = nn.CrossEntropyLoss(reduction='none')
# none 返回所有损失值
# mean 返回损失的平均值
# std 返回方差

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
