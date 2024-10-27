# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

# 创建一个卷积层
conv_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)

# 假设输入的形状为 (batch_size, channels, height, width)
# 这里我们创建一个批量大小为1、1个通道、8x8的输入特征图
input_tensor = torch.tensor([[[[1, 2, 3, 4, 5, 6, 7, 8],
                                [1, 2, 3, 4, 5, 6, 7, 8],
                                [1, 2, 3, 4, 5, 6, 7, 8],
                                [1, 2, 3, 4, 5, 6, 7, 8],
                                [1, 2, 3, 4, 5, 6, 7, 8],
                                [1, 2, 3, 4, 5, 6, 7, 8],
                                [1, 2, 3, 4, 5, 6, 7, 8],
                                [1, 2, 3, 4, 5, 6, 7, 8]]]]).float()

# 应用卷积层
output_tensor = conv_layer(input_tensor)

# 打印输出形状和内容
print("Output shape:", output_tensor.shape)
print("Output tensor:", output_tensor)
