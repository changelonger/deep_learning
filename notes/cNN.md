这段代码实现了一个使用卷积神经网络（CNN）进行MNIST手写数字分类的完整流程。以下是对每个部分的详细解释：

### 1. 文件头部

```python
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 21:14:48 2020

@author: Administrator
"""
```
- 这部分是文件的元数据，指定文件的编码和创建信息。

### 2. 导入库

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
```
- `torch`：PyTorch的核心库。
- `torch.nn`：包含构建神经网络的模块。
- `torchvision`：提供常用数据集和图像处理工具。
- `torchvision.transforms`：用于数据预处理和增强的工具。

### 3. 设备配置

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
```
- 检查是否有可用的GPU，如果有，使用GPU，否则使用CPU。`device`用于后续的计算。

### 4. 超参数定义

```python
num_epochs = 5
num_classes = 10
batch_size = 1
learning_rate = 0.001
```
- **`num_epochs`**：训练的轮数。
- **`num_classes`**：分类的数量（MNIST有10个数字）。
- **`batch_size`**：每个批次的数据量（这里设置为1）。
- **`learning_rate`**：学习率，用于优化器。

### 5. MNIST数据集加载

```python
train_dataset = torchvision.datasets.MNIST(root='../data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../data/',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)
```
- 使用`torchvision.datasets.MNIST`加载MNIST数据集，训练集和测试集。
- `transform=transforms.ToTensor()`将图像转换为PyTorch的Tensor格式。

### 6. 数据加载器

```python
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
```
- `DataLoader`用于批量加载数据，`shuffle=True`表示在训练时打乱数据顺序。

### 7. 卷积神经网络定义

```python
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
```
- 定义一个名为`ConvNet`的卷积神经网络类，继承自`nn.Module`。

#### 第一层（layer1）

```python
self.layer1 = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2))
```
- **`nn.Conv2d`**：输入通道1（灰度图），输出通道16，卷积核大小5x5，步幅1，填充2。
- **`nn.BatchNorm2d(16)`**：对16个通道进行批量归一化。
- **`nn.ReLU()`**：激活函数，增加非线性。
- **`nn.MaxPool2d`**：最大池化层，池化大小2x2，步幅2，减小特征图尺寸。
  
![公式](https://latex.codecogs.com/svg.image?\[\text{Output&space;Size}=\left\lfloor\frac{\text{Input&space;Size}&plus;2\times\text{Padding}-\text{Kernel&space;Size}}{\text{Stride}}\right\rfloor&plus;1\])

#### 第二层（layer2）

```python
self.layer2 = nn.Sequential(
    nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2))
```
- 类似于第一层，但输入通道为16，输出通道为32。

#### 全连接层

```python
self.fc = nn.Linear(7 * 7 * 32, num_classes)
```
- 将卷积层输出展平后，连接到全连接层，输入大小为`7 * 7 * 32`，输出为`num_classes`（10）。

#### 前向传播

```python
def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)  # 展开
    out = self.fc(out)
    return out
```
- 定义前向传播过程，依次通过卷积层和全连接层，并将特征图展平。

### 8. 模型初始化

```python
model = ConvNet(num_classes).to(device)
```
- 创建`ConvNet`模型并移动到指定设备（GPU或CPU）。

### 9. 损失函数和优化器

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
- **`nn.CrossEntropyLoss()`**：用于多分类的交叉熵损失函数。
- **`torch.optim.Adam`**：使用Adam优化器更新模型参数。

### 10. 训练模型

```python
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
```
- 遍历每个epoch和批次，加载数据并移动到设备。

#### 前向传播、损失计算和反向传播

```python
outputs = model(images)
loss = criterion(outputs, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```
- 计算输出，损失，执行反向传播和优化。

#### 打印损失信息

```python
if (i + 1) % 100 == 0:
    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
```
- 每100个批次打印一次当前epoch、步骤和损失。

### 11. 测试模型

```python
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
```
- 将模型设置为评估模式，BatchNorm层将使用移动平均。

#### 评估准确率

```python
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
```

- 在测试集上评估模型准确率，`torch.no_grad()`用于关闭梯度计算以节省内存。

### 12. 保存模型

```python
torch.save(model.state_dict(), 'model.ckpt')
```
- 将训练好的模型参数保存到`model.ckpt`文件中，以便后续加载和使用。

### 总结

这段代码实现了一个完整的MNIST手写数字分类的训练和测试流程，涵盖了数据预处理、模型定义、训练过程、评估和模型保存的各个步骤，非常适合用作深度学习入门的示例。