# sofmax Note
## 1、下载数据集
```python
# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../dataset',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../dataset',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)
```
#### 相关参数：
1. **root：** 数据集所在的地址，`../`表示上两级文件
2. **train:** 表示是否用于训练，有`True`和`False`俩个选项
3. **trainsform:** 个参数指定了对数据进行的预处理操作。`transforms.ToTensor()` 会将图像数据从 `PIL.Image` 或 `numpy.ndarray` 格式转换为 PyTorch 的 `Tensor` 格式，并且会将像素值从 0-255 归一化到 0-1 之间。这是为了使图像数据能够被神经网络模型处理。
4. **download:** 表示没有数据集需要下载
   
## 2、加载数据集
```python
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
```
#### 相关参数：
1. **dataset**: 这里指定了数据集来源，即 `train_dataset`。`train_dataset` 是一个数据集对象，通常是 `torch.utils.data.Dataset` 的子类实例，包含了训练数据和对应的标签。
2. **batch_size:** 小批次的大小，每次从数据集收取多少数据
3. **shuffle:** 是否随机打乱数据集

## 3、超参数
```python
input_size = 28 * 28  # 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
```
#### 相关参数
1. **input_size:** 输入的格式
2. **num_class:** 输出的类的种类个数
3. **num_epochs:** 训练的次数
4. **batch_size:** 每次抽取的个数
5. **learning_rate:** 学习率
## 4、回归模型
```python
model = nn.Linear(input_size, num_classes)
```
## 5、损失函数
```python
criterion = nn.CrossEntropyLoss()
```
`nn.CrossEntropyLoss()`是交叉熵损失函数
#### 实现原理：
```python
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, outputs, labels):
        loss = torch.mean(
            -torch.gather(outputs, 1, labels.reshape(labels.size()[0], 1)) + torch.log(outputs.exp().sum(1)))
        return loss
```

具体看`loss`函数的实现
这个代码片段是在计算某种损失函数，结合了交叉熵损失（Cross-Entropy Loss）和对数似然（Log-Likelihood）。让我们逐步分析这个代码的意思：

1. **`outputs`**：这是模型的输出，通常是经过 softmax 层后的概率分布，形状为 `(batch_size, num_classes)`，表示每个样本属于每个类别的概率。

2. **`labels`**：这是目标标签，通常是长度为 `batch_size` 的向量，其中每个元素是对应样本的正确类别的索引。

3. **`labels.reshape(labels.size()[0], 1)`**：将 `labels` 重新 reshape 成 `(batch_size, 1)` 形状，这样它就可以与 `outputs` 进行索引操作。

4. **`torch.gather(outputs, 1, labels.reshape(labels.size()[0], 1))`**：`torch.gather` 函数从 `outputs` 中选择与 `labels` 中索引对应的那些概率值。这一操作结果的形状为 `(batch_size, 1)`，表示每个样本在正确类别上的预测概率。

5. **`-torch.gather(outputs, 1, labels.reshape(labels.size()[0], 1))`**：计算每个样本的负对数概率，这相当于在计算对数似然损失的一部分。

6. **`outputs.exp().sum(1)`**：计算 `outputs` 在每个样本上的指数和，即每个样本在所有类别上的未归一化概率总和。这步是为了将 softmax 转化回未归一化的 logits。

7. **`torch.log(outputs.exp().sum(1))`**：对上述总和取对数，得到 `log-sum-exp`，这在数值上等价于稳定的 softmax 函数的一部分。

8. **`-torch.gather(...) + torch.log(outputs.exp().sum(1))`**：结合了负对数概率和 `log-sum-exp`，这部分实际上在计算一个类似于最大似然估计中的损失。

9. **`torch.mean(...)`**：对所有样本的损失求平均，得到最终的损失值。

#### 公式推导
![公式](https://latex.codecogs.com/svg.image?\(\text{Loss}_i=-z_i&plus;\log\left(\sum_{j}e^{z_j}\right)\)=\(\text{Loss}=-\log(p_i)\))
1. **Softmax 函数**

   Softmax 函数将 logits 转换为概率分布，其公式为：
   ![公式](https://latex.codecogs.com/svg.latex?p_i%20=%20\frac{e^{z_i}}{\sum_{j}%20e^{z_j}})

   这是将未归一化的 logits \(z\) 转换为概率 \(p\) 的过程，其中 \(p_i\) 是类别 \(i\) 的预测概率。

2. **交叉熵损失函数**

   交叉熵损失用于衡量预测分布和真实分布之间的差异。对于一个样本，其损失公式为：
   ![公式](https://latex.codecogs.com/svg.latex?\text{Loss}%20=%20-\log(p_i))

   对于真实类别 \(i\)，这相当于：
   ![公式](https://latex.codecogs.com/svg.latex?\text{Loss}%20=%20-\log\left(\frac{e^{z_i}}{\sum_{j}%20e^{z_j}}\right))

#### 化简过程

1. **直接带入公式：**

   ![公式](https://latex.codecogs.com/svg.latex?\text{Loss}%20=%20-\log\left(\frac{e^{z_i}}{\sum_{j}%20e^{z_j}}\right))

2. **使用对数的性质**

   ![公式](https://latex.codecogs.com/svg.latex?\text{Loss}%20=%20-\left(\log(e^{z_i})%20-%20\log\left(\sum_{j}%20e^{z_j}\right)\right))

3. **进一步化简：**

   ![公式](https://latex.codecogs.com/svg.latex?\text{Loss}%20=%20-z_i%20+%20\log\left(\sum_{j}%20e^{z_j}\right))

## 6、模型优化
```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```
## 7、训练模型
```python
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size)

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)
        # print(labels.size())
        # print(outputs.size())
        # print(loss.size())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            # print(model.weight)
            # print(model.bias)
```
这段代码是一个典型的用于训练神经网络模型的循环，使用的是PyTorch库。下面逐步解释这段代码：

1. **`total_step = len(train_loader)`**  
   这行代码计算训练数据集的总批次数（steps），即`train_loader`中有多少批数据。这是通过`len(train_loader)`来获取的，`train_loader`是一个数据加载器，用于迭代训练数据集。

2. **`for epoch in range(num_epochs):`**  
   外层循环遍历训练的轮数，`num_epochs`表示训练的总轮数。每次循环表示一个完整的训练周期。

3. **`for i, (images, labels) in enumerate(train_loader):`**  
   内层循环遍历每一个批次的训练数据。`train_loader`每次返回一批`images`和对应的`labels`。`i`是批次的索引，`images`是输入的图像数据，`labels`是对应的标签。

4. **`images = images.reshape(-1, input_size)`**  
   这行代码将图像数据`images`重新调整形状。`-1`表示自动计算这个维度的大小，而`input_size`表示输入层的大小。假设图像是二维的，那么这一步将图像展平为一维的输入向量。

5. **`outputs = model(images)`**  
   这行代码进行前向传播（Forward pass），即将输入`images`传递给模型`model`，得到模型的输出`outputs`。

6. **`loss = criterion(outputs, labels)`**  
   这行代码计算损失（loss）。`criterion`是损失函数，用于计算模型预测`outputs`与实际标签`labels`之间的差距。

7. **`optimizer.zero_grad()`**  
   在进行反向传播之前，这行代码将梯度缓存清零。因为PyTorch默认会累积梯度，所以在每次更新参数之前都要将之前的梯度清零。

8. **`loss.backward()`**  
   这行代码执行反向传播（Backward pass），计算模型参数的梯度。

9. **`optimizer.step()`**  
   这行代码执行优化步骤，利用计算得到的梯度来更新模型的参数，以减少损失。

10. **`if (i + 1) % 100 == 0:`**  
    这行代码用于在每经过100个批次后打印一次当前的训练状态，包括当前的`epoch`，`step`（批次），以及当前的`loss`（损失值）。

11. **`print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))`**  
    这一部分输出训练过程中的信息，帮助监控训练过程。

12. **注释的部分代码**（`print(labels.size())`、`print(outputs.size())`、`print(loss.size())`、`print(model.weight)`、`print(model.bias)`）：
    这些被注释掉的代码是用于调试的，帮助查看张量的维度和模型参数。这些信息在调试时可以帮助确认数据在训练过程中的形状是否正确，模型的权重和偏差是否在预期范围内变化。

总结来说，这段代码通过多轮次、多批次地将训练数据输入模型，逐步调整模型参数以最小化损失函数，从而优化模型的性能。
## 8、测试模型
```python
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * torch.true_divide(correct, total)))

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
```
这段代码主要用于在测试数据集上评估模型的性能，并计算模型的准确率。以下是对代码的逐步解释：

1. **`with torch.no_grad():`**  
   这行代码开始了一个上下文管理器，其中所有计算都不会记录计算图（不会计算梯度）。在评估模型时，我们不需要进行梯度计算，因此使用`torch.no_grad()`可以节省内存和加快计算速度。

2. **`correct = 0` 和 `total = 0`**  
   这两行代码初始化了两个变量：`correct`用于记录模型预测正确的样本数量，`total`用于记录总的测试样本数量。

3. **`for images, labels in test_loader:`**  
   这行代码开始循环遍历测试数据集的每个批次。`test_loader`是测试数据集的加载器，每次迭代返回一批图像`images`和它们的真实标签`labels`。

4. **`images = images.reshape(-1, input_size)`**  
   这行代码将图像数据重新调整形状，类似于训练时的操作。`-1`自动计算批次大小，`input_size`表示输入层的大小。

5. **`outputs = model(images)`**  
   这行代码将图像输入模型，进行前向传播，得到模型的预测输出`outputs`。

6. **`_, predicted = torch.max(outputs.data, 1)`**  
   这行代码从模型的输出中获取预测的类别。`torch.max(outputs.data, 1)`返回每一行（即每个样本）的最大值及其索引，`predicted`保存了预测类别的索引（即预测的标签）。

7. **`total += labels.size(0)`**  
   这行代码累加当前批次中的样本数量，即更新总样本数`total`。

8. **`correct += (predicted == labels).sum()`**  
   这行代码计算当前批次中预测正确的样本数，并将其累加到`correct`中。`predicted == labels`返回一个布尔张量，表示哪些预测是正确的，而`.sum()`计算其中`True`的数量。

9. **`print('Accuracy of the model on the 10000 test images: {} %'.format(100 * torch.true_divide(correct, total)))`**  
   这行代码计算并打印模型在测试数据集上的准确率。`torch.true_divide(correct, total)`计算正确预测数与总样本数之比，乘以100得到百分比形式的准确率。

10. **`torch.save(model.state_dict(), 'model.ckpt')`**  
    这行代码被注释掉了，通常用于保存训练好的模型的状态字典（包含模型的参数）。文件名为`model.ckpt`。这样以后可以加载这个模型进行推理或继续训练。

总结来说，这段代码评估了模型在测试数据集上的准确率，并可以选择性地保存模型的参数。使用`torch.no_grad()`是为了在推理阶段提高效率并减少内存使用。