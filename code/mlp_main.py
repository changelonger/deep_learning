import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters    这些代码定义了超参数，包括输入大小、隐藏层大小、类别数量、训练周期数、批次大小和学习率
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset  这行代码加载MNIST训练集数据，并进行相应的数据处理，包括转换为张量（transforms.ToTensor()）和下载数据。
train_dataset = torchvision.datasets.MNIST(root='../dataset',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
# 这行代码加载MNIST测试集数据，并进行相应的数据处理。
test_dataset = torchvision.datasets.MNIST(root='../dataset',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader   这行代码创建一个训练集数据加载器，用于批量加载训练集数据，并进行随机打乱（shuffle=True）
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
# 这行代码创建一个测试集数据加载器，用于批量加载测试集数据
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Fully connected neural network with one hidden layer  这段代码定义了一个名为NeuralNet的类，该类继承自nn.Module。类中的__init__方法定义了模型的结构，包括两个线性层和一个ReLU激活函数。forward方法定义了模型的前向传播过程。
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)    #这行代码创建了一个NeuralNet的实例，并将其移动到指定的设备上

# Loss and optimizer  这些代码定义了损失函数（交叉熵损失）和优化器（Adam优化器）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model  这些代码开始训练模型。使用嵌套的循环遍历训练数据加载器中的每个批次，并进行前向传播、计算损失、反向传播和优化模型的参数。周期性地打印出当前训练步骤的信息
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model  这段代码在测试阶段，关闭梯度计算（使用torch.no_grad()），遍历测试数据加载器中的每个批次，并计算模型在测试集上的准确率。
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')