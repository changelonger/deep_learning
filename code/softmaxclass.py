# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Hyper-parameters
input_size = 28 * 28  # 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../dataset',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../dataset',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Logistic regression model
model = nn.Linear(input_size, num_classes)
# print(model.weight)
# print(model.bias)
# print(model.in_features)


# customize loss
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, outputs, labels):
        loss = torch.mean(
            -torch.gather(outputs, 1, labels.reshape(labels.size()[0], 1))
            + torch.log(outputs.exp().sum(1)))
        return loss


# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
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

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
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