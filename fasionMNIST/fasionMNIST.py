#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, ToTensor

# ## Working with data
# two primitives to work with data:
#     1. torch.utils.data.DataLoader
#     2. torch.utils.data.Dataset

# In[ ]:


# Download training data from open datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)


# In[ ]:


# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# In[ ]:


batch_size = 64

# Create data loaders
# DataLoader will return a iterator
training_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [Number, Color, Height, Width]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


# ## Creating Models

# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device.".format(device))


# In[ ]:


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)


# ## Optimizing the Model Parameters
# To train a model, we need a loss function and an optimizer.

# In[ ]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):  # 60000 // 64 = 938，共进行938个batch迭代
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Back propagation
        optimizer.zero_grad()  # 因为梯度是累加的，将梯度清零
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:  # 每训练100个batch的数据，显示一次损失值
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)  # 10000 / 64  迭代次数
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches  # 求出平均损失值
    correct /= size  # 求出平均准确率
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# In[ ]:
epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------")
    train(training_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# ## Saving Models

# In[ ]:


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth!")


# ## Loading Models

# In[ ]:


model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))


# In[ ]:


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: '{predicted}', Actual: '{actual}'")
