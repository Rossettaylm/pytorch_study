# PyTorch with fasionMNIST

```python
import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
```

## Working with data

two primitives to work with data:  
    1. torch.utils.data.DataLoader  
    2. torch.utils.data.Dataset


```python
# Download training data from open datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data\FashionMNIST\raw\train-images-idx3-ubyte.gz


    26422272it [00:04, 6230197.58it/s]                               


    Extracting data\FashionMNIST\raw\train-images-idx3-ubyte.gz to data\FashionMNIST\raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data\FashionMNIST\raw\train-labels-idx1-ubyte.gz


    29696it [00:00, 103505.31it/s]                          


    Extracting data\FashionMNIST\raw\train-labels-idx1-ubyte.gz to data\FashionMNIST\raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz


    4422656it [00:03, 1339112.90it/s]                             


    Extracting data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz to data\FashionMNIST\raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz


    6144it [00:00, 757423.03it/s]           
    C:\Users\DELL\anaconda3\lib\site-packages\torchvision\datasets\mnist.py:502: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ..\torch\csrc\utils\tensor_numpy.cpp:143.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


    Extracting data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz to data\FashionMNIST\raw
    
    Processing...
    Done!



```python
# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```


```python
batch_size = 64

# Create data loaders
# DataLoader will return a iterator
training_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
```

    Shape of X [N, C, H, W]:  torch.Size([64, 1, 28, 28])
    Shape of y:  torch.Size([64]) torch.int64


## Creating Models


```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device.".format(device))
```

    Using cpu device.



```python
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
```

    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
      )
    )


## Optimizing the Model Parameters
To train a model, we need a loss function and an optimizer.


```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            
     
            
```


```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------")
    train(training_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

    Epoch 1
    ------------------------
    loss: 2.296762  [    0/60000]
    loss: 2.285266  [ 6400/60000]
    loss: 2.272632  [12800/60000]
    loss: 2.270942  [19200/60000]
    loss: 2.245825  [25600/60000]
    loss: 2.229274  [32000/60000]
    loss: 2.230557  [38400/60000]
    loss: 2.204137  [44800/60000]
    loss: 2.198165  [51200/60000]
    loss: 2.174469  [57600/60000]
    Test Error: 
     Accuracy: 46.8%, Avg loss: 2.160602 
    
    Epoch 2
    ------------------------
    loss: 2.164593  [    0/60000]
    loss: 2.151814  [ 6400/60000]
    loss: 2.109528  [12800/60000]
    loss: 2.128762  [19200/60000]
    loss: 2.066061  [25600/60000]
    loss: 2.020851  [32000/60000]
    loss: 2.042474  [38400/60000]
    loss: 1.975194  [44800/60000]
    loss: 1.973679  [51200/60000]
    loss: 1.914203  [57600/60000]
    Test Error: 
     Accuracy: 58.7%, Avg loss: 1.902848 
    
    Epoch 3
    ------------------------
    loss: 1.929178  [    0/60000]
    loss: 1.895707  [ 6400/60000]
    loss: 1.798237  [12800/60000]
    loss: 1.839237  [19200/60000]
    loss: 1.722130  [25600/60000]
    loss: 1.678195  [32000/60000]
    loss: 1.694907  [38400/60000]
    loss: 1.606613  [44800/60000]
    loss: 1.614090  [51200/60000]
    loss: 1.523664  [57600/60000]
    Test Error: 
     Accuracy: 62.7%, Avg loss: 1.536097 
    
    Epoch 4
    ------------------------
    loss: 1.597228  [    0/60000]
    loss: 1.558755  [ 6400/60000]
    loss: 1.425287  [12800/60000]
    loss: 1.492921  [19200/60000]
    loss: 1.371590  [25600/60000]
    loss: 1.364128  [32000/60000]
    loss: 1.376942  [38400/60000]
    loss: 1.311277  [44800/60000]
    loss: 1.328170  [51200/60000]
    loss: 1.246766  [57600/60000]
    Test Error: 
     Accuracy: 64.4%, Avg loss: 1.267000 
    
    Epoch 5
    ------------------------
    loss: 1.341949  [    0/60000]
    loss: 1.321484  [ 6400/60000]
    loss: 1.167393  [12800/60000]
    loss: 1.266958  [19200/60000]
    loss: 1.140879  [25600/60000]
    loss: 1.160158  [32000/60000]
    loss: 1.184244  [38400/60000]
    loss: 1.127096  [44800/60000]
    loss: 1.150794  [51200/60000]
    loss: 1.088549  [57600/60000]
    Test Error: 
     Accuracy: 65.2%, Avg loss: 1.100314 
    
    Done!


## Saving Models


```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth!")
```

    Saved PyTorch Model State to model.pth!


## Loading Models


```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
```




    <All keys matched successfully>




```python
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
    
```

    Prediced: 'Ankle boot', Actual: 'Ankle boot'

