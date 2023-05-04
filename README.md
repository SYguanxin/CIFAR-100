<!-- TOC -->

* [1. ~~本篇用到的绝大部分代码是李沐大佬的代码的改用~~](#1-)
* [2. ~~代码里的plt.ion()和plt.ioff()我也不知道怎么回事~~](#2-pltionpltioff)
* [一些写代码时学到的方法](#)
* [尝试的模型](#)
  * [0. 用到的一些基础类](#0-)
    * [Animator](#animator)
    * [Accumulator](#accumulator)
    * [train_ch6](#train_ch6)
    * [load_data_CIFAR](#load_data_cifar)
  * [1. 多层感知机(MLP)](#1---mlp-)
    * [代码](#)
    * [迭代20次结果](#20)
    * [使用dropout](#dropout)
  * [2. LeNet](#2-lenet)
    * [代码](#)
    * [迭代50次结果](#50)
  * [3.AlexNet](#3alexnet)
    * [3.1. 仅在Linear层加入Dropout的结果](#31-lineardropout)
      * [模型](#)
    * [3.2. 加入更多Dropout层](#32-dropout)
      * [模型](#)
    * [3.3. 最后附上完整代码](#33-)
  * [4. VGG-11](#4-vgg-11)
    * [4.1. 原版模型代码](#41-)
    * [原版迭代10次结果图](#10)
    * [4.2. 加入批量规范化](#42-)
      * [模型](#)
      * [结果图](#)
  * [5. NiN](#5-nin)
    * [模型](#)
    * [迭代20次结果](#20)
  * [6. GoogLeNet](#6-googlenet)
    * [~~跑不动~~](#)
  * [7. ResNet](#7-resnet)
    * [7.1. ResNet-18](#71-resnet-18)
      * [模型代码](#)
      * [未使用数据增强](#)
      * [使用数据增强](#)
      * [迭代50次结果](#50)
      * [保存模型，修改学习率](#)
      * [完整代码](#)
    * [7.2. ResNet-50](#72-resnet-50)
      * [模型](#)

<!-- TOC -->

### 神经网络都在Net里，github上没有结果图，CSDN上有

### 1. ~~本篇用到的绝大部分代码是李沐大佬的代码的改用~~

### 2. ~~代码里的plt.ion()和plt.ioff()我也不知道怎么回事~~

`不注释掉会报错`
`不写没效果`
`报错信息:NotImplementedError: Implement enable_gui in a subclass`

# 一些写代码时学到的方法

* 用plt显示3通道图片data读取后的shape为(3,32,32)
  -- 对tensor用permute()更改维度顺序

# 尝试的模型

### 0. 用到的一些基础类

##### Animator

```python
class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_utils`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda axid: d2l.set_axes(
            self.axes[axid], xlabel, ylabel, xlim, ylim[axid], xscale, yscale, legend[1-axid:])
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y, loss=False):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        if not loss:
            axid = 0
            self.axes[axid].cla()
            for x, y, fmt in zip(self.X[1:], self.Y[1:], self.fmts[1:]):
                self.axes[axid].plot(x, y, fmt)
        else:
            axid = 1
            self.axes[axid].cla()
            self.axes[axid].plot(self.X[0], self.Y[0], self.fmts[0])
        self.config_axes(axid)
        d2l.plt.pause(0.1)
        # display.display(self.fig)
        # display.clear_output(wait=True)
```

##### Accumulator

```python
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

##### train_ch6

```python
def train_ch6(net, train_iter, test_iter, num_epochs, lr, wd, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_utils`"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[[0, 0.9], [0, 5]],
                        legend=['train loss', 'train acc', 'test acc'],
                        ncols=2, figsize=(7, 2.5))
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (None, train_acc, None))
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, None, None), loss=True)
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

##### load_data_CIFAR

```python
def load_data_CIFAR(batch_size, resize=None, strong_data=False, download=False):
    trans = [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans_h = transforms.Compose([
        transforms.ColorJitter(brightness=(0.5, 1.5)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        *trans,
    ])
    trans_orin = transforms.Compose(trans)
    train_h = torchvision.datasets.CIFAR100(
        root='../data', train=True, transform=trans_h, download=download)
    test = torchvision.datasets.CIFAR100(
        root='../data', train=False, transform=trans_orin, download=download)
    return (data.DataLoader(train_h, batch_size, shuffle=True, pin_memory=True),
            data.DataLoader(test, batch_size, shuffle=False, pin_memory=True))
```

### 1. 多层感知机(MLP)

##### 代码

```python
import torch
from torch import nn
from dataFunc import load_data_CIFAR
cuda = torch.device('cuda')

batch_size = 64
train_iter,test_iter = load_data_CIFAR(batch_size)

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3072,1024),
    nn.LeakyReLU(1e-2),
    nn.Linear(1024,256),
    nn.LeakyReLU(1e-2),
    nn.Linear(256,100),
)
net = net.to(device=cuda)
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none').cuda()
trainer = torch.optim.SGD(net.parameters(),lr=0.1)

num_epochs = 20
from softmax0 import train_ch3
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

##### 迭代20次结果

```python
epoch: 1 loss: 4.345764 train acc: 3.83% test acc: 3.63%
epoch: 2 loss: 3.919320 train acc: 9.28% test acc: 6.63%
epoch: 3 loss: 3.710387 train acc: 12.92% test acc: 13.98%
epoch: 4 loss: 3.569426 train acc: 15.45% test acc: 13.77%
epoch: 5 loss: 3.460782 train acc: 17.61% test acc: 16.15%
epoch: 6 loss: 3.379023 train acc: 19.07% test acc: 16.83%
epoch: 7 loss: 3.303872 train acc: 20.46% test acc: 15.60%
epoch: 8 loss: 3.233504 train acc: 21.46% test acc: 16.67%
epoch: 9 loss: 3.179428 train acc: 22.53% test acc: 17.65%
epoch: 10 loss: 3.120779 train acc: 23.56% test acc: 13.51%
epoch: 11 loss: 3.064707 train acc: 24.87% test acc: 18.30%
epoch: 12 loss: 3.016871 train acc: 25.76% test acc: 21.28%
epoch: 13 loss: 2.964137 train acc: 26.63% test acc: 20.65%
epoch: 14 loss: 2.907337 train acc: 27.65% test acc: 18.72%
epoch: 15 loss: 2.867398 train acc: 28.43% test acc: 18.52%
epoch: 16 loss: 2.819782 train acc: 29.29% test acc: 18.14%
epoch: 17 loss: 2.771111 train acc: 30.26% test acc: 15.98%
epoch: 18 loss: 2.728714 train acc: 31.10% test acc: 18.78%
epoch: 19 loss: 2.678194 train acc: 32.09% test acc: 19.96%
epoch: 20 loss: 2.636123 train acc: 32.91% test acc: 19.80%
```

##### 使用dropout

```python
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3072,1024),
    nn.LeakyReLU(1e-2),
    nn.Dropout(),
    nn.Linear(1024,256),
    nn.LeakyReLU(1e-2),
    nn.Dropout(),
    nn.Linear(256,100),
)
```

* 发现梯度下降速率明显变慢
* 未出现过拟合

### 2. LeNet

##### 代码

```python
import torch
from torch import nn
from dataFunc import load_data_CIFAR
from d2l import torch as d2l
cuda = torch.device('cuda')

batch_size = 64
train_iter,test_iter = load_data_CIFAR(batch_size)

net = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5), nn.ReLU(), nn.Dropout(0.2),
    nn.AvgPool2d(2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5, padding=2), nn.ReLU(), nn.Dropout(),
    nn.AvgPool2d(2, stride=2), nn.Flatten(),
    nn.LazyLinear(256), nn.ReLU(), nn.Dropout(),
    nn.Linear(256, 100)
)
trainer = torch.optim.SGD(net.parameters(),lr=0.1)

num_epochs = 50
# d2l.plt.ion()
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr=0.1, device=cuda)
# d2l.plt.ioff()
d2l.plt.show()
```

##### 迭代50次结果

```python
loss 2.769, train acc 0.302, test acc 0.302
```

### 3.AlexNet

#### 3.1. 仅在Linear层加入Dropout的结果

##### 模型

```python
net = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 100))
```

* $loss 0.152 train_acc 0.946 test_acc 0.492$ 过拟合严重
* 结果图

#### 3.2. 加入更多Dropout层

##### 模型

```python
net = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(p=0.2),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(p=0.2),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(p=0.3),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 100))
```

* loss 0.496 train_acc 0.896 test_acc 0.549
* 结果图

#### 3.3. 最后附上完整代码

```python
import torch
from torch import nn
from torch.nn import functional as F
from dataFunc import load_data_CIFAR
from d2l import torch as d2l
cuda = torch.device('cuda')

batch_size = 64
train_iter,test_iter = load_data_CIFAR(batch_size, resize=224)

net = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(p=0.2),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(p=0.2),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(p=0.3),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 100))

num_epochs = 50
lr = 0.01
# d2l.plt.ion()
train_ch6(net, train_iter, test_iter, num_epochs, lr=lr, device=cuda)
# d2l.plt.ioff()
d2l.plt.show()
```

### 4. VGG-11

##### 4.1. 原版模型代码

```python
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2,512), (2,512))
def vgg(conv_arch):
    conv_blks = []
    in_channels = 3
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(
            num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.LazyLinear(4096), nn.Dropout(0.5), nn.ReLU(),
        nn.Linear(4096, 4096), nn.Dropout(0.5), nn.ReLU(),
        nn.Linear(4096, 100)
    )

net = vgg(conv_arch)
```

##### 原版迭代10次结果图

#### 4.2. 加入批量规范化

##### 模型

```python
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2,512), (2,512))
def vgg(conv_arch):
    conv_blks = []
    in_channels = 3
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(
            num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.LazyLinear(4096), nn.BatchNorm1d(4096), nn.ReLU(),
        nn.Linear(4096, 4096), nn.BatchNorm1d(4096), nn.ReLU(),
        nn.Linear(4096, 100)
    )

net = vgg(conv_arch)
```

##### 结果图

### 5. NiN

##### 模型

```python
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

net = nn.Sequential(
    nin_block(3, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(384, 100, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)
```

##### 迭代20次结果

* loss 0.136, train acc 0.968, test acc 0.557
* 结果图

### 6. GoogLeNet

##### ~~跑不动~~

### 7. ResNet

#### 7.1. ResNet-18

##### 模型代码

```python
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels, use_1x1conv=True))
    return blk

b1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(), nn.Dropout(0.5), nn.LazyLinear(100)
)
```

##### 未使用数据增强

* 测试集准确率: 54.6%

##### 使用数据增强

```python
def load_data_CIFAR(batch_size, resize=None, strong_data=False, download=False):
    trans = [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans_orin = transforms.Compose(trans)
    trans_h = transforms.Compose([
        transforms.ColorJitter(brightness=(0.5, 1.5)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        *trans,
    ])
    train_orin = torchvision.datasets.CIFAR100(
        root='./data', train=True, transform=trans_h, download=download)
    test = torchvision.datasets.CIFAR100(
        root='./data', train=False, transform=trans_orin, download=download)
    return (data.DataLoader(train_orin, batch_size, shuffle=True, num_workers=get_dataloader_workers(),
                            prefetch_factor=2*get_dataloader_workers(), pin_memory=True),
            data.DataLoader(test, batch_size, shuffle=False, num_workers=get_dataloader_workers(),
                            prefetch_factor=2*get_dataloader_workers(), pin_memory=True))
```

##### 迭代50次结果

* 测试集准确度64.1%
* 结果图

##### 保存模型，修改学习率

* 测试集准确率68.8%

##### 完整代码

```python
import torch
from torch import nn
from torch.nn import functional as F
from dataFunc import load_data_CIFAR
from d2l import torch as d2l
cuda = torch.device('cuda')

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels, use_1x1conv=True))
    return blk

b1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(), nn.Dropout(0.5), nn.LazyLinear(100)
)

batch_size, size = 64, 96
train_iter, test_iter = load_data_CIFAR(batch_size, resize=size)
num_epochs = 50
lr = 0.001
params = torch.load('ResNet-18.params')
# d2l.plt.ion()
train_ch6(net, train_iter, test_iter, num_epochs, lr=lr, device=cuda,
          # params=params, lock=False
          )
# d2l.plt.ioff()
d2l.plt.show()

n = 8
from dataFunc import predict, get_CIFAR_label
predict(net, size, test_iter, n, get_CIFAR_label, device=cuda)
d2l.plt.show()

save = input()
if save == 'save':
    print('saving')
    torch.save(net.state_dict(), 'ResNet-18.params', )
```

#### 7.2. ResNet-50

~~尝试过准确率没有ResNet-18高~~

##### 模型

```python
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=1, padding=0, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, 4 * num_channels,
                               kernel_size=1, padding=0)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, 4 * num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(4 * num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.conv4(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0:
            if not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels, use_1x1conv=True))
        else:
            blk.append(Residual(4 * num_channels, num_channels, use_1x1conv=True))
    return blk

b1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
b2 = nn.Sequential(*resnet_block(64, 64, 3, first_block=True))
b3 = nn.Sequential(*resnet_block(256, 128, 4))
b4 = nn.Sequential(*resnet_block(512, 256, 6))
b5 = nn.Sequential(*resnet_block(1024, 512, 3))

net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(), nn.Dropout(0.5), nn.Linear(2048, 100)
)
```
