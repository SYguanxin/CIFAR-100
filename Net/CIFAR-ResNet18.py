import torch
from torch import nn
from torch.nn import functional as F
from dataFunc import load_data_CIFAR
from d2l import torch as d2l
cuda = torch.device('cuda')


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


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device, params=None, lock=False):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_utils`"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    if not params:
        net.apply(init_weights)
    else:
        net.load_state_dict(params)
    def lock_weights(m):
        if type(m) != nn.LazyLinear:
            m.weight.require_grad_ = False
    if lock:
        net.apply(lock_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
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
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
        animator.add(epoch + 1,
                    (train_l, None, None), loss=True)
        print(f"epoch: [{epoch + 1}/{num_epochs}], "
              f"train_l: {train_l: .3f}, train_acc: {train_acc: .3f}")
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, train_acc, test_acc))
        print(f"test_acc: {test_acc: .3f}")
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


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
train_iter, test_iter = load_data_CIFAR(batch_size, resize=size, download=True)
num_epochs = 1
lr = 0.001
params = torch.load('../ResNet-18.params')
# d2l.plt.ion()
train_ch6(net, train_iter, test_iter, num_epochs, lr=lr, device=cuda,
          params=params, lock=False
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
    torch.save(net.state_dict(), '../ResNet-18.params', )