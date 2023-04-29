import torch
from torch import nn
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


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_utils`"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[0, num_epochs], ylim=[[0, 0.9], [0, 5]],
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
                print(f"epoch: {epoch + (i + 1) / num_batches: .1f}, "
                      f"train_l: {train_l: .3f}, train_acc: {train_acc: .3f}")

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f"epoch: {epoch + 1}, test_acc: {test_acc: .3f}")
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


batch_size = 128
train_iter, test_iter = load_data_CIFAR(batch_size, resize=96)

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

num_epochs = 20
lr = 0.05
# d2l.plt.ion()
train_ch6(net, train_iter, test_iter, num_epochs, lr=lr, device=cuda)
# d2l.plt.ioff()
d2l.plt.show()


# from dataFunc import predict, get_CIFAR_label
# test_times = 2
# predict(net, batch_size, test_times, test_iter, get_CIFAR_label, device=cuda)
# d2l.plt.show()