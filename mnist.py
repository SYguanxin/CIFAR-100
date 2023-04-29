import torch
from torch import nn
from dataFunc import load_data_fashion_mnist
from d2l import torch as d2l
import matplotlib.pyplot as plt
cuda = torch.device('cuda')

batch_size = 64
train_iter,test_iter = load_data_fashion_mnist(batch_size)

net = nn.Sequential(
    nn.Flatten(),
    # nn.LazyLinear(256),
    nn.Linear(784,256),
    nn.Dropout(0.2),
    nn.LeakyReLU(),
    nn.Linear(256,256),
    nn.Dropout(0.2),
    nn.LeakyReLU(),
    nn.Linear(256,256),
    nn.Dropout(),
    nn.LeakyReLU(),
    nn.Linear(256,256),
    nn.Dropout(),
    nn.LeakyReLU(),
    nn.Linear(256,10),
)
net = net.to(device=cuda)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('leaky_relu', 1e-2))
net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none').cuda()
trainer = torch.optim.SGD(net.parameters(),lr=0.1)

num_epochs = 20
from softmax0 import train_ch3
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer, GPU=True)
plt.show()

# for epoch in range(num_epochs):
#     for X, y in train_iter:
#         l = loss(net(X.cuda()),y.cuda())
#         trainer.zero_grad()
#         l.mean().backward()
#         trainer.step()
#     l = loss(net(feature.float().cuda()),label.cuda())
#     print(f"epochs: {epoch+1}, loss: {l.mean():f}")
#     ans = 0
#     cnt = 0
#     for X, y in test_iter:
#         cnt += len(y)
#         ans += (net(X.reshape(-1, pixels).cuda()).argmax(axis=1) == y.cuda()).sum()
#     print((ans / cnt).item())