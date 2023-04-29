from torchvision import transforms
import torch
from torch.utils import data
import torchvision
from d2l import torch as d2l


def get_dataloader_workers():
    return 4


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

def load_data_mnist(batch_size, resize=None, download=False):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    train = torchvision.datasets.MNIST(
        root='./data', train=True, transform=trans, download=download)
    test = torchvision.datasets.MNIST(
        root='./data', train=False, transform=trans, download=download)
    return (data.DataLoader(train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))


def load_data_fashion_mnist(batch_size, resize=None, download=False):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    train = torchvision.datasets.FashionMNIST(
        root='./data', train=True, transform=trans, download=download)
    test = torchvision.datasets.FashionMNIST(
        root='./data', train=False, transform=trans, download=download)
    return (data.DataLoader(train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

def show_images(imgs, num_rows, num_cols, titles=None, scale=2.0):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            if img.device != 'cpu':
                img = img.to('cpu')
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_CIFAR_label(Y):
    labels = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone', 90: 'train', 28: 'cup',
     23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle', 71: 'sea',
     8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine_tree', 70: 'rose', 87: 'television', 84: 'table',
     64: 'possum', 52: 'oak_tree', 42: 'leopard', 47: 'maple_tree', 65: 'rabbit', 21: 'chimpanzee', 22: 'clock',
     81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster', 49: 'mountain', 56: 'palm_tree', 76: 'skyscraper',
     89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster',
     55: 'otter', 72: 'seal', 43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile',
     53: 'orange', 92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo',
     66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver', 61: 'plate',
     94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray', 30: 'dolphin', 62: 'poppy',
     63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard',
     7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn_mower', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl',
     57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}
    return [labels[int(i)] for i in Y]

def predict(net, size, test_iter, n, get_labels, device):
    for X, y in test_iter:
        break
    X = X.to(device)
    trues = get_labels(y)
    preds = get_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(
        X[0:n].reshape((n, 3, size, size)).permute(0,2,3,1), 1, n, titles=titles[0:n])


if __name__ == '__main__':
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.CIFAR100(
        root="./data", train=True, transform=trans)
    mnist_test = torchvision.datasets.CIFAR100(
        root="./data", train=False, transform=trans)

    d2l.use_svg_display()
    batch_size = 4
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                 num_workers=1)

    X, y = next(iter(train_iter))

    show_images(X.reshape(batch_size, 3, 32, 32).permute(0,2,3,1), 2, 2, titles=[i.item() for i in y])
    d2l.plt.show()
