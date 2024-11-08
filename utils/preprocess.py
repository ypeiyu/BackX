import torch

img_size_dict = {
    'imagenet': (3, 224, 224),
    'cifar10': (3, 32, 32),
    'cifar100': (3, 32, 32),
    'gtsrb': (3, 32, 32),
    'imagenette': (3, 224, 224)
    # 'imagenette': (3, 256, 256)

}


mean_std_dict = {
    'imagenet': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    'imagenette': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    'cifar10': [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)],
    'cifar100': [(0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)],
    'gtsrb': [(0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)]
}


def preprocess(x, d_name='imagenet'):
    mean_std = mean_std_dict[d_name]
    mean, std = mean_std[0], mean_std[1]
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def undo_preprocess(x, d_name='imagenet'):
    mean_std = mean_std_dict[d_name]
    mean, std = mean_std[0], mean_std[1]
    assert x.size(1) == 3

    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y
