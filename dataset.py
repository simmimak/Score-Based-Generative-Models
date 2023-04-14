import torch
import torchvision.transforms as T
import torchvision.datasets
import numpy as np
from torch.utils.data import Subset


def get_train_data(conf):
    if conf.dataset.name == 'mnist':

        transform = T.Compose(
            [
                #T.RandomHorizontalFlip(),
                T.ToTensor(),
                #T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        transform_test = T.Compose(
            [
                T.ToTensor(),
                #T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        train_set = torchvision.datasets.MNIST(conf.dataset.path,
                                                 train=True,
                                                 transform=transform,
                                                 download=True)
        valid_set = torchvision.datasets.MNIST(conf.dataset.path,
                                                  train=True,
                                                  transform=transform_test,
                                                  download=True)

        num_train  = len(train_set)
        indices    = torch.randperm(num_train).tolist()
        valid_size = int(np.floor(0.05 * num_train))

        train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

        train_set = Subset(train_set, train_idx)
        valid_set = Subset(valid_set, valid_idx)


        test_set = torchvision.datasets.MNIST(conf.dataset.path,
                                                 train=False,
                                                 transform=transform_test,
                                                 download=True)
   
    else:
        raise FileNotFoundError

    return train_set, valid_set, test_set