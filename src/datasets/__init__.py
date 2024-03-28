#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 30-Jan-2023
# Last Modified: 30-Jan-2023
#
# Adapted from FLTK-testbed: https://github.com/JMGaljaard/fltk-testbed
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

from .base_dataset import BaseDataset
from .mnist import MNIST
from .cifar10 import CIFAR10
from .fmnist import FashionMNIST
from .cinic10 import CINIC10
from .svhn import SVHN


def get_dataset(name: str):
    
    available_datasets = {
        'mnist': MNIST,
        'cifar10': CIFAR10,
        'fmnist': FashionMNIST,
        'cinic10': CINIC10,
        'svhn': SVHN,
    }

    if name in available_datasets.keys():
        return available_datasets[name]
    else:
        raise NotImplementedError