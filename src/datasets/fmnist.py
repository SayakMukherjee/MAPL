#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 30-Jan-2023
#
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

import logging

from .base_dataset import BaseDataset
from .utils import SimCLRCollate
from utils import Config
from samplers import get_sampler
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Callable, List, Optional, Type, Tuple, Union

MNIST_NORMALIZE = {"mean": (0.1307,), "std": (0.3081,)}

class DatasetWrapper(datasets.FashionMNIST):

    def __init__(self, 
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):
        super(DatasetWrapper, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        return (index, *data)

class FashionMNIST(BaseDataset):

    def __init__(self, config: Config, client_id: int, isSSL: bool = False, dataset_args: dict = None):
        super(FashionMNIST, self).__init__(config)
        
        self.logger = logging.getLogger(self.__class__.__name__)

        self.data_path = Path(self.config.settings['configurations']['data_path']).joinpath('fmnist')

        if not Path.exists(self.data_path):
            Path.mkdir(self.data_path, parents=True)

        self.batch_size = self.config.settings['hyperParameters']['batchSize']
        self.test_batch_size = self.config.settings['hyperParameters']['testBatchSize']
        self.data_sampler = str.lower(self.config.settings['learningParameters']['dataSampler'])
        self.num_workers = self.config.settings['hyperParameters']['num_workers']
        self.world_size = self.config.settings['configurations']['num_clients']
        self.seed = self.config.settings['configurations']['seed']
        self.client_id = client_id
        self.sampler_args = self.config.settings['learningParameters']['samplerArgs']
        self.img_channels = 1
        self.input_size = 28

        self.collate_fn = None
        self.transform = None

        if isSSL:
            self.logger.info(f"Using collate function")
            if dataset_args is None:
                self.collate_fn = SimCLRCollate(input_size=self.input_size,
                                                min_scale=0.2,
                                                gaussian_blur=0.0,
                                                random_gray_scale=0.0,
                                                normalize=MNIST_NORMALIZE
                                                )
            else:
                self.collate_fn = SimCLRCollate(input_size=self.input_size,
                                                min_scale = dataset_args['min_scale'],
                                                gaussian_blur = dataset_args['gaussian_blur'],
                                                random_gray_scale=0.0,
                                                normalize=MNIST_NORMALIZE
                                                )
        else:
            self.logger.info(f"Using transform")
            self.transform = transforms.Compose([transforms.RandomCrop(self.input_size, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))])
            
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.1307,), (0.3081,))])

        self.__init_train_dataset()
        self.__init_test_dataset()

        self.classes = self.train_sampler.classes
        self.num_samples = len(self.train_sampler.indices)

    def __init_train_dataset(self):

        self.logger.info(f"Loading FashionMNIST train data")

        if self.transform is None:
            self.train_dataset = DatasetWrapper(root = self.data_path, train = True, download = True)
        else:
            self.train_dataset = DatasetWrapper(root = self.data_path, train = True, download = True, transform = self.transform)    

        self.num_classes = len(self.train_dataset.classes)
        self.train_sampler = get_sampler(self.data_sampler)(self.train_dataset, self.world_size, self.client_id, self.seed, self.sampler_args)

        if self.collate_fn is None:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, drop_last=True)
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, sampler=self.train_sampler, drop_last=True)

    def __init_test_dataset(self):

        self.logger.info(f"Loading FashionMNIST test data")
        
        self.test_dataset = DatasetWrapper(root=self.data_path, train=False, download=True, transform=self.test_transform)

        self.test_sampler = get_sampler(self.data_sampler)(self.test_dataset, self.world_size, self.client_id, self.seed, self.sampler_args, isTrain=False)

        self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, sampler=self.test_sampler, num_workers=self.num_workers)
