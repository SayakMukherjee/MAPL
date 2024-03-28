#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 30-Jan-2023
#
# Adapted from FLTK-testbed: https://github.com/JMGaljaard/fltk-testbed
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

from utils import Config
from abc import abstractmethod

class BaseDataset():

    def __init__(self, config: Config):
        self.config = config

        self.train_sampler = None
        self.test_sampler = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.img_channels = None
        self.num_classes = None
        self.classes = []

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_train_sampler(self):
        return self.train_sampler

    def get_test_sampler(self):
        return self.test_sampler

    @abstractmethod
    def init_train_dataset(self):
        pass

    @abstractmethod
    def init_test_dataset(self):
        pass