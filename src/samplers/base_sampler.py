#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 30-Jan-2023
#
# Adapted from FLTK-testbed: https://github.com/JMGaljaard/fltk-testbed
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

import random
from typing import Iterator, List
import numpy as np

from utils import Config
from torch.utils.data import DistributedSampler, Dataset

class BaseSampler(DistributedSampler):

    def __init__(self, dataset: Dataset, num_replicas: int = None, rank: int = None, seed: int = 0) -> None:
        super().__init__(dataset, num_replicas, rank, seed)

        self.client_id = rank
        self.n_clients = num_replicas
        self.n_labels = len(dataset.classes)
        self.seed = seed

        self.epoch_size = 1.0

    def order_by_label(self, dataset: Dataset):
        """ Order the indices by label

        Args:
            dataset (Dataset): _description_

        Returns:
            _type_: _description_
        """
        ordered_by_label: List[List[int]] = [[] for _ in range(len(dataset.classes))]
        for index, target in enumerate(dataset.targets):
            ordered_by_label[target].append(index)
        
        sorted_indices = []
        for target in range(len(ordered_by_label)):
            sorted_indices.extend(ordered_by_label[target])

        return ordered_by_label, sorted_indices

    def set_epoch_size(self, epoch_size: float) -> None:
        """ Sets the epoch size as relative to the local amount of data.
        1.5 will result in the __iter__ function returning the available
        indices with half appearing twice.
        Args:
            epoch_size (float): relative size of epoch
        """
        self.epoch_size = epoch_size

    def __iter__(self) -> Iterator[int]:

        random.seed(self.rank + self.epoch)
        epochs_todo = self.epoch_size
        indices = []
        while (epochs_todo > 0.0):
            random.shuffle(self.indices)
            if epochs_todo >= 1.0:
                indices.extend(self.indices)
            else:
                end_index = int(round(len(self.indices) * epochs_todo))
                indices.extend(self.indices[:end_index])

            epochs_todo = epochs_todo - 1

        ratio = len(indices) / float(len(self.indices))
        np.testing.assert_almost_equal(ratio, self.epoch_size, decimal=2)

        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)