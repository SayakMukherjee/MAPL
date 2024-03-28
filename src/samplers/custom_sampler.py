#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 14-Feb-2023
#
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

import logging
import random
import numpy as np

from typing import Iterator, List
from .base_sampler import BaseSampler
from torch.utils.data import Dataset


class CustomSampler(BaseSampler):

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int, seed: int, sampler_args: dict, isTrain: bool = True):
        super().__init__(dataset, num_replicas, rank, seed)

        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(f'User {self.client_id + 1}')

        # assign classes
        assigned_classes, _, _ = self._assign_classes(seed, 
                                                    self.n_clients,
                                                    len(dataset.classes),
                                                    sampler_args['num_clusters'],
                                                    sampler_args['scenario'],
                                                    sampler_args['num_overlap'])

        indices = []
        ordered_by_label, _ = self.order_by_label(dataset)
        
        self.classes = assigned_classes[self.client_id]

        np.random.seed(seed + self.client_id * 1000)
        random.seed(seed + self.client_id * 1000)
        
        if not isTrain:
            k = np.ones(len(self.classes)).astype(int) * sampler_args['test_shots']
            k_max = sampler_args['test_shots']

        else:
            if sampler_args['scenario'] in [1, 2]:
                k = np.ones(len(self.classes)).astype(int) * sampler_args['shots']
            else:
                k = np.random.randint(100, 300, len(self.classes))
            k_max = sampler_args['train_shots_max']

        for idx, class_name in enumerate(assigned_classes[self.client_id]):

            begin = self.client_id * k_max

            selection = ordered_by_label[class_name][begin:begin+k[idx]]

            indices.extend(selection)

            self.logger.info(f'Assigned {len(selection)} samples from class {class_name}')

        random.shuffle(indices)  # shuffle indices to spread the labels

        self.indices = indices

    def _assign_classes(self, seed: int, num_users: int, num_classes: int, num_clusters: int, scenario: int, num_overlap: int):

        random.seed(seed)
        np.random.seed(seed)

        # shuffle classes
        class_array = np.arange(num_classes)
        # np.random.shuffle(class_array)

        # distribute class among clusters
        comm2class = dict()

        if scenario in [1, 3]:

            class_per_comm = num_classes // num_clusters

            # without overlap
            assigned_idx = 0
            for community_id in range(num_clusters):
                comm2class[community_id] = []
                for _ in range(class_per_comm):
                    comm2class[community_id].append(class_array[assigned_idx])
                    assigned_idx += 1

            comm2class[community_id].extend(class_array[assigned_idx:])

        elif scenario in [2, 4]:

            class_per_comm = (num_classes + (num_overlap * (num_clusters - 1))) // num_clusters

            # with overlap
            assigned_idx = 0
            for community_id in range(num_clusters - 1):
                comm2class[community_id] = []

                for _ in range(class_per_comm):
                    comm2class[community_id].append(class_array[assigned_idx])
                    assigned_idx += 1

                assigned_idx -= num_overlap
            
            community_id += 1
            comm2class[community_id] = []
            comm2class[community_id].extend(class_array[assigned_idx:])

        # distribute clients among clusters
        cluster_length = num_users // num_clusters
        node2comm = dict()

        assigned_clients = 0
        cluster_id = 0
        for member in range(num_users):
            node2comm[member] = cluster_id
            assigned_clients += 1

            if (assigned_clients == cluster_length) and (cluster_id != num_clusters - 1):
                cluster_id += 1
                assigned_clients = 0

        num_clients_per_class = [0] * num_classes
        assigned_classes = {}
        class2clients = {k: [] for k in range(num_classes)}

        for idx in range(num_users):
            curr_community = node2comm[idx]
            assigned_classes[idx] = comm2class[curr_community].copy()

            for class_idx in assigned_classes[idx]:
                num_clients_per_class[class_idx] += 1
                class2clients[class_idx].append(idx)

        return assigned_classes, class2clients, num_clients_per_class