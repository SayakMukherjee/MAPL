#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 05-Apr-2023
#
# Adapted from FedClassAvg: 
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

import math
import torch
import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from .custom_models import AlexnetMNIST
from .base_net import BaseNet
from collections import OrderedDict


class MLP(nn.Module):

    def __init__(self, embedding_size, num_classes):
        super(MLP, self).__init__()

        self.net = nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
                ('fcout', nn.Linear(embedding_size, num_classes))
            ]))

    def forward(self, x, apply_log_softmax = False):

        x = self.net(x)

        if apply_log_softmax:
            x = F.log_softmax(x, dim=1)

        return x

class Classifier(nn.Module):

    def __init__(self, embedding_size, num_classes):
        super(Classifier, self).__init__()

        self.net = nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
                ('fcout', nn.Linear(embedding_size, num_classes))
            ]))

    def forward(self, x, apply_log_softmax = False):

        x = self.net(x)

        if apply_log_softmax:
            x = F.log_softmax(x, dim=1)

        return x

class Projection(nn.Module):
    def __init__(self, dim, hidden_size, output_size, batch_norm_mlp=False):
        super().__init__()
        norm = nn.BatchNorm1d(hidden_size) if batch_norm_mlp else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            norm,
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)
    
class TorchVisionModels(BaseNet):

    def __init__(self, model_args: dict, use_projection: bool = False, use_classifier: bool = True):
        super(TorchVisionModels, self).__init__()
    
        self.num_classes = model_args['num_classes']
        self.embedding_size = model_args['embedding_size']
        self.dataname = model_args['dataname']
        self.model_name = model_args['model_name'] 
        self.use_projection = use_projection
        self.use_classifier = use_classifier

        self.prototypes = torch.nn.Parameter(torch.Tensor(self.num_classes, self.embedding_size))

        if self.model_name == 'resnet18':

            self.model = torchvision.models.resnet18()
            self.model.fc = nn.Sequential(OrderedDict([
                ('fcin', nn.Linear(512, self.embedding_size)),
            ]))

            if 'mnist' in self.dataname:
                self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        elif self.model_name == 'shufflenet':

            self.model = torchvision.models.shufflenet_v2_x1_0()
            self.model.fc = nn.Sequential(OrderedDict([
                ('fcin', nn.Linear(1024, self.embedding_size)),
            ]))

            if 'mnist' in self.dataname:
                self.model.conv1[0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        elif self.model_name == 'googlenet':

            self.model = torchvision.models.googlenet(init_weights=True)
            self.model.fc = nn.Sequential(OrderedDict([
                ('fcin', nn.Linear(1024, self.embedding_size)),
            ]))

            if 'mnist' in self.dataname:
                self.model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        elif self.model_name == 'alexnet':

            if 'imagenet' in self.dataname:
                self.model = torchvision.models.alexnet()
                self.model.classifier = nn.Sequential(nn.Dropout())
                self.model.fc = nn.Sequential(OrderedDict([
                    ('fcin', nn.Linear(256 * 256 * 3, self.embedding_size)),
                ]))

            elif 'mnist' in self.dataname:
                self.model = AlexnetMNIST(1, self.embedding_size)

            else:
                self.model = AlexnetMNIST(3, self.embedding_size)

        if self.use_projection:
            self.projection = Projection(dim=self.embedding_size, hidden_size=4096, output_size=self.embedding_size, batch_norm_mlp=True)

        if self.use_classifier:
            self.classifier = MLP(self.embedding_size, self.num_classes)

        self._reset_protos()

    def _reset_protos(self):

        stdv = 1. / math.sqrt(self.prototypes.size(1))
        self.prototypes.data.uniform_(-stdv, stdv)

    def forward(self, x, apply_log_softmax = False):

        embeddings = self.model(x)

        if embeddings.__class__.__name__ == 'GoogLeNetOutputs':
            embeddings = embeddings.logits

        if self.use_classifier:

            outputs = self.classifier(embeddings, apply_log_softmax)

            if self.use_projection:
                proj = self.projection(embeddings)

                return proj, outputs

            return embeddings, outputs

        else:

            proj = self.projection(embeddings)
            
            return embeddings, proj
        
    def get_assignment(self, x):
        
        x = F.normalize(x, dim=1)
        return F.linear(x, self.prototypes)