#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 30-Jan-2023
#
# Adapted from FedProto: https://github.com/yuetan031/fedproto
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from .base_net import BaseNet
from collections import OrderedDict


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'customresnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'customresnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'customresnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'customresnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'customresnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CustomResNet(nn.Module):

    def __init__(self, model_args: dict, zero_init_residual: bool = False):
        super(CustomResNet, self).__init__()

        self.stride = model_args['stride']

        if model_args['model_name'] == 'customresnet18':
            self.block = BasicBlock
            self.layers = [2, 2, 2, 2]

        elif model_args['model_name'] == 'customresnet34':
            self.block = BasicBlock
            self.layers = [3, 4, 6, 3]

        elif model_args['model_name'] == 'customresnet50':
            self.block = Bottleneck
            self.layers = [3, 4, 6, 3]

        elif model_args['model_name'] == 'customresnet101':
            self.block = Bottleneck
            self.layers = [3, 4, 23, 3]

        elif model_args['model_name'] == 'customresnet152':
            self.block = Bottleneck
            self.layers = [3, 8, 36, 3]

        else:
            raise NotImplementedError

        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride = self.stride[0], padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = self.stride[1], padding=1)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, model_args['embedding_size'], self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(model_args['embedding_size'] * self.block.expansion, model_args['num_classes'])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, apply_log_softmax = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        embeddings = self.layer4(x)

        x = self.avgpool(embeddings)
        x = x.view(x.size(0), -1)
        outputs = self.fc(x)

        if apply_log_softmax:
            outputs = F.log_softmax(outputs, dim=1)

        return embeddings, outputs

def load_resnet_weights(model: BaseNet, model_args:dict):

    model_key = model_args['model_name']

    initial_weight = model_zoo.load_url(model_urls[model_key])
    initial_weight_1 = model.state_dict()
    for key in initial_weight.keys():
        if key[0:3] == 'fc.' or key[0:5]=='conv1' or key[0:3]=='bn1':
            initial_weight[key] = initial_weight_1[key]

    model.load_state_dict(initial_weight)

    return model

class AlexnetMNIST(BaseNet):  

    def __init__(self, img_channels: int, embedding_size: int):
        super(AlexnetMNIST, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding='same'),
            nn.ReLU()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((3,3))
            

        self.fc = nn.Sequential(OrderedDict([
            ('drop1', nn.Dropout(p=0.5)),
            ('fcin', nn.Linear(256 * 3 * 3, 512)),
            ('relu', nn.ReLU()),
            ('fcout', nn.Linear(512, embedding_size)),
        ]))

    def forward(self, x, apply_log_softmax = False):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.fc(out)
        
        return out