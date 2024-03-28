#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 30-Jan-2023
# Last Modified: 30-Jan-2023
#
# Adapted from FLTK-testbed: https://github.com/JMGaljaard/fltk-testbed
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

from .base_net import BaseNet
from .custom_models import CustomResNet, AlexnetMNIST, load_resnet_weights
from .losses import SimCLRLoss, ProtoConLoss, ProtoUniLoss
from .utils import load_model, save_model
from .torchvision_models import TorchVisionModels, Classifier

def get_model(name: str):
    
    available_models = {
        'customresnet': CustomResNet,
        'alexnetmnist': AlexnetMNIST,
        'resnet18': TorchVisionModels,
        'shufflenet': TorchVisionModels,
        'googlenet': TorchVisionModels,
        'alexnet': TorchVisionModels,
        'classifier': Classifier
    }

    return available_models[name]