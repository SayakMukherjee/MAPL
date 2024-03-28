#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 10-Apr-2023
#
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

import wandb
import torch

from .base_net import BaseNet
from pathlib import Path

def save_model(model: BaseNet, path: Path):
    
    return torch.save(model.state_dict(), path)

def load_model(model: BaseNet, path: Path):

    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    return model
