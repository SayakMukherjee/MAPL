#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 06-Apr-2023
# 
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

import torch.nn as nn
import numpy as np

from abc import abstractmethod


class BaseComm(nn.Module):
    """Base class for all communications."""

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def proto_aggregation(self, local_protos_dict: dict):
        raise NotImplementedError