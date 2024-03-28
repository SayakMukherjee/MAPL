#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 30-Jan-2023
# 
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

import torch.nn as nn
import numpy as np


class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()

    def forward(self, *input):
        """Forward pass logic

        Raises:
            NotImplementedError
        """
        raise NotImplementedError