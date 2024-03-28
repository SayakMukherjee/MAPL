#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 30-Jan-2023
#
# Adapted from FLTK-testbed: https://github.com/JMGaljaard/fltk-testbed
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

from .base_sampler import BaseSampler
from .custom_sampler import CustomSampler

def get_sampler(name: str):
    
    available_samplers = {
        'custom': CustomSampler
    }

    if name in available_samplers.keys():
        return available_samplers[name]
    else:
        raise NotImplementedError