#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 30-Jan-2023
# Last Modified: 30-Jan-2023
#
# Adapted from FLTK-testbed: https://github.com/JMGaljaard/fltk-testbed
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

from .fedproto import FedProto
from .fedclassavg import FedClassAvg
from .mapl import MAPL
from .fedavgsim import FedAvgSim

def get_method(name: str):
    
    available_methods = {
        'fedproto': FedProto,
        'fedavgsim': FedAvgSim,
        'fedclassavg': FedClassAvg,
        'mapl': MAPL,
    }

    return available_methods[name]