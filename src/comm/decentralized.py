#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 05-Apr-2023
#
# ---------------------------------------------------------------------------
# Helper functions for decentralized setup
# ---------------------------------------------------------------------------

import logging

from .base_comm import BaseComm
from utils import Config

class Decentralized(BaseComm):

    def __init__(self, config: Config, all_classes: list, client_classes: dict):
        super(Decentralized, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config

    def backbone_aggregation_weighted(self, client_list: list):
        """Aggregation of backbone layers

        Args:
            client_list (list): _description_

        Returns:
            _type_: _description_
        """

        agg_state_dict = dict()

        for idx_i, client_i in enumerate(client_list):

            agg_state_dict[idx_i] = dict()

            # locally learned weights
            weight_array = client_i.weights.detach()
            
            # own contribution
            own_state_dict = client_i.net.model.state_dict()
            for params in client_i.net.model.state_dict():
                if 'weight' in params or 'bias' in params:
                    agg_state_dict[idx_i][params] = own_state_dict[params].clone() * weight_array[idx_i]


            # neighbors contribution
            neighbors = client_i.neighbours

            for idx_j, client_j in enumerate(client_list):

                if (idx_j != idx_i) and (neighbors[idx_j] > 0.1):

                    neighbor_state_dict = client_j.net.model.state_dict()

                    for params in client_i.net.model.state_dict():

                        if 'weight' in params or 'bias' in params:
                            
                            agg_state_dict[idx_i][params] += neighbor_state_dict[params].clone() * weight_array[idx_j]
        
        return agg_state_dict

    def classifier_aggregation_weighted(self, client_list: list):
        """Aggregation of classifier layers

        Args:
            client_list (list): _description_

        Returns:
            _type_: _description_
        """

        agg_state_dict = dict()

        for idx_i, client_i in enumerate(client_list):

            agg_state_dict[idx_i] = dict()

            # locally learned weights
            weight_array = client_i.weights.detach()
            
            # own contribution
            own_state_dict = client_i.net.classifier.state_dict()
            for params in client_i.net.classifier.state_dict():
                if 'weight' in params or 'bias' in params:
                    agg_state_dict[idx_i][params] = own_state_dict[params].clone() * weight_array[idx_i]


            # neighbors contribution
            neighbors = client_i.neighbours

            for idx_j, client_j in enumerate(client_list):

                if (idx_j != idx_i) and (neighbors[idx_j] > 0.1):

                    neighbor_state_dict = client_j.net.classifier.state_dict()

                    for params in client_i.net.classifier.state_dict():

                        if 'weight' in params or 'bias' in params:
                            
                            agg_state_dict[idx_i][params] += neighbor_state_dict[params].clone() * weight_array[idx_j]
        
        return agg_state_dict
      
    def projector_aggregation_weighted(self, client_list: list):
        """Aggregation of projection layers

        Args:
            client_list (list): _description_

        Returns:
            _type_: _description_
        """

        agg_state_dict = dict()

        for idx_i, client_i in enumerate(client_list):

            agg_state_dict[idx_i] = dict()

            # locally learned weights
            weight_array = client_i.weights.detach()
            
            # own contribution
            own_state_dict = client_i.net.projection.state_dict()
            for params in client_i.net.projection.state_dict():
                if 'weight' in params or 'bias' in params:
                    agg_state_dict[idx_i][params] = own_state_dict[params].clone() * weight_array[idx_i]


            # neighbors contribution
            neighbors = client_i.neighbours

            for idx_j, client_j in enumerate(client_list):

                if (idx_j != idx_i) and (neighbors[idx_j] > 0.1):

                    neighbor_state_dict = client_j.net.projection.state_dict()

                    for params in client_i.net.projection.state_dict():

                        if 'weight' in params or 'bias' in params:
                            
                            agg_state_dict[idx_i][params] += neighbor_state_dict[params].clone() * weight_array[idx_j]
        
        return agg_state_dict
    
    def protolayer_aggregation_weighted(self, client_list: list):
        """Aggregation of prototypes

        Args:
            client_list (list): _description_

        Returns:
            _type_: _description_
        """

        agg_protolayer = dict()

        for idx_i, client_i in enumerate(client_list):

            # locally learned weights
            weight_array = client_i.weights.detach()

            # own contribution
            agg_protolayer[idx_i] = client_i.net.prototypes.data.clone() * weight_array[idx_i]

            # neighbors contribution
            neighbors = client_i.neighbours

            for idx_j, client_j in enumerate(client_list):
                if (idx_j != idx_i) and (neighbors[idx_j] > 0.1):
                    agg_protolayer[idx_i] += client_j.net.prototypes.data.clone() * weight_array[idx_j]

        return agg_protolayer

    def proto_aggregation_weighted(self, local_protos_dict: dict, client_list: list):
        """Average of class-wise prototypes

        Args:
            local_protos_dict (dict): _description_
            client_list (list): _description_

        Returns:
            _type_: _description_
        """

        agg_proto_dict = dict()

        for idx_i, client_i in enumerate(client_list):

            # neighbors contribution
            neighbors = client_i.neighbours

            agg_proto_dict[idx_i] = dict()

            for label in local_protos_dict[idx_i].keys():

                agg_proto_dict[idx_i][label] = [local_protos_dict[idx_i][label][0]]

                for idx_j, client_j in enumerate(client_list):

                    if (idx_j != idx_i) and (neighbors[idx_j] > 0.1) and (label in local_protos_dict[idx_j].keys()):
                        
                        agg_proto_dict[idx_i][label].append(local_protos_dict[idx_j][label][0])

                proto_list = agg_proto_dict[idx_i][label]
                if len(proto_list) > 1:
                    proto = 0 * proto_list[0].data
                    for i in proto_list:
                        proto += i.data
                    agg_proto_dict[idx_i][label] = [proto / len(proto_list)]
                else:
                    agg_proto_dict[idx_i][label] = [proto_list[0].data]

        return agg_proto_dict