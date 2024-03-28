#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 05-Apr-2023
#
# ---------------------------------------------------------------------------
# Helper functions for centralized setup
# ---------------------------------------------------------------------------

import logging

from .base_comm import BaseComm
from utils import Config

class Centralized(BaseComm):

    def __init__(self, config: Config, all_classes: list, client_classes: dict):
        super(Centralized, self).__init__()
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

        for params in client_list[0].net.model.state_dict():
            if 'weight' in params or 'bias' in params:

                count = 0

                for _, client_i in enumerate(client_list):

                    count += 1

                    if params in agg_state_dict:
                        agg_state_dict[params] += client_i.net.model.state_dict()[params].clone()
                    else:
                        agg_state_dict[params] = client_i.net.model.state_dict()[params].clone()

                agg_state_dict[params] /= count
        
        return agg_state_dict

    def classifier_aggregation_weighted(self, client_list: list):
        """Aggregation of classifier layers

        Args:
            client_list (list): _description_

        Returns:
            _type_: _description_
        """

        agg_state_dict = dict()

        for params in client_list[0].net.classifier.state_dict():
            if 'weight' in params or 'bias' in params:

                count = 0

                for _, client_i in enumerate(client_list):

                    count += 1

                    if params in agg_state_dict:
                        agg_state_dict[params] += client_i.net.classifier.state_dict()[params].clone()
                    else:
                        agg_state_dict[params] = client_i.net.classifier.state_dict()[params].clone()

                agg_state_dict[params] /= count
        
        return agg_state_dict
      
    def projector_aggregation_weighted(self, client_list: list):
        """Aggregation of projection layers

        Args:
            client_list (list): _description_

        Returns:
            _type_: _description_
        """

        agg_state_dict = dict()

        for params in client_list[0].net.projection.state_dict():
            if 'weight' in params or 'bias' in params:

                count = 0

                for _, client_i in enumerate(client_list):

                    count += 1

                    if params in agg_state_dict:
                        agg_state_dict[params] += client_i.net.projection.state_dict()[params].clone()
                    else:
                        agg_state_dict[params] = client_i.net.projection.state_dict()[params].clone()

                agg_state_dict[params] /= count
        
        return agg_state_dict

    def proto_aggregation_weighted(self, local_protos_dict: dict, client_list: list):

        self.logger.info('Global proto aggregation')

        agg_protos_label = dict()
        for client_id in local_protos_dict:
            local_protos = local_protos_dict[client_id]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label][0])
                else:
                    agg_protos_label[label] = [local_protos[label][0]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label