#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 05-Apr-2023
#
# ---------------------------------------------------------------------------
# Implementation of MAPL
# ---------------------------------------------------------------------------

import wandb
import logging
import torch
import copy
import pickle
import random
import numpy as np
import cvxpy as cp
import networkx as nx
import torch.nn.functional as F

from pathlib import Path
from comm import Centralized, Decentralized
from models import BaseNet, SimCLRLoss, ProtoConLoss, ProtoUniLoss, get_model, load_model, save_model
from datasets import BaseDataset, get_dataset
from utils import Config

__fedproto_models__ = ['customresnet18', 'customresnet34', 'customresnet50', 'customresnet101', 'customresnet152', 'cnnmnist', 'cnnfemnist']
__fedclassavg_models__ = ['resnet18', 'shufflenet', 'googlenet', 'alexnet']
__fedcma_models__ = ['cnn1', 'cnn2', 'mlp1', 'mlp2']
__criterions__ = {
    'nllloss': torch.nn.NLLLoss,
    'crossentropy': torch.nn.CrossEntropyLoss,
    'mse': torch.nn.MSELoss,
    'supcon': SimCLRLoss,
    'con': SimCLRLoss,
    'protocon': ProtoConLoss,
    'protouni': ProtoUniLoss
}

class MAPL():
    """
    Distributed Trainer
    """

    def __init__(self, config: Config):
        """Initialize trainer class

        Args:
            config (Config): experiment configuration file
        """
            
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.device = config.settings['configurations']['device']
        self.embedding_size = self.config.settings['hyperParameters']['embeddingSize']
        self.isDecentralized = self.config.settings["type"] == "decentralized"

        self.export_path = Path(self.config.settings['configurations']['output_path'])\
                    .joinpath(self.config.settings['type'])\
                    .joinpath(self.config.settings['experiment_name'])

        if not Path.exists(self.export_path):
            Path.mkdir(self.export_path, parents=True)

        self.num_clients = self.config.settings['configurations']['num_clients']

        self.clients, client_classes, all_classes = self.__initialize_clients()

        self.comm = None
        if self.isDecentralized:
            self.comm = Decentralized(self.config, client_classes=client_classes, all_classes=all_classes)
        else:
            self.comm = Centralized(self.config, client_classes=client_classes, all_classes=all_classes)

        if self.config.settings['train']:
            self.exec_rounds() # training rounds

        self.test() # test

    def __initialize_clients(self):
        """Initialize client models and datasets

        Returns:
            list: list of client objects
        """
        models = self.config.settings['modelConfiguration']['models'].split(',')

        isFedProtoModels = set(models).issubset(set(__fedproto_models__))
        isFedClassAvgModels = set(models).issubset(set(__fedclassavg_models__))
        isFedCMAModels = set(models).issubset(set(__fedcma_models__))

        client_classes = {}
        all_classes = []
        confidence = [0.] * self.num_clients

        clients = []
        
        for client_id in np.arange(self.num_clients):

            dataset = get_dataset(self.config.settings['modelConfiguration']['dataset'])(self.config, client_id, isSSL=True)

            client_classes[client_id] = dataset.classes

            for class_name in dataset.classes:
                if class_name not in all_classes:
                    all_classes.append(class_name)

            confidence[client_id] = dataset.num_samples

            model_name = models[client_id % len(models)]

            if isFedProtoModels:

                if 'resnet' in model_name:
                    choice = random.uniform(0,1)

                    if choice > 0.5:
                        stride = [1,4]
                    else:
                        stride = [2,2]

                    out_channels = None
                else:
                    stride = None
                    out_channels = np.random.choice([18, 20, 22])

                model_args = {'img_channels': dataset.img_channels, 
                            'out_channels': out_channels, 
                            'num_classes': dataset.num_classes, 
                            'embedding_size': self.embedding_size,
                            'stride': stride,
                            'model_name': model_name}
                
            elif isFedClassAvgModels:

                model_args = {'img_channels': dataset.img_channels,
                            'num_classes': dataset.num_classes, 
                            'embedding_size': self.embedding_size,
                            'dataname': self.config.settings['modelConfiguration']['dataset'],
                            'model_name': model_name}
                
            elif isFedCMAModels:

                model_args = {'img_channels': dataset.img_channels,
                              'input_size': dataset.input_size,
                            'num_classes': dataset.num_classes, 
                            'embedding_size': self.embedding_size,
                            'dataname': self.config.settings['modelConfiguration']['dataset'],
                            'model_name': model_name}
                
            else:

                raise NotImplementedError

            client_model = get_model(model_name)(model_args, use_projection=True)

            client = MAPLClients(client_id, self.config, client_model, dataset)

            neighbours = torch.ones(self.num_clients).to(self.device) # fully connected graph

            client.init_weight(neighbours)

            clients.append(client)

            self.logger.info(f'User: {client_id + 1} assigned {model_name}')

            wandb.define_metric("loss_client_" + str(client_id + 1), summary="min")
            wandb.define_metric("acc_client_" + str(client_id + 1), summary="max")

        self.logger.info(client_classes)
        self.logger.info(all_classes)

        # compute confidence based on local dataset samples vs total
        self.confidence = torch.tensor(confidence).to(self.device)
        self.confidence = self.confidence / torch.sum(self.confidence)
        self.logger.info(self.confidence)

        # send the computed confidence array to all clients
        for client_id in np.arange(self.num_clients):
            clients[client_id].set_confidence(self.confidence)
        
        return clients, client_classes, all_classes
    
    def update_local_prototypes(self):

        for idx, client in enumerate(self.clients):

            client.net.prototypes.data.copy_(self.global_protos[idx])
            client.net.prototypes.to(self.device)
    
    def exec_rounds(self):
        """Execute training rounds
        """

        self.logger.info('Starting execution rounds')

        clients = np.arange(self.num_clients)

        num_rounds = self.config.settings['learningParameters']['rounds']

        train_loss, train_accuracy = [], []

        num_warmup = self.config.settings['learningParameters']['warmUpEpochs']

        for round in range(num_rounds):

            self.logger.info(f'| Global Training Round : {round + 1} |')

            local_accuracy, local_losses = [], []

            # update local prototypes
            if round > 0:
                self.update_local_prototypes()

            for idx, cur_client in enumerate(clients):

                loss, acc = self.clients[cur_client].train(round = round)

                local_losses.append(copy.deepcopy(loss['total']))
                local_accuracy.append(copy.deepcopy(acc))

                self.logger.info('| User: {} | \tLoss: {:.3f} | Acc: {:.3f}'.format(idx + 1, loss['total'], acc))

                wandb.log({"acc_client_" + str(idx + 1): acc, "loss_client_" + str(idx + 1): loss['total']})

            # compute weights after warmup rounds
            if round > num_warmup - 1:
                for idx, cur_client in enumerate(clients):
                    self.clients[cur_client].update_weights(self.clients)

            # update global prototypes
            if self.config.settings['networkConfiguration']['aggregation'] == 'gossip':
                raise NotImplementedError

            else:
                self.global_protos = self.comm.protolayer_aggregation_weighted(self.clients)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            acc_avg = sum(local_accuracy) / len(local_accuracy)
            train_accuracy.append(acc_avg)

            self.logger.info(f'Global Loss {loss_avg} | Global Accuracy {acc_avg}')

            wandb.log({"acc": acc_avg, "loss": loss_avg})
                            
            self.test()

    def test(self):
        """Test all clients
        """

        self.logger.info('Starting test')

        accuracy_local = []
        loss_local = []

        clients = np.arange(self.num_clients)

        for client_id in clients:
            
            client_acc_local, client_loss_local = self.clients[client_id].test()

            accuracy_local.append(client_acc_local)
            loss_local.append(client_loss_local)

        self.logger.info(f'For all users, mean of test acc is {round(np.mean(accuracy_local), 5)}, std of test acc is {round(np.std(accuracy_local), 5)}')
        self.logger.info(f'For all users, mean of test loss is {round(np.mean(loss_local), 5)}, std of test loss is {round(np.std(loss_local), 5)}')

        best_idx = np.argmax(accuracy_local)
        self.logger.info(f'Best performing user is {best_idx + 1} | Accuracy {round(accuracy_local[best_idx], 5)} Loss {round(loss_local[best_idx], 5)}' )

        worst_idx = np.argmin(accuracy_local)
        self.logger.info(f'Worst performing user is {worst_idx + 1} | Accuracy {round(accuracy_local[worst_idx], 5)} Loss {round(loss_local[worst_idx], 5)}' )

        wandb.log({"test_acc_mean": round(np.mean(accuracy_local),5), 
                   "test_acc_std": round(np.std(accuracy_local),5),
                   "best_test_acc": round(accuracy_local[best_idx],5),
                   "worst_test_acc": round(accuracy_local[worst_idx], 5)
                   })  
    
class MAPLClients():
    """
    Client Trainer
    """

    def __init__(self, id: int, config: Config, model: BaseNet, dataset: BaseDataset):
        """Initialize client trainer

        Args:
            id (int): client id
            config (Config): experiment configuration
            model (BaseNet): client model
            dataset (BaseDataset): client dataset
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        self.id = id
        self.config = config
        self.dataset = dataset
        self.classes = dataset.classes
        self.device = config.settings['configurations']['device']
        self.num_rounds = self.config.settings['learningParameters']['rounds']
        self.net = model
        self.model_path = Path(self.config.settings['configurations']['output_path'])\
                            .joinpath(self.config.settings['type'])\
                            .joinpath(self.config.settings['experiment_name'])\
                            .joinpath('chkpts')
        
        if not Path.exists(self.model_path):
            Path.mkdir(self.model_path, parents=True)

        if self.config.settings['resume'] or (not self.config.settings['train']):
            self.logger.info(f'| User: {self.id} | Loading from checkpoint')
            self.net = load_model(self.net, self.model_path.joinpath('client_' + str(self.id) + '.tar'))
        self.net.to(self.device)

    def init_weight(self, neighbours: torch.Tensor):
        
        self.neighbours = neighbours

        if self.config.settings['resume'] or (not self.config.settings['train']):
            self.weights = torch.load(self.model_path.joinpath('client_' + str(self.id) + '_weights.tar'))
        else:
            self.weights = (self.neighbours / torch.sum(self.neighbours)).to(self.device)

        self.logger.info(f'| User: {self.id} | Weights {self.weights}')

        # remove self loop
        self.neighbours[self.id]  = 0
        self.logger.info(f'| User: {self.id} | Neighbours {self.neighbours}')

        self.degree = torch.sum(self.weights).to(self.device) - self.weights[self.id]
        self.logger.info(f'| User: {self.id} | Degree {self.degree}')

    def set_confidence(self, confidence: torch.Tensor):
        self.confidence = confidence

    def proj_simplex(self, v, z=1):
        """Compute the closest point (orthogonal projection) on the
        generalized `(n-1)`-simplex of a vector :math:`\mathbf{v}` wrt. to the Euclidean
        distance, thus solving:

        .. math::
            \mathcal{P}(w) \in \mathop{\arg \min}_\gamma \| \gamma - \mathbf{v} \|_2

            s.t. \ \gamma^T \mathbf{1} = z

                \gamma \geq 0

        If :math:`\mathbf{v}` is a 2d array, compute all the projections wrt. axis 0

        .. note:: This function is backend-compatible and will work on arrays
            from all compatible backends.

        Reference: https://pythonot.github.io/_modules/ot/utils.html#proj_simplex

        Parameters
        ----------
        v : {array-like}, shape (n, d)
        z : int, optional
            'size' of the simplex (each vectors sum to z, 1 by default)

        Returns
        -------
        h : ndarray, shape (`n`, `d`)
            Array of projections on the simplex
        """
        n = v.shape[0]
        if v.ndim == 1:
            d1 = 1
            v = v[:, None]
        else:
            d1 = 0
        d = v.shape[1]

        # sort u in ascending order
        u, _ = torch.sort(v, axis=0)

        # take the descending order
        u = torch.flip(u, (0,))

        cssv = torch.cumsum(u, 0) - z
        
        ind = torch.arange(n, device=v.device)[:, None] + 1

        cond = u - cssv / ind > 0

        rho = torch.sum(cond, 0, keepdim=False)

        theta = cssv[rho - 1, torch.arange(d)] / rho

        w = torch.maximum(v - theta[None, :], torch.zeros(v.shape, dtype=v.dtype, device=v.device))
        
        if d1:
            return w[:, 0]
        else:
            return w

    def update_weights(self, client_list: dict):

        own_classifier = self.net.classifier.net.fcout.weight.detach()

        with torch.no_grad():
            similarity = - torch.ones(self.neighbours.shape).to(self.device)

            for idx, is_neighbor in enumerate(self.neighbours):

                if is_neighbor > 0.1: 

                    neigh_classifier = client_list[idx].net.classifier.net.fcout.weight.detach()

                    similarity[idx] = F.cosine_similarity(own_classifier, neigh_classifier, dim=-1).mean()

            # max normalization
            similarity /= torch.max(similarity)

            # self similarity is always 1
            similarity[self.id] = 1.

            # to minimize negative similarity
            similarity *= -1
        
        edge_weights = self.weights.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([edge_weights], lr = 0.1,
                                    weight_decay = 0,
                                    betas=(0.5, 0.999))

        loss = 0.5 * (similarity * edge_weights * self.confidence).sum() + 0.1 * ( self.config.settings['hyperParameters']['graphl2weight'] * edge_weights.norm(p=2) - torch.log(self.degree + 1e-6))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            edge_weights.data = self.proj_simplex(edge_weights)
            
        self.degree = torch.sum(edge_weights).to(self.device) - edge_weights[self.id]
        
        self.weights = edge_weights.clone().detach()

        torch.save(self.weights, self.model_path.joinpath('client_' + str(self.id) + '_weights.tar'))

    def train(self, round: int = 0, warm_up: bool = False, verbose=False):
        """Training loop

        Returns:
            epoch_loss (dict): dictionary containing total loss and each indivial components
            acc_val (float): accuracy of the last batch of the last epoch 
        """

        # Set mode to train model
        self.net.train()

        # Set optimizer for the local updates
        if self.config.settings['hyperParameters']['optimizerConfig']['type'] == 'SGD':
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config.settings['hyperParameters']['optimizerConfig']['learningRate'],
                                        momentum=self.config.settings['hyperParameters']['optimizerConfig']['momentum'])
        elif self.config.settings['hyperParameters']['optimizerConfig']['type'] == 'Adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.settings['hyperParameters']['optimizerConfig']['learningRate'],
                                         weight_decay=self.config.settings['hyperParameters']['optimizerConfig']['weightDecay'],
                                         betas=(0.5, 0.999))
        
        criterion = __criterions__[self.config.settings['modelConfiguration']['loss']]().to(self.device)

        apply_log_softmax = False
        if self.config.settings['modelConfiguration']['loss'] == 'nllloss':
            apply_log_softmax = True

        contrastive_loss_fn = __criterions__[self.config.settings['modelConfiguration']['sslloss']](temperature=self.config.settings['hyperParameters']['temp']).to(self.device)

        proto_loss_fn = __criterions__['protocon'](temperature=self.config.settings['hyperParameters']['temp']).to(self.device)

        proto_uni_loss_fn = __criterions__['protouni'](temperature=self.config.settings['hyperParameters']['temp']).to(self.device)
        
        trainloader = self.dataset.get_train_loader()

        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        num_epochs = self.config.settings['learningParameters']['epochsPerRound']

        for iter in range(num_epochs):

            batch_loss = {'total':[],'1':[], '2':[], '3':[]}

            for batch_idx, (indexes, images, labels) in enumerate(trainloader):

                images = torch.cat([images[0], images[1]], dim=0)
                images, labels, indexes = images.to(self.device), labels.to(self.device), indexes.to(self.device)
                # bsz = labels.shape[0]

                index = labels
                index = index.repeat(2) # considering 2 views

                projections, outputs = self.net(images, apply_log_softmax)

                # local supervised contrastive loss
                loss1 = contrastive_loss_fn(projections, index) #supcontrast
                
                # similarity with class-wise prototypes
                preds = self.net.get_assignment(projections)
                loss2 = proto_loss_fn(preds, index)

                # prototype uniformity on hypersphere
                loss3 = proto_uni_loss_fn(self.net.prototypes)

                # cross-entropy loss
                loss4 = criterion(outputs, index)
                                
                loss = loss1 + loss2 + loss3 + loss4

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, y_hat = outputs.max(1)
                acc_val = torch.eq(y_hat, index).float().mean()
                
                if verbose:
                    self.logger.info('| User: {} | Local Epoch : {} | [{:.0f}%]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        self.id, iter, 
                        100. * (batch_idx + 1) / len(trainloader),
                        loss.item(),
                        acc_val.item()))
                
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())

            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        save_model(self.net, self.model_path.joinpath('client_' + str(self.id) + '.tar'))

        return epoch_loss, acc_val.item()

    def test(self):

        # Set mode to train model
        self.net.eval()

        accuracy_local, loss_local = 0.0, 0.0

        loss, total, correct = 0.0, 0.0, 0.0

        criterion = __criterions__[self.config.settings['modelConfiguration']['loss']]().to(self.device)

        apply_log_softmax = False
        if self.config.settings['modelConfiguration']['loss'] == 'nllloss':
            apply_log_softmax = True

        testloader = self.dataset.get_test_loader()

        # test (use local loss)
        for batch_idx, (_, images, labels) in enumerate(testloader):

            images, labels = images.to(self.device), labels.to(self.device)

            _, outputs = self.net(images, apply_log_softmax)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy_local = correct / total
        loss_local = loss / total
        self.logger.info(f'| User: {self.id} | Global Test: Accuracy {round(accuracy_local, 3)} Loss {round(loss_local, 3)}')

        return accuracy_local, loss_local