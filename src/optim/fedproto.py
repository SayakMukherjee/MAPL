#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 05-Apr-2023
#
# ---------------------------------------------------------------------------
# Implementation of FedProto
# ---------------------------------------------------------------------------

import wandb
import logging
import torch
import copy
import random
import pickle
import networkx as nx
import numpy as np

from pathlib import Path
from comm import Centralized, Decentralized
from models import BaseNet, SimCLRLoss, get_model, load_model, save_model
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
    'con': SimCLRLoss
}

class FedProto():
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
        self.comm = None
        self.isDecentralized = self.config.settings["type"] == "decentralized"

        self.export_path = Path(self.config.settings['configurations']['output_path'])\
                            .joinpath(self.config.settings['type'])\
                            .joinpath(self.config.settings['experiment_name'])

        if not Path.exists(self.export_path):
            Path.mkdir(self.export_path, parents=True)

        self.num_clients = self.config.settings['configurations']['num_clients']

        # load network graph
        if self.config.settings['modelConfiguration']['learnGraphWeights']:
            self.logger.info('Loading adjacency')
            self.adj = self.get_adjacency()

        self.clients, client_classes, all_classes = self.__initialize_clients()

        if self.isDecentralized:
            self.comm = Decentralized(self.config, client_classes=client_classes, all_classes=all_classes)
        else:
            self.comm = Centralized(self.config, client_classes=client_classes, all_classes=all_classes)

        if self.config.settings['train']:
            self.exec_rounds() # training rounds
        # self.test() # test

    def get_adjacency(self):

        prob_edge = self.config.settings['networkConfiguration']['probEdgeCreation']

        if str.lower(self.config.settings['learningParameters']['dataSampler']) == 'community':
            import_path = Path(self.config.settings['configurations']['output_path']).joinpath('cP' + str(int(prob_edge*100)) + 'N' + str(self.num_clients) + '.pickle')
        else:
            import_path = Path(self.config.settings['configurations']['output_path']).joinpath('gP' + str(int(prob_edge*100)) + 'N' + str(self.num_clients) + '.pickle')

        if not Path.exists(Path(self.config.settings['configurations']['output_path'])):
            raise FileNotFoundError

        elif not Path.exists(import_path):
            raise FileNotFoundError

        G = None

        with open(import_path, 'rb') as fp:
            G = pickle.load(fp)

        adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes())) + np.eye(self.num_clients)
        
        return adj

    def __initialize_clients(self):
        """Initialize client models and datasets

        Returns:
            list: list of client objects
        """
        models = self.config.settings['modelConfiguration']['models'].split(',')

        isFedProtoModels = set(models).issubset(set(__fedproto_models__))
        isFedClassAvgModels = set(models).issubset(set(__fedclassavg_models__))
        isFedCMAModels = set(models).issubset(set(__fedcma_models__))

        clients = []

        client_classes = {}
        all_classes = []
        confidence = [0.] * self.num_clients
        
        for client_id in np.arange(self.num_clients):
            
            dataset = get_dataset(self.config.settings['modelConfiguration']['dataset'])(self.config, client_id)
            
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
                
                # Resolve resnet type to resnet class
                if 'resnet' in model_name:
                    model_name = 'customresnet'
                
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


            client_model = get_model(model_name)(model_args)

            client = FedProtoClients(client_id, self.config, client_model, dataset)

            if self.config.settings['modelConfiguration']['learnGraphWeights']: 
                neighbours = torch.tensor(self.adj[client_id]).to(self.device)
            else:
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
    
    def exec_rounds(self):
        """Execute training rounds
        """

        self.logger.info('Starting execution rounds')

        clients = np.arange(self.num_clients)
        
        self.global_protos = None

        if self.config.settings['resume']:
            with open(self.export_path.joinpath('globalProtos.pickle'), 'rb') as fp:
                self.global_protos = pickle.load(fp)
        else:
            # initialize
            if self.isDecentralized:
                for client_id in clients:
                    self.global_protos[client_id] = []
            else:
                self.global_protos = []
                

        num_rounds = self.config.settings['learningParameters']['rounds']

        train_loss, train_accuracy = [], []

        for round in range(num_rounds):

            local_accuracy, local_losses, local_protos = [], [], {}

            self.logger.info(f'| Global Training Round : {round + 1} |')

            proto_loss = 0

            for idx, cur_client in enumerate(clients):
                
                if self.isDecentralized:
                    loss, acc, protos = self.clients[cur_client].train(agg_protos = self.global_protos[cur_client])

                else:
                    loss, acc, protos = self.clients[cur_client].train(agg_protos = self.global_protos)

                local_losses.append(copy.deepcopy(loss['total']))
                local_accuracy.append(copy.deepcopy(acc))

                self.logger.info('| User: {} | \tLoss: {:.3f} | Acc: {:.3f}'.format(idx + 1, loss['total'], acc))

                wandb.log({"acc_client_" + str(idx + 1): acc, "loss_client_" + str(idx + 1): loss['total']})

                local_protos[cur_client] = protos

                proto_loss += loss['2']

            # update global protos
            self.global_protos = self.comm.proto_aggregation_weighted(local_protos, self.clients)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            acc_avg = sum(local_accuracy) / len(local_accuracy)
            train_accuracy.append(acc_avg)

            self.logger.info(f'Global Loss {loss_avg} | Global Accuracy {acc_avg}')

            wandb.log({"acc": acc_avg, "loss": loss_avg})

            with open(self.export_path.joinpath('globalProtos.pickle'), 'wb') as fp:
                pickle.dump(self.global_protos, fp)

            with open(self.export_path.joinpath('localProtos.pickle'), 'wb') as fp:
                pickle.dump(local_protos, fp)

            self.test()


    def test(self):
        """Test all clients
        """

        self.logger.info('Starting test')

        if not self.config.settings['train']:
            with open(self.export_path.joinpath('globalProtos.pickle'), 'rb') as fp:
                self.global_protos = pickle.load(fp)

        accuracy_local = []
        accuracy_proto = []
        loss_local = []
        loss_proto = []

        clients = np.arange(self.num_clients)

        for client_id in clients:
            
            if self.isDecentralized:
                client_acc_local, client_loss_local, client_acc_proto, client_loss_proto = self.clients[client_id].test(agg_protos = self.global_protos[client_id])
            else:
                client_acc_local, client_loss_local, client_acc_proto, client_loss_proto = self.clients[client_id].test(agg_protos = self.global_protos)

            accuracy_local.append(client_acc_local)
            loss_local.append(client_loss_local)

            accuracy_proto.append(client_acc_proto)
            loss_proto.append(client_loss_proto)

        self.logger.info(f'For all users (with protos), mean of test acc is {round(np.mean(accuracy_proto), 5)}, std of test acc is {round(np.std(accuracy_proto), 5)}')
        self.logger.info(f'For all users (with protos), mean of test loss is {round(np.mean(loss_proto), 5)}, std of test loss is {round(np.std(loss_proto), 5)}')
        
        best_idx = np.argmax(accuracy_proto)
        self.logger.info(f'Best performing user (with protos) is {best_idx + 1} | Accuracy {round(accuracy_proto[best_idx], 5)} Loss {round(loss_proto[best_idx], 5)}' )

        worst_idx = np.argmin(accuracy_proto)
        self.logger.info(f'Worst performing user (with protos) is {worst_idx + 1} | Accuracy {round(accuracy_proto[worst_idx], 5)} Loss {round(loss_proto[worst_idx], 5)}' )

        self.logger.info(f'For all users (w/o protos), mean of test acc is {round(np.mean(accuracy_local), 5)}, std of test acc is {round(np.std(accuracy_local), 5)}')
        self.logger.info(f'For all users (w/o protos), mean of test loss is {round(np.mean(loss_local), 5)}, std of test loss is {round(np.std(loss_local), 5)}')

        best_idx = np.argmax(accuracy_local)
        self.logger.info(f'Best performing user (w/o protos) is {best_idx + 1} | Accuracy {round(accuracy_local[best_idx], 5)} Loss {round(loss_local[best_idx], 5)}' )

        worst_idx = np.argmin(accuracy_local)
        self.logger.info(f'Worst performing user (w/o protos) is {worst_idx + 1} | Accuracy {round(accuracy_local[worst_idx], 5)} Loss {round(loss_local[worst_idx], 5)}' )

        wandb.log({"test_acc_mean": round(np.mean(accuracy_local),5), 
            "test_acc_std": round(np.std(accuracy_local),5),
            "best_test_acc": round(accuracy_local[best_idx],5),
            "worst_test_acc": round(accuracy_local[worst_idx], 5)
            })
    
class FedProtoClients():
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

        self.weights = (self.neighbours / torch.sum(self.neighbours)).to(self.device)
        self.weights.requires_grad = True
        self.logger.info(f'| User: {self.id} | Weights {self.weights}')

        # remove self loop
        self.neighbours[self.id]  = 0
        self.logger.info(f'| User: {self.id} | Neighbours {self.neighbours}')

        self.degree = torch.sum(self.weights).to(self.device) - self.weights[self.id]
        self.logger.info(f'| User: {self.id} | Degree {self.degree}')

    def set_confidence(self, confidence: torch.Tensor):
        self.confidence = confidence

    def train(self, agg_protos: list, verbose: bool = False):
        """Training loop

        Args:
            agg_protos (list): globally aggregated prototypes (class-wise mean embeddings)

        Returns:
            epoch_loss (dict): dictionary containing total loss and each indivial components
            acc_val (float): accuracy of the last batch of the last epoch 
            agg_protos_local (dict): local prototypes
        """
        
        # Set mode to train model
        self.net.train()

        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.config.settings['hyperParameters']['optimizerConfig']['type'] == 'SGD':
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config.settings['hyperParameters']['optimizerConfig']['learningRate'],
                                        momentum=self.config.settings['hyperParameters']['optimizerConfig']['momentum'])
        elif self.config.settings['hyperParameters']['optimizerConfig']['type'] == 'Adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.settings['hyperParameters']['optimizerConfig']['learningRate'],
                                         weight_decay=self.config.settings['hyperParameters']['optimizerConfig']['weightDecay'])
        
        criterion = __criterions__[self.config.settings['modelConfiguration']['loss']]().to(self.device)

        apply_log_softmax = False
        if self.config.settings['modelConfiguration']['loss'] == 'nllloss':
            apply_log_softmax = True

        proximal_loss_fn = __criterions__[self.config.settings['modelConfiguration']['regularizer']]().to(self.device)

        trainloader = self.dataset.get_train_loader()

        for iter in range(self.config.settings['learningParameters']['epochsPerRound']):

            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}

            for batch_idx, (_, images, labels) in enumerate(trainloader):

                images, labels = images.to(self.device), labels.to(self.device)

                # loss1: negative log-likelihood loss, loss2: proto distance loss
                self.net.zero_grad()
                embeddings, outputs = self.net(images, apply_log_softmax)
                loss1 = criterion(outputs, labels)

                if len(agg_protos) == 0:
                    loss2 = 0*loss1
                else:
                    proto_new = copy.deepcopy(embeddings.data)
                    i = 0
                    for label in labels:
                        if label.item() in agg_protos.keys():
                            proto_new[i, :] = agg_protos[label.item()][0].data
                        i += 1
                    loss2 = proximal_loss_fn(proto_new, embeddings)

                loss = loss1 + loss2 * self.config.settings['hyperParameters']['proxLossWeight']
                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if labels[i].item() in agg_protos_label:
                        agg_protos_label[labels[i].item()].append(embeddings[i,:])
                    else:
                        agg_protos_label[labels[i].item()] = [embeddings[i,:]]

                # outputs = outputs[:, 0:self.dataset.num_classes]
                _, y_hat = outputs.max(1)
                acc_val = torch.eq(y_hat, labels.squeeze()).float().mean()

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

        agg_protos_local = self.agg_func(agg_protos_label)

        return epoch_loss, acc_val.item(), agg_protos_local
  
    def agg_func(self, protos: dict, verbose=False):
        """ Returns the average of the protos.

        Args:
            protos (dict): _description_

        Returns:
            dict: average of the protos
        """
        if verbose:
            self.logger.info('Local proto generation')

        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].detach()
                for i in proto_list:
                    proto += i.detach()
                protos[label] = [proto / len(proto_list)]
            else:
                protos[label] = [proto_list[0].detach()]

        return protos

    def test(self, agg_protos: list):

        # Set mode to train model
        self.net.eval()

        accuracy_local, loss_local, accuracy_proto, loss_proto = 0.0, 0.0, 0.0, 0.0

        loss, total, correct = 0.0, 0.0, 0.0

        proximal_loss_fn = __criterions__[self.config.settings['modelConfiguration']['regularizer']]().to(self.device)

        criterion = __criterions__[self.config.settings['modelConfiguration']['loss']]().to(self.device)

        apply_log_softmax = False
        if self.config.settings['modelConfiguration']['loss'] == 'nllloss':
            apply_log_softmax = True

        testloader = self.dataset.get_test_loader()

        # test (use local loss)
        for batch_idx, (_, images, labels) in enumerate(testloader):

            images, labels = images.to(self.device), labels.to(self.device)
            
            self.net.zero_grad()
            
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
        self.logger.info(f'| User: {self.id} | Global Test Acc w/o protos: Accuracy {round(accuracy_local, 3)} Loss {round(loss_local, 3)}')

        # test (use global proto)
        if agg_protos != []:
            
            loss, total, correct = 0.0, 0.0, 0.0

            for batch_idx, (_, images, labels) in enumerate(testloader):
            
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.net.zero_grad()
                
                embeddings, _ = self.net(images, apply_log_softmax)

                # compute the dist between protos and global_protos
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], self.dataset.num_classes)).to(self.device)  # initialize a distance matrix
                for i in range(images.shape[0]):
                    for j in range(self.dataset.num_classes):
                        if j in agg_protos.keys() and j in self.classes:
                            d = proximal_loss_fn(embeddings[i, :], agg_protos[j][0])
                            dist[i, j] = d

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                # compute loss
                proto_new = copy.deepcopy(embeddings.data)
                i = 0
                for label in labels:
                    if label.item() in agg_protos.keys():
                        proto_new[i, :] = agg_protos[label.item()][0].data
                    i += 1

                loss += proximal_loss_fn(proto_new, embeddings).item()

            accuracy_proto = correct / total
            loss_proto = loss / total
            self.logger.info(f'| User: {self.id} | Global Test Acc with protos: Accuracy {round(accuracy_proto, 3)} Loss {round(loss_proto, 3)}')

        return accuracy_local, loss_local, accuracy_proto, loss_proto 