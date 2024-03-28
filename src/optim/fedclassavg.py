#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 05-Apr-2023
#
# ---------------------------------------------------------------------------
# Implementation of FedClassAvg
# ---------------------------------------------------------------------------

import pickle
import wandb
import logging
import torch
import copy
import random
import networkx as nx
import numpy as np

from pathlib import Path
from comm import Centralized, Decentralized
from models import BaseNet, SimCLRLoss, get_model, load_model, save_model
from datasets import BaseDataset, get_dataset
from utils import Config

__fedproto_models__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'cnnmnist', 'cnnfemnist']
__fedclassavg_models__ = ['resnet18', 'shufflenet', 'googlenet', 'alexnet']
__fedcma_models__ = ['cnn1', 'cnn2', 'mlp1', 'mlp2']
__criterions__ = {
    'nllloss': torch.nn.NLLLoss,
    'crossentropy': torch.nn.CrossEntropyLoss,
    'mse': torch.nn.MSELoss,
    'supcon': SimCLRLoss,
    'con': SimCLRLoss
}

class FedClassAvg():
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

        if 'sup' in self.config.settings['modelConfiguration']['sslloss']:
            self.logger.info('Using supervised contrastive loss')
        else:
            self.logger.info('Using unsupervised contrastive loss')

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

            dataset = get_dataset(self.config.settings['modelConfiguration']['dataset'])(self.config, client_id, isSSL=True)
            
            client_classes[client_id] = dataset.classes

            for class_name in dataset.classes:
                if class_name not in all_classes:
                    all_classes.append(class_name)
            
            confidence[client_id] = dataset.num_samples

            model_name = models[client_id % len(models)]

            if isFedProtoModels:

                if 'ResNet' in model_name:
                    choice = random.uniform(0,1)

                    if choice > 0.5:
                        stride = [1,4]
                    else:
                        stride = [2,2]

                else:
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


            client_model = get_model(model_name)(model_args)

            client = FedClassAvgClients(client_id, self.config, client_model, dataset)

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
    
    def update_local_classifiers(self):

        # update only weight and bias
        for idx, client in enumerate(self.clients):

            if self.isDecentralized:
                agg_state_dict = {}

                for params in self.global_classifier[idx]:
                    if 'weight' in params or 'bias' in params:
                        agg_state_dict[params] = self.global_classifier[idx][params]
            else:
                agg_state_dict = {}

                for params in self.global_classifier:
                    if 'weight' in params or 'bias' in params:
                        agg_state_dict[params] = self.global_classifier[params]

            client_dict = client.net.classifier.state_dict()
            client_dict.update(agg_state_dict)
            client.net.classifier.load_state_dict(client_dict)
            client.net.classifier.to(self.device)
    
    def exec_rounds(self):
        """Execute training rounds
        """

        self.logger.info('Starting execution rounds')

        clients = np.arange(self.num_clients)

        num_rounds = self.config.settings['learningParameters']['rounds']

        train_loss, train_accuracy = [], []

        for round in range(num_rounds):

            self.logger.info(f'| Global Training Round : {round + 1} |')

            local_accuracy, local_losses = [], []

            # update local models
            if round > 0:
                self.update_local_classifiers()

            for idx, cur_client in enumerate(clients):

                loss, acc = self.clients[cur_client].train(round)

                local_losses.append(copy.deepcopy(loss['total']))
                local_accuracy.append(copy.deepcopy(acc))

                self.logger.info('| User: {} | \tLoss: {:.3f} | Acc: {:.3f}'.format(idx + 1, loss['total'], acc))

                wandb.log({"acc_client_" + str(idx + 1): acc, "loss_client_" + str(idx + 1): loss['total']})

            # update global classifier
            if self.config.settings['networkConfiguration']['aggregation'] == 'gossip':
                raise NotImplementedError
            else:
                self.global_classifier = self.comm.classifier_aggregation_weighted(self.clients)

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
    
class FedClassAvgClients():
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

        if config.settings['resume'] or (not self.config.settings['train']):
            self.logger.info(f'| User: {self.id} | Loading from checkpoint')
            self.net = load_model(self.net, self.model_path.joinpath('client_' + str(self.id) + '.tar'))
        self.net.to(self.device)

    def init_weight(self, neighbours: torch.Tensor):
        
        self.neighbours = neighbours

        self.weights = (self.neighbours / torch.sum(self.neighbours)).to(self.device)
        self.logger.info(f'| User: {self.id} | Weights {self.weights}')

        # remove self loop
        self.neighbours[self.id]  = 0
        self.logger.info(f'| User: {self.id} | Neighbours {self.neighbours}')

        self.degree = torch.sum(self.weights).to(self.device) - self.weights[self.id]
        self.logger.info(f'| User: {self.id} | Degree {self.degree}')

    def set_confidence(self, confidence: torch.Tensor):
        self.confidence = confidence
        self.weights = confidence.to(self.device)

    def train(self, round: int, verbose=False):
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

        proximal_loss_fn = __criterions__[self.config.settings['modelConfiguration']['regularizer']]().to(self.device)

        initial_classifier = copy.deepcopy(self.net.classifier)

        trainloader = self.dataset.get_train_loader()

        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        for iter in range(self.config.settings['learningParameters']['epochsPerRound']):

            batch_loss = {'total':[],'1':[], '2':[], '3':[]}

            for batch_idx, (indexes, images, labels) in enumerate(trainloader):

                images = torch.cat([images[0], images[1]], dim=0)
                images, labels, indexes = images.to(self.device), labels.to(self.device), indexes.to(self.device)
                bsz = labels.shape[0]

                self.net.zero_grad()

                embeddings, outputs = self.net(images, apply_log_softmax)

                if 'sup' in self.config.settings['modelConfiguration']['sslloss']:
                    index = labels
                else:
                    index = indexes
                
                index = index.repeat(2) # considering 2 views

                loss1 = contrastive_loss_fn(embeddings, index) #supcontrast

                logits, _ = torch.split(outputs, [bsz, bsz], dim=0)

                loss2 = criterion(logits, labels)

                loss3 = 0*loss1
                
                if batch_idx != 0:
                    prox = torch.tensor(0.).to(self.device)
                    for name, param in self.net.classifier.named_parameters():
                        initial_weight = initial_classifier.state_dict()[name]
                        prox += proximal_loss_fn(param, initial_weight)
                    if prox != 0:
                        loss3 = (prox * self.config.settings['hyperParameters']['proxLossWeight'] / 2)
                
                loss = loss1 + loss2 + loss3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, y_hat = logits.max(1)
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
                batch_loss['3'].append(loss3.item())

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