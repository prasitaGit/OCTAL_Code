from train import *
#from linkPred import *
#from mlpExp import *
from mlp import *
#from mlpGCN import *
from GNNMLP import *
import random
import torch
import time
import os
from accTest import *
import pdb
import argparse
import torch_geometric
from torch_geometric.data import Data
from ogb.linkproppred import Evaluator
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Interface for GRL_MCLTL framework')

    # general model and training setting
    parser.add_argument('--dataset', type=str, default='ogbl-citation2', help='dataset name',
                        choices=['ogbl-ppa', 'ogbl-ddi', 'ogbl-citation2', 'ogbl-collab', 'ogbn-mag'])
    parser.add_argument('--model', type=str, default='RNN', help='base model to use',
                        choices=['RNN', 'MLP', 'Transformer', 'GNN'])
    parser.add_argument('--layers', type=int, default=2, help='number of layers')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--x_dim', type=int, default=0, help='dim of raw node features')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='mask partial edges for training')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='use partial valid set')
    parser.add_argument('--test_ratio', type=float, default=1.0, help='use partial test set')
    parser.add_argument('--metric', type=str, default='mrr', help='metric for evaluating performance',
                        choices=['auc', 'mrr', 'hit'])
    parser.add_argument('--seed', type=int, default=2014, help='seed to initialize all the random modules')
    parser.add_argument('--device', type=int, default=6, help='gpu id')
    parser.add_argument('--root_path', type=str, default='/homes/yinht/lfs/Workspace/OCTAL/GNNLTL_NeurIPS_Code/', help='path to file')
    parser.add_argument('--data_train_path', type=str, default='ShortDirected.pt', help='path to file')
    parser.add_argument('--data_test_path', type=str, default='RERSDirected.pt', help='path to file')

    # model training
    parser.add_argument('--optim', type=str, default='adam', help='optimizer to use')
    parser.add_argument('--epoch', type=int, default=499, help='num of epoches')
    parser.add_argument('--percent', type=float, default=0.1, help='num of epoches')
    parser.add_argument('--eval_steps', type=int, default=20, help='number of steps to test')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size (train)')
    parser.add_argument('--batch_num', type=int, default=2000, help='mini-batch size (test)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--l2', type=float, default=0., help='l2 regularization (weight decay)')
    parser.add_argument('--patience', type=int, default=5, help='early stopping steps')
    parser.add_argument('--repeat', type=int, default=5, help='number of training instances to repeat')


    args = parser.parse_args()
    #call dataset here
    #load dataset
    data_list = torch.load(f'{args.root_path}{args.data_train_path}')
    data_testAnalysis = torch.load(f'{args.root_path}{args.data_test_path}')
    
    data_list = [data for data in data_list if data[0].edge_index.shape[1] < 100000]
    data_testAnalysis = [data for data in data_testAnalysis if data[0].edge_index.shape[1] < 100000]
    
    
    def generateDistribution(data_gen,saveFileName):
        fDist = open(saveFileName,'w')
        for dataBA,dataLTL,label,num_nodes,num_edges,start_node,gid in data_gen:
            sizeLTL = num_nodes - start_node
            #write
            fDist.write(str(sizeLTL)+","+str(num_nodes)+","+str(num_edges)+"\n")
        fDist.close()

    def set_random_seed(seedVal = 1):
        seed = seedVal
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


    def neg_gen(dataset, neg=1):
        num_data = len(dataset)
        if neg>1:
            for i in range(num_data):
                LTLZero = dataset[i][1]
                ind = np.random.randint(num_data, size=neg)
                while(i in ind):
                    ind = np.random.randint(num_data, size=neg)
                for j in ind:
                    dataset.append([dataset[j][0], LTLZero, 0])  
        else:
            for i in range(num_data):
                LTLZero = dataset[i][1]
                ind = np.random.randint(num_data)
                while(ind == i):
                    ind = np.random.randint(num_data)
                BAZero = dataset[ind][0]
                dataset.append([BAZero, LTLZero, 0])
                #idx = np.random.permutation(num_data)
                np.random.shuffle(dataset)
        return dataset

    
    def dropEdgeList(exList, nodelist):
        for node in nodelist:
            lenEdge = len(exList[0])
            #this gives us the indices for which the source vertex is 2 (hence corresponding dest vertex has to be deleted)       
            delDest = [index for index in range(lenEdge) if exList[0][index] == node]
            delSource = [index for index in range(lenEdge) if exList[1][index] == node]
            #select vertices that are not in delSource or delDest
            exList0 = [exList[0][index] for index in range(lenEdge) if index not in delSource]
            exList1 = [exList[1][index] for index in range(lenEdge) if index not in delDest]
            #filter out node
            exList0 = [data for data in exList[0] if data != node]
            exList1 = [data for data in exList[1] if data != node]
            exList = torch.tensor([exList0, exList1], dtype=torch.long)
         
        return exList
    
    def dropNodes(dataset, percent):
        index = 0 
        dataZeros = [] 
        for data in dataset:
            nodes = data[0].x.shape[0]
            corrNum = int(percent * nodes)
            print("Analyzing: ",index)
            if(corrNum > 0):
                nodeList = random.sample(range(0, nodes), corrNum)
                eList = dropEdgeList(data[0].edge_index,nodeList)
                datan = Data(x=data[0].x, edge_index=eList)
                dataZeros.append([datan, data[1],0])
                
            index += 1

        finalSet = dataset + dataZeros
        return finalSet
            
   
    for cexec in range(1):
 
        train_ratio = args.train_ratio
        valid_ratio = args.valid_ratio
        test_ratio = args.test_ratio
        num_train = int(len(data_list)*train_ratio)
        num_valid = int(len(data_list)*valid_ratio)
        num_test = int(len(data_testAnalysis)*test_ratio)
        #np.random.shuffle(data_list)
        np.random.shuffle(data_testAnalysis)
        data_train = data_list[:num_train]
        data_valid = data_list[num_train:num_train + num_valid]
        data_test = data_testAnalysis[-num_test:]
        for iexec in range(1):
            #generate random seed values
            seedVal = random.randint(100,10000)
            set_random_seed(seedVal)
            logFileName = "GINDirected"+str(iexec + 1)+".log"
            print("Iteration: ",cexec," logger name: ",logFileName)
            #create and configure logger - Level set to debug
            logging.basicConfig(filename=logFileName, format='%(asctime)s %(message)s', filemode='w', level=logging.DEBUG,      force=True)
            logger=logging.getLogger()

            device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
            logger.info("Device: "+str(device))
            logger.info("Seed: "+str(seedVal))
            #np.random.seed(args.seed)

            print(f"#train set: {len(data_train)}, #valid set: {len(data_valid)}, #test set: {len(data_test)}")
            logger.info("Training set size: "+str(len(data_train))+" Validation set size: "+str(len(data_valid))+" Test set size:    "+str(len(data_test)))
            valid_loader = torch_geometric.data.DataLoader(data_valid, batch_size=args.batch_size, shuffle = True)
            test_loader = torch_geometric.data.DataLoader(data_test, batch_size=args.batch_size, shuffle = True)

            #define models
            modelClassifier = Classifier().to(device)
            #call train
            #trainCode(modelClassifier, data_list, valid_loader, args, device, logger)
            trainCode(modelClassifier, data_train, valid_loader, args, device, logger)
            #1 is the tag for test
            evaluator = Evaluator(name = 'ogbl-citation2')
            acc = calcAccuracy(modelClassifier, test_loader, device, logger)
