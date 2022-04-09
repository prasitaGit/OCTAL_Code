import torch_geometric.loader
from train import *
#from mlp import *
#from MLPGNN import *
#from GNNMLP import * 
from MLPMLP import * 
import random
import torch
import time
from accValid import *
import pdb
import argparse
from torch_geometric.data import Data
from ogb.linkproppred import Evaluator

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
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='use partial valid set')
    parser.add_argument('--test_ratio', type=float, default=1.0, help='use partial test set')
    parser.add_argument('--metric', type=str, default='mrr', help='metric for evaluating performance',
                        choices=['auc', 'mrr', 'hit'])
    parser.add_argument('--seed', type=int, default=2022, help='seed to initialize all the random modules')
    parser.add_argument('--device', type=int, default=7, help='gpu id')
    parser.add_argument('--root_path', type=str, default='/homes/mukher39/scratch/', help='path to file')

    # model training
    parser.add_argument('--optim', type=str, default='adam', help='optimizer to use')
    parser.add_argument('--epoch', type=int, default=499, help='num of epoches')
    parser.add_argument('--percent', type=float, default=0.1, help='num of epoches')
    parser.add_argument('--eval_steps', type=int, default=20, help='number of steps to test')
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size (train)')
    parser.add_argument('--batch_num', type=int, default=2000, help='mini-batch size (test)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--l2', type=float, default=0., help='l2 regularization (weight decay)')
    parser.add_argument('--patience', type=int, default=5, help='early stopping steps')
    parser.add_argument('--repeat', type=int, default=5, help='number of training instances to repeat')


    args = parser.parse_args()
    #call dataset here
    #load dataset

    
    data_train_path = 'syntheticDatasetSmallTripartite.pt'
    data_test_path = 'syntheticDatasetRes19Tripartite.pt'
    data_list = torch.load(f'{args.root_path}{data_train_path}')
    data_testAnalysis = torch.load(f'{args.root_path}{data_test_path}')

    data_list = [data for data in data_list if data[0].edge_index.shape[1] < 100000]

    train_ratio = args.train_ratio
    num_train = int(len(data_list)*train_ratio)
    print(f"Train and valid: {len(data_list)} \n Test: {len(data_testAnalysis)}")
    np.random.shuffle(data_list)
    np.random.shuffle(data_testAnalysis)
    data_train = data_list[:num_train]
    data_valid = data_list[num_train:]
    data_test = data_testAnalysis
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    np.random.seed(args.seed)

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
        
        for data in dataset:
            nodes = data[0].x.shape[0]
            corrNum = int(percent * nodes)
            if(corrNum > 0):
                nodeList = random.sample(range(0, nodes), corrNum)
                eList = dropEdgeList(data[0].edge_index,nodeList)
                datan = Data(x=data[0].x, edge_index=eList)
                dataset.append([datan, data[1],0])
                
            index += 1

        return dataset
            

    #data_train = dropNodes(data_train,args.percent)
    #print(data_train)
    #pdb.set_trace()
    #validSet = neg_gen(data_valid)
    #testSet = neg_gen(data_test)

    print(f"#valid set: {len(data_valid)}, #test set: {len(data_test)}")
    valid_loader = torch_geometric.loader.DataLoader(data_valid, batch_size=args.batch_size, shuffle = True)
    test_loader = torch_geometric.loader.DataLoader(data_test, batch_size=args.batch_size, shuffle = True)

    #define models
    modelClassifier = Classifier().to(device)
    #call train
    trainCode(modelClassifier, data_train, valid_loader, args, device)
    #1 is the tag for test
    start = time.time()
    evaluator = Evaluator(name = 'ogbl-citation2')
    acc = calcAccuracy(modelClassifier, test_loader, device)
    print("Time: ",time.time() - start)
