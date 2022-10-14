import pdb
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch_geometric.nn import global_mean_pool
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.dropout = 0.1
        self.layers = Sequential(
               Linear(66,256),
               ReLU(), 
               Linear(256,64)
               )
        self.lin2 = Linear(64, 1)
    def forward(self, dataBA,num_nodes, num_edges, start_node, gid, checkStatus):
        '''Forward pass'''
        x,batch = dataBA.x.float(),dataBA.batch.to(dataBA.x.device)
        x = self.layers(x)
        x = F.relu(x)
        #mean pooling batch
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x
