import torch
from torch import nn
from torch.nn import init
from random import random
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, global_mean_pool
    
class NetLTL(nn.Module):
    def __init__(self, dropout = 0.1, in_channels = 62, dim = 256, out_channels = 128):#,init_eps=np.random.uniform(1.5, 99.5),learn_eps=True):
        super(NetLTL, self).__init__()


        self.dropout = dropout

        self.conv1 = GCNConv(in_channels,dim)
        self.conv2 = GCNConv(dim,dim)
        self.conv3 = GCNConv(dim,dim)

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, data):

        x, edge_index, batch = data.x.float(), data.edge_index, data.batch.to(data.x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x


