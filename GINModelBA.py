import torch
from torch import nn
from torch.nn import init
from random import random
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear
from torch_geometric.nn import GINConv, global_mean_pool
    
class NetBA(nn.Module):
    def __init__(self, dropout = 0.1, in_channels = 66, dim = 256, out_channels = 128):#,init_eps=np.random.uniform(1.5, 99.5),learn_eps=True):
        super(NetBA, self).__init__()


        self.dropout = dropout

        self.conv1 = GINConv(Sequential(Linear(in_channels,dim),BatchNorm1d(dim)))
        self.conv2 = GINConv(Sequential(Linear(dim,dim), BatchNorm1d(dim)))
        self.conv3 = GINConv(Sequential(Linear(dim,dim),BatchNorm1d(dim)))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, data, node_num, edge_num, start_node, gid, checkStatus):

        x, edge_index, batch = data.x.float(), data.edge_index, data.batch.to(data.x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.lin1(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x


