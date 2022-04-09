from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout
import pdb
from torch_geometric.nn import global_add_pool

class MLPBA(nn.Module):
    def __init__(self, dropout = 0.1, in_channelsNode = 64, in_channelsEdge = 53, dim = 256, out_channels = 128):#,init_eps=np.random.uniform(1.5, 99.5),learn_eps=True):
        super(MLPBA, self).__init__()

        self.dropout = dropout
       
        self.layersNode = Sequential(
                Linear(in_channelsNode, dim),
                ReLU(),
                Linear(dim, dim)
                )

        self.lin2 = Linear(dim, out_channels)
        self.lin1 = Linear(dim, dim)



    def forward(self, data):
        xn,batch = data.x.float(),data.batch
        x = self.layersNode(xn)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x


