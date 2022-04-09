from torch import nn
from torch.nn import Sequential, Linear, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
import pdb

class MLPLTL(nn.Module):
    def __init__(self, dropout=0.1, in_channels=62, dim=256,
                 out_channels=128):  # ,init_eps=np.random.uniform(1.5, 99.5),learn_eps=True):
        super(MLPLTL, self).__init__()

        self.layers = Sequential(
            Linear(in_channels, dim),
            ReLU(),
            Linear(dim, dim)
        )

        self.dropout = dropout
        self.lin1 = Linear(dim,dim)
        self.lin2 = Linear(dim, out_channels)


    def forward(self, data):
        x,batch = data.x.float(),data.batch
        x = self.layers(x)
        x = global_max_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x
