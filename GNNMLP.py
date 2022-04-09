import pdb
import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from MLPModelBA import MLPBA
from GCNModelLTL import NetLTL
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.layer_BA = MLPBA()
        self.layer_LTL = NetLTL()
        self.layers = Sequential(
                #nn.Flatten(),
                Linear(128, 64),
                ReLU(),
                Linear(64, 1)
                )

    def forward(self, dataBA, dataLTL):
        '''Forward pass'''
        outBA = self.layer_BA(dataBA)
        outLTL = self.layer_LTL(dataLTL)
        #pdb.set_trace()
        x = outBA * outLTL
        #x = torch.cat([outBA, outLTL], dim=-1)
        x = self.layers(x)
        return x
