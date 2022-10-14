import pdb
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout, LeakyReLU
from GCNModel import NetBA
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.layer_BA = NetBA()
        #self.layer_LTL = NetBA(in_channels=62)
        self.dropout = 0.1
        self.layers = Sequential(
                #nn.Flatten(),
                Linear(256, 64),
                ReLU(),
                #Dropout(p=0.5),
                Linear(64, 1)
                )
        self.layerLink = Sequential(
               Linear(128,64),
               ReLU(), 
               Linear(64,1)
               )
    def forward(self, dataBA,num_nodes, num_edges, start_node, gid, checkStatus):
        '''Forward pass'''
        outBA = self.layer_BA(dataBA)
        x = outBA
         
        x = self.layerLink(x)
        return x
