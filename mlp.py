import pdb
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout, LeakyReLU
from  GINModelBA import NetBA
from GINModelLTL import NetLTL
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
    def forward(self, dataBA):
        '''Forward pass'''
        outBA = self.layer_BA(dataBA)
        #outLTL = self.layer_LTL(dataLTL)
        #pdb.set_trace()
        x = outBA
         
        #x = torch.cat([outBA, outLTL], dim=-1)
        #x = x.float()
        #pdb.set_trace()
        x = self.layerLink(x)
        return x
