import torch
import os
import pdb
import networkx as nx
import matplotlib.pyplot as plt
from netgraph import Graph
from torch import nn
from torch.nn import init
from random import random
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear
from torch_geometric.nn import GINConv, global_mean_pool
    
class NetBA(nn.Module):
    def __init__(self, dropout = 0.1, in_channels = 64, dim = 256, out_channels = 1):#,init_eps=np.random.uniform(1.5, 99.5),learn_eps=True):
        super(NetBA, self).__init__()


        self.dropout = dropout

        self.conv1 = GINConv(Sequential(Linear(in_channels,dim),BatchNorm1d(dim)))
        self.conv2 = GINConv(Sequential(Linear(dim,dim), BatchNorm1d(dim)))
        self.conv3 = GINConv(Sequential(Linear(dim,dim),BatchNorm1d(dim)))

        self.lin1 = Linear(dim, 128)
        self.lin2 = Linear(128, out_channels)

    def forward(self, data, node_num, edge_num, start_node, gid, checkStatus):

        x, edge_index, batch = data.x.float(), data.edge_index, data.batch.to(data.x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = x.sigmoid()
        if(checkStatus == True):
            self.drawGraph(x, edge_index, node_num, edge_num, start_node, gid, [1])
        x = global_mean_pool(x, batch)

        return x

    

    def saveGraph(self,vAttributes, edgeList, gid, start_node):
        G = nx.Graph()
        for ind in range(len(vAttributes)):
            color ='green'
            if(vAttributes[ind] == 0):
                color = 'red'
            G.add_node(ind,color=color)

        for ind in range(len(edgeList)):
            color = 'black'
            if ((edgeList[ind][0] >= start_node or edgeList[ind][1] >= start_node) and not(edgeList[ind][0] >= start_node and edgeList[ind][1] >= start_node)):
                color='blue'
            G.add_edge(edgeList[ind][0],edgeList[ind][1],color=color)
        #construct graph
        ecolors = nx.get_edge_attributes(G,'color').values()
        ncolors = nx.get_node_attributes(G,'color').values()
        if(len(G.nodes()) > len(ncolors)):
            pdb.set_trace()
        pos = nx.spring_layout(G,k=1.0)
        plt.figure(figsize=(15,15))
        nx.draw_networkx(G,pos,edge_color=ecolors,node_color=ncolors)
        fname = 'Graph'+str(len(os.listdir('/homes/yinht/lfs/Workspace/OCTAL/GNNLTL_NeurIPS_Code/GraphImages/'+str(gid))))+".png"
        plt.savefig('/homes/yinht/lfs/Workspace/OCTAL/GNNLTL_NeurIPS_Code/GraphImages/'+str(gid)+'/'+fname, format="PNG")
        plt.clf()
        exit()




    def drawGraph(self, x, edge_index, num_nodes, num_edges, start_nodes, gid, gLabels):
        
        gIDs = [gNum.item() for gNum in gid if gNum.item() in gLabels]
        if(len(gIDs) == 0):
            return

        count = 0
        countEdges = 0
        lenPre = 0
        
        for ind in range(len(num_nodes)):
            if(gid[ind].item() not in gIDs):
                count += num_nodes[ind].item()
                countEdges += num_edges[ind].item()

            else:
                vAttriButes = []
                eList = []
                nodes = num_nodes[ind].item()
                edges = num_edges[ind].item()
                for innd in range(nodes):
                    if(x[count + innd].item() <= 0.5):
                        vAttriButes.append(0)
                    else:
                        vAttriButes.append(1)
                #edges
                for innd in range(edges):
                    elem = countEdges + innd
                    eList.append([edge_index[0][elem].item() - count, edge_index[1][elem].item() - count])
                
                self.saveGraph(vAttriButes,eList,gid[ind].item(),start_nodes[ind].item())
                lenPre += 1
                if(lenPre == len(gIDs)):
                    break
                count += nodes
                countEdges += edges



