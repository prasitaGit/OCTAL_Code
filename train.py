import pdb
import torch
import torch_geometric.loader
from torch_geometric.data import Data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from accValid import *

def trainCode(modelClassifier, data_train, valid_loader, args, device):#train_loader, valid_loader, test_loader, device):
    
    optimizer = torch.optim.Adam(modelClassifier.parameters(), lr=args.lr)#,weight_decay=1e-4)
    lossfunc = torch.nn.BCEWithLogitsLoss()
    loss_train = []
    maxacc = 0
    maxValid = 0
    trainacc = 0
    countLess = 0 
    checkBool = False
    lenOne = len(data_train)
    train_loader = torch_geometric.loader.DataLoader(data_train, batch_size=args.batch_size, shuffle = True)
    modelClassifier.train()
    for epoch in range(args.epoch):
        totalLoss = 0
        numlabels = 0
        calcacc = 0
        #obtain zeros
        if(checkBool == True):
            arrIndex = np.arange(lenOne)
            dataList_zeros = []
            np.random.shuffle(arrIndex)
            for i in range(len(arrIndex)):
                if (arrIndex[i] != i):
                    ind = arrIndex[i]
                    BAZero = data_train[i][0]
                    LTLZero = data_train[ind][1]
                    dataList_zeros.append([BAZero, LTLZero, 0])
            netList = data_train + dataList_zeros

        #train_loader = torch_geometric.loader.DataLoader(netList, batch_size=64, shuffle = True)
        #call accuracy on validation
        for batchBA,batchLTL,label in train_loader:
            batchBA = batchBA.to(device)
            #batchLTL = batchLTL.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outClassifier = modelClassifier(batchBA).squeeze()
            pred_tag = torch.sigmoid(outClassifier) > 0.5
            calcacc += (pred_tag == label).sum().item()
            loss = lossfunc(outClassifier, label.float())
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()*len(label)
            numlabels += len(label)

        #store the loss val
        lossVal = totalLoss/numlabels
        trainacc = calcacc/numlabels
        if(maxacc < trainacc):
            maxacc = trainacc
        loss_train.append(lossVal)
        #early stopping
        if(min(loss_train) < 0.0001):
            break
            
        if((epoch+1) % args.eval_steps == 0):
            print("Validation set")
            validacc = calcAccuracy(modelClassifier, valid_loader, device)
            if(validacc < maxValid):
                countLess += 1
            else:
                maxValid = validacc
            if(trainacc > 0.88 and countLess > 5):
                #accuracy not increasing anymore
                break

        print(f"Epoch: {epoch} Loss: {loss_train[-1]:.4f} Accuracy: {trainacc:.4f}")
    print("Completed training...")

    #construct graph
    # plotting the points
    plt.plot(np.arange(len(loss_train)), loss_train)
    # naming the x axis
    plt.xlabel('Epochs')
    # naming the y axis
    plt.ylabel('Loss')
    # giving a title to my graph
    plt.title('Training loss curve')
    #save graph
    plt.savefig('loss_curvennf.png')
