import pdb
import torch
import torch_geometric
from torch_geometric.data import Data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from accTest import *
#from accValid import *
import logging

def trainCode(modelClassifier, data_train, valid_loader, args, device, logger):#train_loader, valid_loader, test_loader, device):
    
    optimizer = torch.optim.Adam(modelClassifier.parameters(), lr=args.lr)#, weight_decay=1e-6)
    lossfunc = torch.nn.BCEWithLogitsLoss()
    loss_train = []
    trainAccList = []
    maxacc = 0
    maxValid = 0
    trainacc = 0
    countLess = 0 
    prevValidacc = 0
    checkBool = False
    lenOne = len(data_train)
    train_loader = torch_geometric.data.DataLoader(data_train, batch_size=args.batch_size, shuffle = True)
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
            np.random.shuffle(netList)
            print("Train size: ", len(netList))
            train_loader = torch_geometric.data.DataLoader(netList, batch_size=args.batch_size, shuffle = False)
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
            if torch.isnan(loss):
                print('Loss NaN')
            #if torch.isnan(outClassifier.grad).any():
                #print('grad NaN')
            optimizer.step()
            totalLoss += loss.item()*len(label)
            numlabels += len(label)

        #store the loss val
        lossVal = totalLoss/numlabels
        trainacc = calcacc/numlabels
        trainAccList.append(trainacc)
        if(maxacc < trainacc):
            maxacc = trainacc
        loss_train.append(lossVal)
        #early stopping
        #if(min(loss_train) < 0.0001):
            #break
            
        if((epoch+1) % args.eval_steps == 0):
            print("Validation set accuracy")
            validacc = calcAccuracy(modelClassifier, valid_loader, device,logger)
            if(validacc < prevValidacc):
                countLess += 1
            else:
                countLess = 0
            if(trainacc > 0.88 and countLess > 5):
                #accuracy not increasing anymore
                break
            prevValidacc = validacc

        print(f"Epoch: {epoch} Loss: {loss_train[-1]:.4f} Accuracy: {trainacc:.4f}")
        logger.info("Epoch: "+ str(epoch) + " Loss: "+str(loss_train[-1]) + " Accuracy: " + str(trainacc))
    print("Completed training...")

    #construct graph
    # plotting the points
    #plt.plot(np.arange(len(loss_train)), loss_train)
    # naming the x axis
    #plt.xlabel('Epochs')
    # naming the y axis
    #plt.ylabel('Loss')
    # giving a title to my graph
    #plt.title('Training loss curve')
    #save graph
    #plt.savefig('loss_curvennf.png')
