import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pdb
import torch.nn.functional as F

def calcAccuracy(modelClassifier, acc_loader, device, num_neg=1, evaluator=None):
    modelClassifier.eval()
    preds = []
    labels = []
    for batchBA,batchLTL,label in acc_loader:
        batchBA=batchBA.to(device)
        pred = modelClassifier(batchBA)
        preds.append(pred)
        labels.append(label)    
    preds = torch.cat(preds).squeeze().sigmoid()
    labels = torch.cat(labels)
    
    acc = ((preds > 0.5) == labels.to(device)).sum().item() / len(labels)
    auc = roc_auc_score(labels, preds.cpu().detach().numpy())
    print(f"ACC: {acc:.4f} AUC: {auc:.4f}")

    if num_neg > 1:
        pdb.set_trace()
        num_pos = labels.sum().item()
        pred_pos, pred_neg = preds[:num_pos], preds[num_pos:]
        result_dict = evaluator.eval({"y_pred_pos": pred_pos, "y_pred_neg": pred_neg.view(num_pos, -1)})
        for key, value in result_dict.items():
            print(key, value.mean().item())
    return acc
