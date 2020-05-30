from __future__ import print_function
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support
import sys
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser()
parser.add_argument('--input_tsv', type=str, help='Format: cleannet_all.tsv')
parser.add_argument('--save_file', type=str, help='save path')
parser.add_argument('--val_tsv', type=str, help='Format: cleannet_val.tsv')

HDIM = 32
LR = 0.001
BS = 64
NUM_EPOCHS = 10
MAX_ITERS = 10

class AE(nn.Module):
    
    def __init__(self, inp_dim, hid_dim):
        super(AE, self).__init__()
        self.h = nn.Linear(inp_dim, hid_dim)
        self.out = nn.Linear(hid_dim, inp_dim)
    
    def forward(self, x):
        x = self.h(x)
        x = self.out(x)
        return x
    
    
def train_AE(dataloader, model, loss_crit, device, opt):
    print('training AE on positives')
    for epoch in range(NUM_EPOCHS):  
        epoch_loss = 0.0
        num_batches = 0
        for X in dataloader:
            X = X.to(device)
            Xhat = model(X)
            loss = loss_crit(Xhat, X)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            num_batches += 1
        print("epoch = {}, loss = {:.3f}".format(epoch, epoch_loss/num_batches))
    print('done training AE on positives')
            
    
def get_reconstruction_errors(full_dl, model, reconstruction_criterion, device):
    all_recs = []
    with torch.no_grad():
        for X in full_dl:
            X = X.to(device)
            Xhat = model(X)
            recs = reconstruction_criterion(Xhat, X)
            recs_err = torch.sum(recs, dim=1).cpu().numpy().tolist()
            all_recs += recs_err
        all_recs = np.array(all_recs)
    return all_recs
    
    
def cluster_reconstruction_errors(reconstruction_errors):
    d = reconstruction_errors.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(d)
    p = kmeans.labels_
    c = kmeans.cluster_centers_
    c0 = kmeans.cluster_centers_[0][0]
    c1 = kmeans.cluster_centers_[1][0]
    ret_labels = p
    # means cluster 1 is for outlier (higher recons error) # end label should be 0
    if c1 > c0:
        ret_labels = 1-p
    return ret_labels
    

def is_assignment_updated(cluster_assignments, new_cluster_assignments):
    return not np.array_equal(cluster_assignments, new_cluster_assignments)
    
    
def get_dataloader(X, cluster_assignments=None):
    if cluster_assignments is not None:
        indices = np.where(cluster_assignments == 1)[0]  # clean = 1
        X = X[indices]
    dataloader = DataLoader(X, shuffle=True, batch_size=BS)
    return dataloader


def DRAE(X, sample_keys, val_keys, val_vlabels, val_classes):
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(torch.cuda.get_device_name(0))
    
    inp_dim = X.shape[1]
    model = AE(inp_dim, hid_dim=HDIM)
    model = model.to(device)
    
    reconstruction_criterion = nn.MSELoss(reduction='none')
    loss_crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    unshuffled_full_dataloader = DataLoader(X, shuffle=False, batch_size=BS)
    
    # start alternating between training autoencoder and clustering
    dataloader = get_dataloader(X)
    iterations = 0
    cluster_assignments = None
    while iterations < MAX_ITERS:
        print('iteration = {}'.format(iterations))
        train_AE(dataloader, model, loss_crit, device, opt)
        reconstruction_errors = get_reconstruction_errors(
            unshuffled_full_dataloader, model, reconstruction_criterion, device)
        new_cluster_assignments = cluster_reconstruction_errors(reconstruction_errors)
        if cluster_assignments is None or is_assignment_updated(cluster_assignments, new_cluster_assignments):
            dataloader = get_dataloader(X, new_cluster_assignments)
            cluster_assignments = new_cluster_assignments
        else: break
        iterations += 1
        val_preds = filter_preds(sample_keys, cluster_assignments, val_keys)
        report_metrics(val_preds, val_vlabels, val_classes)
    return cluster_assignments
    
    
def read_data(input_file):
    list_X = []
    list_sample_keys = []
    chunksize = 10 ** 3
    for i, chunk in enumerate(pd.read_csv(input_file, sep='\t', header=None, chunksize=chunksize)):
        print("\r >> Running... {}".format(i), end="")  # works across both major versions
        X = chunk[3].apply(lambda x: x.split(','))
        X = np.array(X.values.tolist(), dtype=np.float32)
        list_sample_keys += chunk[0].values.tolist()
        list_X.append(X)
#         if i % 7 == 6:
#             break
    ret_X = np.vstack(list_X)
    ret_keys = np.array(list_sample_keys, dtype=np.int32)
    print(ret_X.shape, ret_keys.shape)
    return ret_X, ret_keys
        
    # df = pd.read_csv(input_file, sep='\t', header=None, nrows=10)
    # print(df.head())
    # X = df[3].apply(lambda x: x.split(','))
    # X = np.array(X.values.tolist(), dtype=np.float32)
    # sample_keys = np.array(df[0].values, dtype=np.int32)
    # return X, sample_keys

def read_val_data(val_file):
    df = pd.read_csv(val_file, sep='\t', header=None)
    # print(df.head())
    val_keys = np.array(df[0].values, dtype=np.int32)
    val_vlabels = df[3].values
    val_classes = df[2].values
    # map classes (class labels used only in evaluations)
    class_names = np.unique(val_classes)
    d = dict(zip(class_names, np.arange(len(class_names))))
    val_classes = [d[c] for c in val_classes]
    return val_keys, val_vlabels, val_classes
    
def filter_preds(sample_keys, preds, val_keys):
    print(sample_keys.shape, preds.shape, val_keys.shape)
    sorted_preds = np.array([p for k, p in sorted(zip(sample_keys, preds))])
    return sorted_preds[val_keys]

def report_metrics(preds, targets, classes):
    C = len(np.unique(classes))
    per_class_accuracies = np.zeros((C,))
    for c in np.unique(classes):
        ind = np.where(classes == c)[0]
        class_acc = f1_score(targets[ind], preds[ind], average='micro')
        per_class_accuracies[c] = class_acc
    
    print('P/R/F1 (noise)', precision_recall_fscore_support(targets, preds, pos_label=0, average='binary'))
    print('f1_metrics (macro/unweighted mean)', f1_score(targets, preds, average='macro'))
    print('per class accuracy:', per_class_accuracies)
    acc_classes = np.mean(per_class_accuracies)
    print('avg accuracy over classes', acc_classes, 'AvgErrorRate', 1 - acc_classes)
    
    report_additional = False
    if report_additional:
        print()
        acc = f1_score(targets, preds, average='micro')
        print('f1_metrics (accuracy/micro)', acc, 'ErrorRate', 1 - acc)
        print('P/R/F1 (clean)', precision_recall_fscore_support(targets, preds, pos_label=1, average='binary'))
        print('f1_metrics (weighted mean of f1)', f1_score(targets, preds, average='weighted'))
    # cr = classification_report(targets, preds)

    
if __name__=='__main__':
    print('starting..')
    args = parser.parse_args()
    print(args)
    X, sample_keys = read_data(args.input_tsv)
    print('done reading train')
    val_keys, val_vlabels, val_classes = read_val_data(args.val_tsv)
    print('done reading val')
    preds = DRAE(X, sample_keys, val_keys, val_vlabels, val_classes)  # clean = 1, noisy/outlier = 0
    print('saving..')
    np.savetxt(args.save_file, preds)
    print('saved..')
    val_preds = filter_preds(sample_keys, preds, val_keys)
    report_metrics(val_preds, val_vlabels, val_classes)
    print('finished..')
    
    
    
    
    
    
    

