import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support
import sys
import os
import glob
from pprint import pprint
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--val_tsv', type=str, help='cleannet_val.tsv | format: [sample_key, content_url, class_name, vlabel, feats]')
parser.add_argument('--val_logits_file', type=str, help='logits np file for cleannet_val.tsv')


def noise_filtering(logits, labels, vlabels):
    nb_cls = len(np.unique(labels))
    for k in range(nb_cls):    
        print('\n\n-------------starting for k={}-----------------'.format(k))
        rev_ind_sorted = np.argsort(-logits, axis=1)
        filtered_classes = rev_ind_sorted[:, :k]
        print('considered classes (by k filtering)', filtered_classes.shape)
        
        preds = np.zeros(len(vlabels))
        for i in range(len(vlabels)):
            binary = np.zeros((nb_cls))
            binary[filtered_classes[i]] = 1
            preds[i] = binary[labels[i]]  # if its in topK, then pred=1 (clean)
        
        report_metrics(vlabels, preds, labels)
    
    
def report_metrics(targets, preds, classes):
    C = np.max(classes) + 1
    per_class_accuracies = np.zeros((C,))
    for c in range(C):
        ind = np.where(classes == c)[0]
        class_acc = f1_score(targets[ind], preds[ind], average='micro')
        per_class_accuracies[c] = class_acc
    
    print('P/R/F1 (noise)', precision_recall_fscore_support(targets, preds, pos_label=0, average='binary'))
    print('f1_metrics (macro/unweighted mean)', f1_score(targets, preds, average='macro'))
    acc_classes = np.mean(per_class_accuracies)
    print('avg accuracy over classes', acc_classes, 'AvgErrorRate', 1 - acc_classes)
    
    extra_logging = False
    if extra_logging:
        print()
        acc = f1_score(targets, preds, average='micro')
        print('f1_metrics (accuracy/micro)', acc, 'ErrorRate', 1 - acc)
        print('P/R/F1 (clean)', precision_recall_fscore_support(targets, preds, pos_label=1, average='binary'))
        print('f1_metrics (weighted mean of f1)', f1_score(targets, preds, average='weighted'))
        # cr = classification_report(targets, preds)
        
        
def read_data(args):
    
    print('read validation tsv for evaluation (cleannet_val.tsv format: [sample_key, content_url, class_name, vlabel, feats])')
    val_df = pd.read_csv(args.val_tsv, sep='\t', header=None)
    val_df.columns = ['sample_key', 'url', 'class_name', 'vlabel', 'feats']
    vlabels = val_df.vlabel
    
    print('read logits computed for cleannet_val.tsv')
    logits = np.loadtxt(args.val_logits_file)
    
    print('process data')
    class_names = np.unique(val_df.class_name)
    class_map = dict(zip(class_names, np.arange(len(class_names))))
    print(class_map)
    labels = val_df['class_name'].apply(lambda x: class_map[x])
    
    return logits, labels, vlabels
    
    
if __name__=='__main__':
    print('starting program..')
    args = parser.parse_args()
    print(args)
    logits, labels, vlabels = read_data(args)
    noise_filtering(logits, labels, vlabels)
    print('finished program..')




    


