import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
from pprint import pprint
from tqdm import tqdm, trange
import pandas as pd
import io
import os
import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
import argparse
parser = argparse.ArgumentParser(description='Input Args')
parser.add_argument('--input_train', type=str, help='train tsv[text|label]')
parser.add_argument('--input_test', type=str, help='test tsv [text|label]')
parser.add_argument('--results_dir', type=str)
parser.add_argument('--model_file_name', type=str, help='save/saved model file name')
parser.add_argument('--emb_file_name', type=str, help='save emb file name')
parser.add_argument('--logits_file_name', type=str, help='logits file name')
parser.add_argument('--mode', type=str, help='train/ext/test: \
                    train(train+test+featext) ; ext/test:loadmodel+featext/test')

parser.add_argument('--nb_cls', type=int, default=2)
parser.add_argument('--max_seqlen', type=int, default=128)

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=4)

parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)


def flat_accuracy(preds, labels):
    # Function to calculate the accuracy of our predictions vs labels
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def report_metrics(targets, preds):
    pprint(classification_report(targets, preds))
    print('matthews corr coeff:', matthews_corrcoef(targets, preds))
    

def _reading_dataset(input_tsv):
    df = pd.read_csv(input_tsv, delimiter='\t',
                     header=0,
                     names=['id', 'text', 'gt_label', 'label', 'vlabel', 'test'])
                     # header=None, 
                     # names=['sentence_source', 'label', 'label_notes', 'text']) # .head()
                     # names=['text', 'label'])
                    
    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    sentences = df.text.values
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    labels = df.label.values  # df.gt_label.values  # df.label.values
    print(labels)
    print(np.unique(labels))
    print(type(labels))
    print("read data:", len(df))
    return sentences, labels
   
    
def _tokenize(sentences, tokenizer, max_seqlen):
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=max_seqlen, dtype="long", truncating="post", padding="post")
    # Create attention masks of 1s for each token followed by 0s for padding
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return input_ids, attention_masks
    
    
def _load_dataset(input_tsv, tokenizer, max_seqlen, batch_size, sampler_type):
    sentences, labels_values = _reading_dataset(input_tsv)
    input_ids, attention_masks = _tokenize(sentences, tokenizer, max_seqlen)
    inputs = torch.tensor(input_ids)
    labels = torch.tensor(labels_values)
    masks = torch.tensor(attention_masks)
    data = TensorDataset(inputs, masks, labels)
    data_sampler = sampler_type(data)
    feat_data_sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=data_sampler, batch_size=batch_size)
    feat_loader = DataLoader(data, sampler=feat_data_sampler, batch_size=batch_size)
    return dataloader, feat_loader, labels_values


def load_dataset(tokenizer, input_train, input_test, max_seqlen, batch_size, nb_cls):
    train_dataloader, feat_loader, train_labels = _load_dataset(
        input_train, tokenizer, max_seqlen, batch_size, RandomSampler)
    test_dataloader, _, _ = _load_dataset(input_test, tokenizer, max_seqlen, batch_size, SequentialSampler)
    # feat loader is for feature extraction (sequential sampler) with input_train
    # train set (noisy) and test set (clean)
    cls_weights = np.array(compute_class_weight('balanced', np.arange(nb_cls), train_labels), dtype=np.float32)
    return train_dataloader, test_dataloader, feat_loader, cls_weights


def train_model(epochs, model, train_dataloader, test_dataloader, cls_weights, results_dir, model_file_name):
    # Store our loss and accuracy for plotting
    train_loss_set = []
    print('computed train class_weights: {}'.format(cls_weights))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cls_weights).to(device))
    
    # trange is a tqdm wrapper around the normal python range
    for epoch_no in trange(epochs, desc="Epoch"):
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        # Training tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            ce_loss, logits, _ = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            weighted_ce_loss = criterion(logits, b_labels)
            loss = weighted_ce_loss  # ce_loss
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # break
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        evaluate_model(model, test_dataloader)
        torch.save(model.state_dict(), os.path.join(results_dir, model_file_name + "_{}".format(epoch_no)))
    return model
    

def evaluate_model(model, test_dataloader):
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    all_preds, all_labels = [], []
    all_logits = []
                     
    # Evaluate data for one epoch
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits, _ = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        all_preds.append(pred_flat)
        all_labels.append(labels_flat)
        all_logits.append(logits)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
        # break

    print("Test Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    
    preds = np.concatenate(all_preds, 0)
    labels = np.concatenate(all_labels, 0)
    ret_logits = np.concatenate(all_logits, 0)
    return preds, labels, ret_logits


def get_vectors(save_emb_path, model, feat_dl):
    
    # Feature extraction
    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    # predictions , true_labels = [], []
    # all_sentence_embeddings = []  # from [CLS] token (aggregate representation)
    # hidden state at layer_11 (12th/last layer): when fine-tuned on cls task, this rep makes sense.
    # after that there is a projection and tanh, followed by the classification Linear layer.
    # counter = 0
    f = open(save_emb_path, 'w')
    # Predict 
    for batch in feat_dl:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits, encoded_layers = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            encodings = encoded_layers[11][:, 0:1, :].squeeze()
        # Move logits and labels to CPU
        # logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()
        # Move sentence embeddings to CPU
        encodings = encodings.detach().cpu().numpy()
       
        # all_sentence_embeddings.append(encodings)
        np.savetxt(f, encodings, fmt='%.3f')
        # break
        # Store predictions and true labels
        # predictions.append(logits)
        # true_labels.append(label_ids)
        # counter += len(b_labels)
    
    # sentence_embeddings = np.concatenate(all_sentence_embeddings, 0)
    # print(sentence_embeddings.shape)
    f.close()
    # print(counter)
    

if __name__=='__main__':
    
    args = parser.parse_args()
    print(args)
    
    # files
    input_train = args.input_train
    input_test = args.input_test
    results_dir = args.results_dir
    model_file_name = args.model_file_name
    emb_file_name = args.emb_file_name
    logits_file_name = args.logits_file_name
    mode = args.mode
    
    # Hyperparameters
    nb_cls = args.nb_cls
    max_seqlen = args.max_seqlen
    batch_size = args.batch_size
    epochs = args.epochs  # Number of training epochs (authors recommend between 2 and 4) 
    lr = args.lr  # Number of training epochs (authors recommend 2e-5, 2e-4, 2e-3)
    weight_decay = args.weight_decay
    
    print('starting program..')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(torch.cuda.get_device_name(0))
    
    print('starting data loading..')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    train_dl, test_dl, feat_dl, cls_weights = load_dataset(
        tokenizer, input_train, input_test, max_seqlen, batch_size, nb_cls)
    
    print("loading pretrained model..")
    # Load BertForSequenceClassification, the pretrained BERT model with a linear cls layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=nb_cls, output_hidden_states=True)
    model = model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    # HERE: I haven't used learning rate scheduling (constant for me for all iterations)
    # HERE: I haven't used gradient clipping.
    #, correct_bias=False # To reproduce BertAdam specific behavior set correct_bias=False
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm=1.0)
    # loss.backward(), gradientclipping, optimizer.step(), schedulerstep, opt.zero_grad().
    # scheduler = get_linear_schedule_with_warmup(optimizer, 
    #                                             num_warmup_steps=100, 
    #                                             num_training_steps=1000)  # Pytorch scheduler

    if mode == 'train':
        print('starting model fine-tuning..')
        model = train_model(epochs, model, train_dl, test_dl, cls_weights, results_dir, model_file_name)
    else:
        print('loading saved fine-tuned model weights..')
        model.load_state_dict(torch.load(os.path.join(results_dir, model_file_name)))
        
    print('create results dir if not exists..')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if mode =='train':
        print('finished model evaluation and saving..')
        torch.save(model.state_dict(), os.path.join(results_dir, model_file_name))
        preds, labels, _ = evaluate_model(model, test_dl)
        report_metrics(labels, preds)
        print('finished saving representations..')
        get_vectors(os.path.join(results_dir, emb_file_name), model, feat_dl) 
    elif mode == 'test':
        print('finished model evaluation..')
        preds, labels, _ = evaluate_model(model, test_dl)
        report_metrics(labels, preds)
    elif mode == 'ext':
        print('finished saving representations..')
        get_vectors(os.path.join(results_dir, emb_file_name), model, feat_dl) 
    elif mode == 'class_filtering':
        print('finish saving logits for classification filtering baseline..')
        _, labels, logits = evaluate_model(model, feat_dl)
        np.savetxt(os.path.join(results_dir, logits_file_name), logits)   
    print('finished program.')
    