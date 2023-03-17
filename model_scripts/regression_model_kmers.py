#!/usr/bin/env python

#  Bidirectional LSTM model

import os
import re
import sys
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import date
import tensorboard as tf
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from prettytable import PrettyTable
from tape import ProteinBertModel, TAPETokenizer
import gzip
import json
import ast

# conda activate pytorch_dev2


class LoadData:
    """
    Loads in kmerized sequences.
    """
    def __init__(self, jsonfilename, csv_file):
        self.jsonfilename = jsonfilename
        self.csv_file = csv_file

        def loading_json():
            """Read in json file, return data"""

            # labels = []
            # log10_ka = []
            seqs = []
            try:
                with open(self.jsonfilename, 'r') as fh:
                    s = fh.read()
                    # correct_format = re.sub(", *\n *}", "}", s)
                    # correct_format = re.sub("\"", "\"", correct_format)
                    # correct_format = re.sub("'", "\"", correct_format)
                    # data_json = json.loads(correct_format)
                    data_json = json.loads(s)
                    # for label in data_json.keys():
                    #     labels.append(label)

                    for seq in data_json.values():
                        seqs.append(seq)

            except FileNotFoundError:
                print(f'File not found error: {self.jsonfilename}.', file=sys.stderr)
                sys.exit(1)
            return seqs
            
        def loading_csv():
            labels = []
            log10_ka = []
            try:
                with open(csv_file, 'r') as fh:
                    next(fh) #skip header
                    for line in fh.readlines():
                        line = line.split(',')
                        labels.append(line[0])
                        log10_ka.append(np.float32(line[14]))
                        
            except FileNotFoundError:
                print(f'File not found error: {csv_file}.', file=sys.stderr)
                sys.exit(1)
            return labels, log10_ka

        self.labels, self.log10_ka = loading_csv()
        self.seqs = loading_json()
        self._longest_seq = len(max(self.seqs,key=len))


    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx):
        try:
            # features = self.transformer.embed(self._longest_seq,self.seqs[idx])
            features = self._longest_seq,self.seqs[idx]

            return self.labels[idx], features, self.log10_ka[idx]
            # return self.labels[idx], features, self.log10_ka[idx]

        except IndexError:
            print(f'List index out of range: {idx}, length: {len(self.labels)}.',
                  file=sys.stderr)
            sys.exit(1)
    



class BLSTM(nn.Module):
    """Bidirectional LSTM
    """
    def __init__(self,
                 batch_size,         # Batch size of the tensor
                 lstm_input_size,    # The number of expected features.
                 lstm_hidden_size,   # The number of features in hidden state h.
                 lstm_num_layers,    # Number of recurrent layers in LSTM.
                 lstm_bidirectional, # Bidrectional LSTM.
                 fcn_hidden_size,    # The number of features in hidden layer of CN.
                 device):            # Device ('cpu' or 'cuda')
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        # bidirection looks at past and future
        # LSTM layer
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bidirectional=lstm_bidirectional,
                            batch_first=True)               

        # FCN fcn layer; fully connected nn that does reg calc
        if lstm_bidirectional:
            self.fcn = nn.Linear(2 * lstm_hidden_size, fcn_hidden_size)
        else:
            self.fcn = nn.Linear(lstm_hidden_size, fcn_hidden_size)

        # FCN output layer
        self.out = nn.Linear(fcn_hidden_size, 1)
    def forward(self, x):
        # Initialize hidden and cell states to zeros.
        num_directions = 2 if self.lstm.bidirectional else 1
        h_0 = torch.zeros(num_directions * self.lstm.num_layers,
                          x.size(0),
                          self.lstm.hidden_size).to(self.device)
        c_0 = torch.zeros(num_directions * self.lstm.num_layers,
                          x.size(0),
                          self.lstm.hidden_size).to(self.device)

        # call lstm with input, hidden state, and internal state
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n.detach()
        c_n.detach() # detach from tensor, don't call gradient on it
        lstm_final_out = lstm_out[:,-1,:]  # last hidden state from every batch. size: N*H_cell
        lstm_final_state = lstm_final_out.to(self.device)
        fcn_out = self.fcn(lstm_final_state)
        prediction = self.out(fcn_out)
        return prediction


def run_lstm(model: BLSTM,
             train_set: Dataset,
             test_set: Dataset,
             n_epochs: int,
             batch_size: int,
             device: str,
             save_as: str):
    """Run LSTM model
    model: BLSTM,
    train_set: training set dataset
    test_set: test det dataset
    n_epochs: number of epochs
    batch_size: batch size
    device: 'gpu' or 'cpu'
    save_as: path and file name to save the model results
    """

    L_RATE = 1e-5              # learning rate 
    model = model.to(device)



    loss_fn = nn.MSELoss(reduction='sum').to(device)  # MSE loss with sum
    optimizer = torch.optim.SGD(model.parameters(), L_RATE)  # SGD optimizer

    train_loss_history = []
    test_loss_history = []
    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        test_loss = 0

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

        train_count = 0
        # for batch, (label, feature, target) in enumerate(train_loader):
        for label, feature, target in train_loader:
            train_count += 0
            optimizer.zero_grad()
            feature, target = feature.to(device), target.to(device)
            pred = model(feature).flatten()
            batch_loss = loss_fn(pred, target)        # MSE loss at batch level
            train_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()


        # for batch, (label, feature, target) in enumerate(test_loader):
        test_count = 0
        for label, feature, target in test_loader:
            test_count += 1
            feature, target = feature.to(device), target.to(device)
            with torch.no_grad():
                pred = model(feature).flatten()
                batch_loss = loss_fn(pred, target)
                test_loss += batch_loss.item()

        train_loss_history.append(train_loss) # list to write out to csv
        test_loss_history.append(test_loss)

        if epoch < 11:
            print(f'Epoch {epoch}, Train MSE: {train_loss}, Test MSE: {test_loss}')
        elif epoch%10 == 0:
            print(f'Epoch {epoch}, Train MSE: {train_loss}, Test MSE: {test_loss}')

        save_model(model, optimizer, epoch, save_as + '.model_save') # save model at diff states

    return train_loss_history, test_loss_history


def save_model(model: BLSTM, optimizer: torch.optim.SGD, epoch: int, save_as: str):
    """Save model parameters.
    model: a BLSTM model object
    optimizer: model optimizer
    epoch: number of epochs in the end of the model running
    save_as: file name for saveing the model.
    """
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               save_as)

def plot_history(train_losses: list, n_train: int, test_losses: list,
                 n_test: int, save_as: str):
    """Plot training and testing history per epoch
    train_losses: a list of per epoch error from the training set
    n_train: number of items in the training set
    test_losses: a list of per epoch error from the test set
    n_test: number of items in the test set
    """
    history_df = pd.DataFrame(list(zip(train_losses, test_losses)),
                              columns = ['training','testing'])

    history_df['training'] = history_df['training']/n_train  # average error per item
    history_df['testing'] = history_df['testing']/n_test

    print(history_df)

    sns.set_theme()
    sns.set_context('talk')
    sns.set_style("white")

    plt.ion()
    fig = plt.figure(figsize=(20, 15))
    ax = sns.scatterplot(data=history_df, x=history_df.index, y='training', label='Training', edgecolor='k', s=400)    
    sns.scatterplot(data=history_df, x=history_df.index, y='testing', label='Testing', edgecolor='k', s=400)
    ax.set(xlabel='Epochs', ylabel='Average MSE per sample')
    plt.title('Number of Epochs vs. Mean Squared Error', fontsize=55, pad=30) 
    plt.xlabel('Epochs', fontsize=45, labelpad=30)
    plt.ylabel('Average MSE per sample', fontsize=45, labelpad=30)
    plt.legend(fontsize=45, loc='upper right')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    fig.savefig(save_as + '.png')
    history_df.to_csv(save_as + '.csv')

def count_parameters(model):
    """Count model parameters and print a summary
    A nice hack from:
    https://stackoverflow.com/a/62508086/1992369
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}\n")
    return total_params

if __name__=='__main__':
    ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    ROOT_DIR = os.path.abspath(ROOT_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, 'rferina')

    KD_JSON = os.path.join(DATA_DIR, 'tapes_kmer_6_emb_alphaseq_4.json')
    # KD_CSV = os.path.join(DATA_DIR, 'alpha_seq_test.csv')
    KD_CSV = os.path.join(DATA_DIR, 'clean_avg_alpha_seq.csv')


    # Run setup
    # DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
    DEVICE = 'cpu'
    BATCH_SIZE = 75 # was 32
    N_EPOCHS = 700 # was 20

    ## TAPE LSTMb
    LSTM_INPUT_SIZE = 768       # lstm_input_size
    LSTM_HIDDEN_SIZE = 50       # lstm_hidden_size
    LSTM_NUM_LAYERS = 2         # lstm_num_layers
    LSTM_BIDIRECTIONAL = True   # lstm_bidrectional
    FCN_HIDDEN_SIZE = 100        # fcn_hidden_size
    
    # data_obj = LoadData(KD_CSV)
    # data_set = data_obj.loading_json()

    data_set = LoadData(KD_JSON, KD_CSV)

    print(len(data_set))

    TRAIN_SIZE = int(0.8 * len(data_set))  # 80% goes to training
    
    TEST_SIZE = len(data_set) - TRAIN_SIZE
   
    train_set, test_set = random_split(data_set, (TRAIN_SIZE, TEST_SIZE))
    # print(train_set)
    print(test_set)
    # print(test_set[10])

    model = BLSTM(BATCH_SIZE,
                  LSTM_INPUT_SIZE,
                  LSTM_HIDDEN_SIZE,
                  LSTM_NUM_LAYERS,
                  LSTM_BIDIRECTIONAL,
                  FCN_HIDDEN_SIZE,
                  DEVICE)

    count_parameters(model)
    model_result = f'blstm_1000_Layers_2_batch_75_LR_-5_TAPE_epochs_{N_EPOCHS}_train_{TRAIN_SIZE}_test_{TEST_SIZE}_{date.today()}'
    model_result = os.path.join(DATA_DIR, f'plots/{model_result}') 
    train_losses, test_losses = run_lstm(model, train_set, test_set,
                                         N_EPOCHS, BATCH_SIZE, DEVICE, model_result)
    plot_history(train_losses, TRAIN_SIZE, test_losses, TEST_SIZE, model_result)
