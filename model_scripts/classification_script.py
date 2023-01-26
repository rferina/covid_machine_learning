#!/usr/bin/env python

import os
import re
import sys
import torch
import numpy as np 
import pandas as pd
import seaborn as sns 
from datetime import date
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix


class Transformer:
    '''Transforms input sequence to features'''

    def __init__(self, method: str, **kwargs):

        if method == 'one_hot':
            self.method = 'one_hot'
            self.__LETTERS__ = "ACDEFGHIKLMNPQRSTVWY"
        
        else: 
            print('unimplemented transform method', file=sys.stderr)
            sys.exit(1)
    
    def embed(self, max_len, seq) -> torch.Tensor:
        '''Embed sequence feature using a defined method.
        seq: sequence of input.
        '''

        if self.method == 'one_hot':
            return self._embed_one_hot(max_len, seq)
        
        print(f'Undefined embedding method: {self.method}', file=sys.stderr)
        sys.exit(1)
    
    def _embed_tape(self, max_len: int, seq: str) -> torch.Tensor:
        '''Embed sequence feature using TAPE.
        TAPE mode gives two embeddings, one seq level embedding and one pooled_embedding.
        We will use the seq level emebdedding since the authors of TAPE suggest its performance
        is superior compared to the pooled embedding.
        '''
        _token = self.tape_tokenizer.encode(seq)
        try:
            pad_len = max_len - len(seq)
            padden_token = np.pad(_token, (0, pad_len), 'constant')
            token = torch.tensor(np.array([padded_token]))
        except ValueError:
            print(f'Incorrect length detected {len(_token)}')
        with torch.no_grad():
            seq_embedding, pooled_embedding = self.tape_model(token)

        return torch.squeeze(seq_embedding)

    def _embed_one_hot(self, max_len: int, seq: str) -> np.array:
        '''Embeded sequence feature using one-hot'''
        data = np.zeros((max_len, 20), dtype=int)   # returns a new array of shape(row:max_len, col:20), filled with zeros
        for idx, aa in enumerate(seq):
            assert aa in self.__LETTERS__
            aa_num = self.__LETTERS__.index(aa)
            data[idx, aa_num] = 1
        data = data.flatten().transpose()           # reshape data into a 1D tensor, and return tranpsosed version of data (dim0 and dim1 swapped)
        data = torch.from_numpy(data)               # convert numpy.ndarray to tensor first for one_hot
        return data

class AAData(Dataset):

    def __init__(self, csv_file: str, seq_type: str, transformer: Transformer):

        '''
        CSV: Covabdab dedpued and cleaned - labels are dependent on how you represent them numerically
        Load in labels for classification and sequences

        If seq_type is from the variable chain then VHorVHH & VL will need to be embedded and concatenated
        If seq_type is from the complimentary region CDRL3 & CDRH3 will need to be embedded and concatenated
        '''

        # Read in csv and set to instance class
        self.covabd_df = pd.read_csv('covabdab_split_binary.csv')
        # Labels will be 0, 1 for the binary dataset
        self.labels = self.covabd_df['Labels'].tolist()

        # Dealing with domain choices
        if seq_type == 'variable_chain':
            self.seq_1 = self.covabd_df['VHorVHH'].to_list()
            self.seq_2 = self.covabd_df['VL'].to_list()
        if seq_type == 'complimentary_region':
            self.seq_1 = self.covabd_df['CDRL3'].to_list()
            self.seq_2 = self.covabd_df['CDRH3'].to_list()

        # Find length of longest seq from first sequene class to pad
        self._longest_seq_1 = len(max(self.seq_1, key=len))
        # Find length of second seq class to pad
        self._longest_seq_2 = len(max(self.seq_2, key=len))
        # Set self transformer
        self.transformer = transformer 

    # Returns the number of samples in our dataset
    def __len__(self) -> int:
        return len(self.labels)

    # Loads and returns a sample from the dataset at the given index
    def __getitem__(self, idx):

        # Get label at idx
        labels = self.labels[idx]
        # Embed first seq at idx - this is calling embed method from the transformer class
        seq_1 = self.transformer.embed(self._longest_seq_1, self.seq_1[idx])
        # Embed second seq at idx - calling embed method from transformer class 
        seq_2 = self.transformer.embed(self._longest_seq_2, self.seq_2[idx])

        features = torch.cat((seq_1, seq_2))

        return labels, features
    

class clasifier(nn.Module):
    '''Classification model with an lstm layer'''

    # Define all layers used in the model 
    def __init__(self, vocab_size, embedding_dim, hidden_dim, out_dim, n_layers,
                bidirectional, dropout):
        
        super().__init__()
        # embedding layer
        self.embedding= nn.Embedding(vocab_size+1, embedding_dim)
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        # Dense layer
        self.fc = nn.Linear(hidden_dim*2, out_dim)
        # Activation function
        self.act = nn.Sigmoid()

    def forward(self, text): 

        embedded = self.embedding(text)
        packed_output, (hidden, cell) = self.lstm(embedded)
        # Concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        dense_outputs = self.fc(hidden)
        # Final activation function
        outputs = self.act(dense_outputs)

        return outputs 


### Functions

def binary_accuracy(preds, y):
    # Round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, dataloader, optiizer, criterion):

    # Initialize every epoch 
    epoch_loss = 0
    epoch_acc = 0

    # Set the model in training phase
    model.train()

    for idx, (label,text) in enumerate(dataloader):

        text = torch.tensor(text).long()
        label = torch.tensor(label)
        # Resets the gradients after every batch
        optimizer.zero_grad()
        # Convert to 1D tensor
        predictions = model(text).squeeze()
        # Compute the loss
        loss = criterion(predictions, label.float())
        # Backpropogate the loss and compute the gradients
        loss.backward()
        # Update the weights
        optimizer.step()
        acc = binary_accuracy(predictions, label)
        # Loss and accuracy 
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def evaluate(model, dataloader, criterion):

    # Initialize every epoch
    epoch_loss = 0 
    epoch_acc = 0 

    # Set the model in testing phase 
    model.eval()

    # Deactivates autograd
    with torch.no_grad():

        for idx, (label, text) in enumerate(dataloader):

            # convert to 1D tensor 
            predictions = model(text).squeeze()
            # Compute loss and accuracy
            loss = criterion(predictions, label.float())
            acc = binary_accuracy(predictions, label)
            # Keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

    save_model(model, optiizer, epoch, save_as + ',model_save')

def save_model(model: clasifier, optimizer: torch.optim.SGD, epoch: int, save_as: str):
    ''' Save model parameters.
        model: a classifier model object
        optimizier: model optimizer
        epoch: number of epochs in the end of the model running
        save_as: file name for saving the model.
        '''
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_state_dict()},
                save_as)

def plot_history(train_losses: list, n_train: int, test_losses: list,
                n_test: int, save_as: str):
    ''' Plot training and testing history per epoch
        train_losses: a list of per epoch error from the training set
        n_train: number of items in the training set
        test_losses: a list of per epoch error from the training set
        n_test: number of items in the test set
        '''
    history_df = pd.DataFrame(list(zip(train_loss_history, valid_loss_history)),
                            columns = ['training', 'testing']) 
    history_df['training'] = history_df['training']/n_train
    history_df['testing'] = history_df['testing']/n_train

    print(history_df)

    sns.set_theme()
    sns.set_context('talk')
    sns.set_style('white')

    plt.ion()
    fig = plt.figure(figsize=(20, 15))
    ax = sns.scatterplot(data=history_df, x=history_df.index, y='training', label='Training', edgecolor='k')
    sns.scatterplot(data=history_df, x=history_df.index, y='testing', label='Testing', edgecolor='k')
    ax.set(xlabel='Epochs', ylabel='Loss')
    plt.title('Number of Epochs vs. Loss', fontsize=55, pad=30)
    plt.xlabel('Epochs', fontsize=45, labelpad=30)
    plt.ylabel('Loss', fontsize=45, labelpad=30)
    plt.legend(fontsize=45, loc='upper right')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    fig.savefig(save_as + '.png')
    history_df.to_csv(save_as + '.csv')

def count_parameters(model):
    '''Count model parameters and print a summary'''

    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f'Total Trainable Params: {total_params}\n')
    return total_params

if __name__=='__main__':
    ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    ROOT_DIR = os.path.join(ROOT_DIR)
    # This will set ../ as the root directory, so DATA_DIR will be full path to where the data directory is
    DATA_DIR = os.path.join(ROOT_DIR, 'classification_model')

    COVDAB_CSV = os.path.join(DATA_DIR, 'covabdab_split_binary.csv')

    # Choose which seqs to concatanate
    sequence_domain = 'complimentary_region'
    # Choose one_hot_transformer or tape_transformer (call in data_set)
    one_hot_transformer = Transformer('one_hot')
    # tape_trasformer = Transformer('tape')
    # Use dataset class to fetch data
    data_set = AAData(COVDAB_CSV, sequence_domain, one_hot_transformer)
    print('dataset loader done')

    EPOCHS = 50
    LR = 0.01
    BATCH_SIZE = 32
    TRAIN_SIZE = int(0.8 * len(data_set))   # 80% train
    TEST_SIZE = len(data_set) - TRAIN_SIZE  # 20% test
    # Set training set and test set
    train_set, test_set = random_split(data_set, (TRAIN_SIZE, TEST_SIZE))
    # Load in train loader and test loader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
    print('data split done!')

    # Set number of classes 
    num_class = len(set([label for (label,text) in train_loader]))
    vocab_size = len([feature for (label, feature) in train_loader])
    embsize = 50
    num_hidden_nodes = 32
    num_output_nodes = 1
    num_layers = 2
    bidirection = True
    dropout = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = clasifier(vocab_size, embsize, num_hidden_nodes, num_output_nodes, num_class,
                    bidirectional=True, dropout=dropout)
    print('set classes done')

    # Set train size 
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None

    train_loss_history: list = []
    valid_loss_history: list = []
    for epoch in range(EPOCHS):
        # Train the model
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        # Evaluate the model
        valid_loss, valid_acc = evaluate(model, test_loader, criterion)
        # Save the model
        best_valid_loss = 1
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        
        print(f'Epoch {epoch}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tVal. loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%\n')
    
    count_parameters(model)
    model_result = f'classifier_onehot_epochs_{EPOCHS}_LR_{LR}_BS_{BATCH_SIZE}_train_{TRAIN_SIZE}_test_{TEST_SIZE}_{date.today()}'
    model_result = os.path.join(DATA_DIR, f'plots/{model_result}')
    plot_history(train_loss, TRAIN_SIZE, valid_loss, TEST_SIZE, model_result)
    print('model done!')
