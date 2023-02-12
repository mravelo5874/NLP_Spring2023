# models.py

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List


class LanguageModel(nn.Module):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, d_in:int, d_out: int, d_model: int, n_heads: int, d_hid: int, n_layers: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(d_in, d_model)
        self.pos_encode = PositionalEncoding(d_model, dropout, 5000)
        encode_layers = nn.TransformerEncoderLayer(d_model, n_heads, d_hid, dropout)
        self.trans_encoder = nn.TransformerEncoder(encode_layers, n_layers)
        self.linear = nn.Linear(d_model, d_out)
        
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

    def get_next_char_log_probs(self, context):
        print('context: ', context)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        x: Tensor, shape [seq_len, batch_size]
        mask: Tensor, shape [seq_len, seq_len]
        """
        print ('x.shape: ', x.shape)
        x = self.embed(x) * math.sqrt(self.d_model)
        x = self.pos_encode(x)
        out = self.trans_encoder(x, mask)
        out = self.linear(out)
        return out

def generate_square_mask(size: int) -> Tensor:
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(pos * div_term)
        pe[:, 0, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MY_DATASET(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

def set_up_dataloader(train: List[str], chunk_size: int, batch_size: int) -> DataLoader:
    # create input and output strings
    input_list = []
    output_list = []

    for i in range(len(train)):
        input_list.append(train[i])
        # create output string
        output = train[i]
        # remove first character add correct last character
        if i == (len(train) - 1):  output = output[1:] + " "
        else: output = output[1:] + train[i+1][chunk_size - 1]
        output_list.append(output)

    # create dataset and dataloader
    dataset = MY_DATASET(input_list, output_list)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    return dataloader

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """

    # split train_text into equally sized chunks, leaving a space as the first character of each chunk
    chunk_size = 20
    chunks = [train_text[i:i + (chunk_size - 1)] for i in range(0, len(train_text), (chunk_size - 1))]
    for i in range(len(chunks)): chunks[i] = ' ' + chunks[i]
    # pad last chunk to be of size 'chunk_size'
    last_chunk = chunks[len(chunks) - 1]
    chunks[len(chunks) - 1] = last_chunk.ljust(chunk_size, ' ')
    print ('chunks: ', len(chunks))
    print ('first chunk size: ', len(chunks[0]))
    print ('second to last chunk size: ', len(chunks[len(chunks) - 2]))
    print ('last chunk size: ', len(chunks[len(chunks) - 1]))

    # create dataset and dataloader for batching
    batch_size = 8
    train_dataloader = set_up_dataloader(chunks, chunk_size, batch_size)

    # create neural model
    d_in = chunk_size
    d_out = 27
    d_model = 512
    n_heads = 2
    d_hid = 2048
    n_layers = 1
    dropout = 0.1
    model = NeuralLanguageModel(d_in, d_out, d_model, n_heads, d_hid, n_layers, dropout)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    mask = generate_square_mask(chunk_size)

    # train model
    num_epochs = 2
    iter_list = []
    loss_list = []
    # keep track of total iterations
    iteration = 0
    print ('[starting training]')
    for epoch in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(epoch)
        # loss function
        loss_func = nn.NLLLoss()
        for x, y in tqdm(train_dataloader):
            # inc. iteration
            iteration += 1
            
            print ('x.shape: ', x.shape)
            print ('y.shape: ', y.shape)
            
            # forward pass
            model.zero_grad()
            p = model.forward(x, mask)
            #print ('p.shape: ', p.shape)
            #plot_maps(maps, c)
            #p = torch.argmax(p, dim=2)

            # fix pred and y tensors for NLLLoss
            p = torch.reshape(p, [batch_size * 20, 3])
            y = torch.reshape(y, [batch_size * 20])
            #print('y.shape: ', y.shape)
            #print('p.shape: ', p.shape)
            
            #print('y: ', y)
            #print('p: ', p)
            #loss = func.cross_entropy(p.view(-1, p.size(-1)))
            loss = loss_func(p, y)
            loss_list.append(loss.item())
            iter_list.append(iteration)
            
            # backward pass
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print ('total loss over epoch %i: %f' % (epoch + 1, loss_this_epoch))
    model.eval()
    
    plt.plot(iter_list, loss_list)
    plt.title('loss over training iterations')
    plt.ylabel('loss')
    plt.xlabel('training iterations')
    plt.show()

    return model
