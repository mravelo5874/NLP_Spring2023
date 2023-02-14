# models.py

import time
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
from utils import *

import transformer


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
    
    
'''
Custom NeuralLanguageModel that uses my custom transformer layer instead of pytorch's nn.TransformerEncoder module
'''
class MyNeuralLanguageModel(LanguageModel):
    def __init__(self, model_name: str, vocab_size: int, d_out: int, d_model: int, n_heads: int, d_hid: int, n_layers: int, dropout: float, indexer: Indexer):
        super().__init__()
        self.model_name = model_name
        self.indexer = indexer
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encode = PositionalEncoding(d_model, dropout, 5000)
        self.trans_encoder = nn.ModuleList([transformer.my_transformer_layer(d_model, n_heads, d_hid, dropout) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, d_out)
        self.logsoft = nn.LogSoftmax(dim=2)
        
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

    def get_next_char_log_probs(self, context):
        # make sure context is not empty
        if (len(context) <= 0):
            context = ' '
        # set model to evaluation mode
        self.eval()
        # convert string to indicies and to tensor
        indicies = index_data(context, self.indexer)
        seq = torch.IntTensor(indicies)
        # add extra dimension for 'batch size' of 1
        seq = torch.unsqueeze(seq, dim=0)
        # send through forward() to get predictions
        mask = generate_square_mask(len(context))
        pred = self.forward(seq, mask)
        # remove batch dimension
        pred = torch.squeeze(pred, dim=0)
        # get last tensor row and convert to numpy array
        last_index = pred.shape[0]-1
        pred = pred[last_index].detach().numpy()
        return pred

    def forward(self, x: Tensor, mask: Tensor, print_shapes = False) -> Tensor:
        """
        x: Tensor, shape [batch_size, seq_len]
        mask: Tensor, shape [seq_len, seq_len]
        """
        if print_shapes: print ('init x.shape: ', x.shape)
        x = self.embed(x) * math.sqrt(self.d_model)
        if print_shapes: print ('embed x.shape: ', x.shape)
        x = self.pos_encode(x)
        if print_shapes: print ('encode x.shape: ', x.shape)
        # x = torch.permute(x, (1, 0, 2))
        # if print_shapes: print ('permute x.shape: ', x.shape)
        for layer in self.trans_encoder:
            x = layer(x, mask)
        if print_shapes: print ('trans x.shape: ', x.shape)
        x = self.linear(x)
        if print_shapes: print ('linear x.shape: ', x.shape)
        out = self.logsoft(x)
        if print_shapes: print ('logsoft x.shape: ', x.shape)
        # out = torch.permute(x, (1, 0, 2))
        # if print_shapes: print ('permute x.shape: ', out.shape)
        return out

'''
NeuralLanguageModel using pytorch's nn.TransformerEncoder module
'''
class NeuralLanguageModel(LanguageModel):
    def __init__(self, model_name: str, vocab_size: int, d_out: int, d_model: int, n_heads: int, d_hid: int, n_layers: int, dropout: float, indexer: Indexer):
        super().__init__()
        self.model_name = model_name
        self.indexer = indexer
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encode = PositionalEncoding(d_model, dropout, 5000)
        encode_layers = nn.TransformerEncoderLayer(d_model, n_heads, d_hid, dropout)
        self.trans_encoder = nn.TransformerEncoder(encode_layers, n_layers)
        self.linear = nn.Linear(d_model, d_out)
        self.logsoft = nn.LogSoftmax(dim=2)
        
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

    def get_next_char_log_probs(self, context):
        # make sure context is not empty
        if (len(context) <= 0):
            context = ' '
        # set model to evaluation mode
        self.eval()
        # convert string to indicies and to tensor
        indicies = index_data(context, self.indexer)
        seq = torch.IntTensor(indicies)
        # add extra dimension for 'batch size' of 1
        seq = torch.unsqueeze(seq, dim=0)
        # send through forward() to get predictions
        mask = generate_square_mask(len(context))
        pred = self.forward(seq, mask)
        # remove batch dimension
        pred = torch.squeeze(pred, dim=0)
        # get last tensor row and convert to numpy array
        last_index = pred.shape[0]-1
        pred = pred[last_index].detach().numpy()
        return pred

    def forward(self, x: Tensor, mask: Tensor, print_shapes = False) -> Tensor:
        """
        x: Tensor, shape [batch_size, seq_len]
        mask: Tensor, shape [seq_len, seq_len]
        """
        if print_shapes: print ('init x.shape: ', x.shape)
        x = self.embed(x) * math.sqrt(self.d_model)
        if print_shapes: print ('embed x.shape: ', x.shape)
        x = self.pos_encode(x)
        if print_shapes: print ('encode x.shape: ', x.shape)
        x = torch.permute(x, (1, 0, 2))
        if print_shapes: print ('permute x.shape: ', x.shape)
        x = self.trans_encoder(x, mask)
        if print_shapes: print ('trans x.shape: ', x.shape)
        x = self.linear(x)
        if print_shapes: print ('linear x.shape: ', x.shape)
        x = self.logsoft(x)
        if print_shapes: print ('logsoft x.shape: ', x.shape)
        out = torch.permute(x, (1, 0, 2))
        if print_shapes: print ('permute x.shape: ', out.shape)
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

# convert a string to indecies
def index_data(string: str, indexer: Indexer) -> List[int]:
    indcies = []
    # for each char in string
    for i in range(len(string)):
        # convert to index
        indcies.append(indexer.index_of(string[i]))
    return indcies

def set_up_dataloader(data: List[str], chunk_size: int, batch_size: int, indexer: Indexer) -> DataLoader:
    # get number of chunks
    n_data = len(data)
    # create input and output strings
    input_list = []
    output_list = []

    for i in range(n_data):
        input_list.append(index_data(data[i], indexer))
        # create output string
        output = data[i]
        # remove first character add correct last character
        if i == (n_data - 1):
            output = output[1:]
            output += ' '
        else: 
            output = output[1:]
            output += data[i+1][1]
        
        output_list.append(index_data(output, indexer))
        # print (i, ' ', data[i], ' -> ', input_list[i])
        # print (i, ' ', output, ' -> ', output_list[i])

    # print ('input_list: ', len(input_list))
    # print ('output_list: ', len(output_list))
    input_list = np.array(input_list, dtype='int')
    output_list = np.array(output_list, dtype='int')

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
    chunk_size = 32
    chunks = [train_text[i:i + (chunk_size - 1)] for i in range(0, len(train_text), (chunk_size - 1))]
    for i in range(len(chunks)): chunks[i] = ' ' + chunks[i]
    # pad last chunk to be of size 'chunk_size'
    last_chunk = chunks[len(chunks) - 1]
    chunks[len(chunks) - 1] = last_chunk.ljust(chunk_size, ' ')
    print ('chunk size: ', chunk_size)
    print ('chunks: ', len(chunks))
    
    # split dev_text
    dev_chunks = [dev_text[i:i + (chunk_size - 1)] for i in range(0, len(dev_text), (chunk_size - 1))]
    for i in range(len(dev_chunks)): dev_chunks[i] = ' ' + dev_chunks[i]
    # pad last chunk to be of size 'chunk_size'
    last_dev_chunk = dev_chunks[len(dev_chunks) - 1]
    dev_chunks[len(dev_chunks) - 1] = last_dev_chunk.ljust(chunk_size, ' ')

    # create train dataset and dataloader for batching
    batch_size = 8
    train_dataloader = set_up_dataloader(chunks, chunk_size, batch_size, vocab_index)
    
    # create dev dataloader
    dev_dataloader = set_up_dataloader(dev_chunks, chunk_size, 1, vocab_index)

    # create neural model
    vocab_size = 27
    d_out = vocab_size
    d_model = 512
    n_heads = 4
    d_hid = 2048
    n_layers = 3
    dropout = 0.0
    #model = MyNeuralLanguageModel('custom transformer', vocab_size, d_out, d_model, n_heads, d_hid, n_layers, dropout, vocab_index)
    model = NeuralLanguageModel('pytorch\'s transformer', vocab_size, d_out, d_model, n_heads, d_hid, n_layers, dropout, vocab_index)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    mask = generate_square_mask(chunk_size)

    # train model
    num_epochs = 10
    iter_list = []
    loss_list = []
    # keep track of total iterations
    iteration = 0
    
    print ('epochs: ', num_epochs)
    print ('batch_size: ', batch_size)
    print ('[starting training]')
    start_time = time.time()
    for epoch in range(0, num_epochs):
        model.train()
        loss_this_epoch = 0.0
        random.seed(epoch)
        # loss function
        loss_func = nn.NLLLoss()
        for x, y in tqdm(train_dataloader):
            # inc. iteration
            iteration += 1

            # forward pass
            model.zero_grad()
            p = model.forward(x, mask)

            # fix p tensor
            p = torch.permute(p, (0, 2, 1))
        
            # compute loss
            loss = loss_func(p, y.long())
            loss_list.append(loss.item())
            iter_list.append(iteration)
            
            # backward pass
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print ('total loss over epoch %i: %f' % (epoch + 1, loss_this_epoch))
        model.eval()
        evaluate_model(model, dev_dataloader, chunk_size)
    
    print ("--- total train time: %.4s seconds ---" % (time.time() - start_time))
    print ("--- 10 mins = 600 seconds ---")
    print ("model name: ", model.model_name)
    
    # plot loss over time
    plt.plot(iter_list, loss_list)
    plt.title('loss over training iterations')
    plt.ylabel('loss')
    plt.xlabel('training iterations')
    plt.show()

    return model

def evaluate_model(model: NeuralLanguageModel, dev_dataloader: DataLoader, chunk_size):
    eval_total = 0
    eval_correct = 0
    mask = generate_square_mask(chunk_size)
    for x, y in dev_dataloader:
        p = model.forward(x, mask)
        p = torch.argmax(p, dim=2)
        # compare p and y
        p = p.squeeze(dim=0).numpy()
        y = y.squeeze(dim=0).numpy()
        local_total = len(y)
        local_correct = 0
        for i in range(local_total):
            if p[i] == y[i]:
                local_correct += 1
        eval_total += local_total
        eval_correct += local_correct
    #print ('eval_correct: %i, eval_total: %i' % (eval_correct, eval_total))
    print ('eval model accuraccy: %.3f' % (eval_correct/eval_total))