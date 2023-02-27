# transformer.py

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_pos, d_model, num_heads, d_ff, num_classes, num_layers, dropout):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """        
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encode = PositionalEncoding(d_model, num_pos, True)
        self.trans_layers = nn.ModuleList([my_transformer_layer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, num_classes)
        self.softmax = nn.LogSoftmax(dim=2)
        
        nn.init.xavier_uniform_(self.linear.weight)
        
    def get_atten_maps(self):
        maps_list = []
        for layer in self.trans_layers:
            maps_list.append(layer.get_atten_maps())
        return torch.cat(maps_list)
        
    def forward(self, x, print_shapes = False) -> torch.Tensor:
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        
        if (print_shapes): print ('x.shape init: ', x.shape)
        x = self.embed(x)
        if (print_shapes): print ('x.shape embed: ', x.shape)
        # (1) adding positional encodings to the input (see the PositionalEncoding class; but we recommend leaving these out for now)
        x = self.pos_encode(x) # input -> [batch size, seq len, embedding dim]
        if (print_shapes): print ('x.shape pos encode: ', x.shape)
        # (2) using one or more of your TransformerLayers;
        for layer in self.trans_layers:
            x = layer(x)
        if (print_shapes): print ('x.shape trans: ', x.shape)
        # (3) using Linear and softmax layers to make the prediction
        x = self.linear(x)
        if (print_shapes): print ('x.shape linear: ', x.shape)
        x = self.softmax(x)
        if (print_shapes): print ('x.shape softmax: ', x.shape)
        x = torch.squeeze(x)
        return x, self.get_atten_maps()

# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
# used https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51 by Frank Odom to help my implementation :)
class my_transformer_layer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_k: The "k" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.atten = my_residual(
            my_multi_head_atten(num_heads, d_model),
            d_model,
            dropout,
        )
        self.ff = my_residual(
            my_ff(d_model, d_ff),
            d_model,
            dropout,
        )
        
    def get_atten_maps(self):
        return self.atten.sublayer.get_atten_maps()
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.atten(x, x, x, mask)
        x = self.ff(x)
        return x
        
class my_transformer_encoder_layer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.atten = my_residual(
            my_multi_head_atten(num_heads, d_model),
            d_model,
            dropout,
        )
        self.ff = my_residual(
            my_ff(d_model, d_ff),
            d_model,
            dropout,
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.atten(x, x, x, mask)
        x = self.ff(x)
        return x

# implements a residual connection used in the transformer layer
class my_residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dim: int, dropout: float):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        # this assumes that the "q" tensor is in index 0
        return self.norm(tensors[0] + self.drop(self.sublayer(*tensors)))
    
# implements a feed-forward module that is used by the residual layer in the transformer layer
def my_ff(d_model: int, d_ff: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Linear(d_ff, d_model)
    )

# implements multiple attention heads
class my_multi_head_atten(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        d_k = d_model // num_heads
        self.heads = nn.ModuleList([my_attention_head(d_model, d_k) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * d_k, d_model)
        self.atten = None
        
        nn.init.xavier_uniform_(self.linear.weight)
    
    # averages together maps from each head independent of batch size
    def get_atten_maps(self):
        maps_list = []
        for h in self.heads:
            head_map = h.get_atten_map()
            head_map = torch.unsqueeze(head_map, dim=0)
            maps_list.append(head_map)
        maps = torch.cat(maps_list)
        maps = torch.mean(maps, dim=0)
        return maps
    
    def forward(self, _q: torch.Tensor, _k: torch.Tensor, _v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([h(_q, _k, _v, mask) for h in self.heads], dim=-1))
    
# implements a single attention head
class my_attention_head(nn.Module):
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.q = nn.Linear(d_model, d_k)
        self.k = nn.Linear(d_model, d_k)
        self.v = nn.Linear(d_model, d_k)
        self.atten_maps = None
        
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
    
    def get_atten_map(self):
        return self.atten_map
        
    def forward(self, _q: torch.Tensor, _k: torch.Tensor, _v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        res, atten_map = scaled_dot_product_attention(self.q(_q), self.k(_k), self.v(_v), self.d_k, mask)
        self.atten_map = atten_map
        return res
    
# used for forward() of single attention head
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, d_k: int, mask: torch.Tensor):
    scores = q.bmm(k.transpose(1, 2)) / math.sqrt(d_k)
    #scores *= mask
    atten_map = func.softmax(scores, dim=-1)
    output = atten_map.bmm(v)
    return output, atten_map


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)

class MY_DATASET(Dataset):
    
    def __init__(self, features, labels, chars):
        self.features = features
        self.labels = labels
        self.chars = chars
        
    def get_chars(self):
        return self.chars
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.chars[idx]
        
def set_up_dataloader(train: List[LetterCountingExample], batch_size: int) -> DataLoader:
    input_list = []
    output_list = []
    chars_list = []
    # put example inputs and outputs into lists
    for ex in train:
        input_list.append(ex.input_tensor)
        output_list.append(ex.output_tensor)
        chars_list.append(ex.input)
    # create dataset and dataloader
    dataset = MY_DATASET(input_list, output_list, chars_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

# restructures y to be [batch_size, 20, 3] from [batch_size, 20]
def one_hot_y(_y: torch.Tensor, batch_size: int) -> torch.Tensor:
    y_oh = np.zeros(shape=[batch_size, 20, 3])
    for i in range(batch_size):
        y_oh[i][_y[i]] = 1
    y_oh = torch.FloatTensor(y_oh)
    return y_oh

def plot_maps(attn_maps, chars, plots_to_show=1):
    plots_shown = 0
    for j in range(0, len(attn_maps)):
        attn_map = attn_maps[j]
        fig, ax = plt.subplots()
        im = ax.imshow(attn_map, cmap='hot', interpolation='nearest')
        ax.set_xticks(np.arange(len(chars[j])), labels=chars[j])
        ax.set_yticks(np.arange(len(chars[j])), labels=chars[j])
        ax.xaxis.tick_top()
        plt.show()
        # stop showing plots
        plots_shown +=1
        if (plots_shown >= plots_to_show): return

# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    # set up dataloader for batching
    batch_size = 8
    train_dataloader = set_up_dataloader(train, batch_size)
    # create model
    model = Transformer(vocab_size=27, num_pos=20, d_model=512, num_heads=2, d_ff=2048, num_classes=3, num_layers=4, dropout=0.1)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # train model
    num_epochs = 5
    iter_list = []
    loss_list = []
    # keep track of total iterations
    iteration = 0
    
    print ('epochs: ', num_epochs)
    print ('batch_size: ', batch_size)
    print ('[starting training]')
    for epoch in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(epoch)
        # You can use batching if you'd like
        epoch_maps = []
        epoch_chars = []
        loss_func = nn.NLLLoss()
        for x, y, c in tqdm(train_dataloader): # tqdm removed for autograder
            # inc. iteration
            iteration += 1
            # restructure y to be [batch_size, 20, 3]
            #y = func.one_hot(y)
            
            # forward pass
            model.zero_grad()
            p, maps = model.forward(x)
            
            # convert string into char array
            chars_list = []
            for item in c:
                chars = []
                for j in item:
                    chars.append(j)
                chars_list.append(chars)
            chars_list = np.array(chars_list)
            #print ('c.shape: ', chars_list.shape)
            #print ('maps.shape: ', maps.shape)
            
            epoch_maps.append(maps)
            epoch_chars.append(chars_list)
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
        # rearrange maps and chars for vizualization
        np_epoch_maps = torch.cat(epoch_maps).detach().numpy()
        np_epoch_chars = np.array(epoch_chars)
        np_epoch_chars = np.reshape(np_epoch_chars, (np_epoch_chars.shape[0] * np_epoch_chars.shape[1], np_epoch_chars.shape[2]))
        print ('np_epoch_maps.shape', np_epoch_maps.shape)
        print ('np_epoch_chars.shape', np_epoch_chars.shape)
        #plot_maps(np_epoch_maps, np_epoch_chars, 16)
    model.eval()
    
    plt.plot(iter_list, loss_list)
    plt.title('loss over training iterations')
    plt.ylabel('loss')
    plt.xlabel('training iterations')
    plt.show()
        
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False, do_attention_normalization_test=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
        do_attention_normalization_test = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        if do_attention_normalization_test:
            normalizes = attention_normalization_test(attn_maps)
            print("%s normalization test on attention maps" % ("Passed" if normalizes else "Failed"))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))


def attention_normalization_test(attn_maps):
    """
    Tests that the attention maps sum to one over rows
    :param attn_maps: the list of attention maps
    :return:
    """
    for attn_map in attn_maps:
        total_prob_over_rows = torch.sum(attn_map, dim=1)
        if torch.any(total_prob_over_rows < 0.99).item() or torch.any(total_prob_over_rows > 1.01).item():
            print("Failed normalization test: probabilities not sum to 1.0 over rows")
            print("Total probability over rows:", total_prob_over_rows)
            return False
    return True