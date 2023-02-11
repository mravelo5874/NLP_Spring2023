# models.py

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

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
    def __init__(self, d_in:int, d_out: int, d_model: int, n_head: int, d_hid: int, n_layers: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.encoder = nn.Embedding(d_in, d_model)
        self.pos_encode = PositionalEncoding(d_model, dropout, 5000)
        encode_layers = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
        self.trans_encoder = nn.TransformerEncoder(encode_layers, n_layers)
        self.linear = nn.Linear(d_model, d_out)
        
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

    def get_next_char_log_probs(self, context):
        raise Exception("Implement me")

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        x: Tensor, shape [seq_len, batch_size]
        mask: Tensor, shape [seq_len, seq_len]
        """
        x = self.encoder(x) * math.sqrt(self.d_model)
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


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # TODO do this!!!
    model = NeuralLanguageModel()
