from typing import List
from facebook import m2m100
import torch
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

class similar:
    def __init__(self, _words: List[str], _lang: str):
        self.words = _words
        self.lang = _lang
        self.model = m2m100()
        self.vectors = []
        self.sim_vectors = []
        
    def cosine_sim(self, A, B):
        return np.dot(A, B) / (norm(A)*norm(B))
        
    def generate_semantic_relation_matrix(self):
        # get embedding vector for each word
        for i in range(len(self.words)):
            self.model.encode(self.words[i], self.lang)
            input_embed = self.model.get_model().get_input_embeddings()
            tokens = self.model.get_tokens()
            res = input_embed.forward(tokens)
            # remove first dim
            res = torch.squeeze(res, dim=0)
            # sum vectors together into one [1024] vector
            res = res.sum(dim=0)
            print ('word: ', self.words[i], ' res: ', res.shape)
            self.vectors.append(res.detach().numpy())
            
        # generate similarity vectors for each word
        for i in range(len(self.words)):
            sim_vec = []
            for j in range(len(self.words)):
                sim_vec.append(self.cosine_sim(self.vectors[i], self.vectors[j]))
            self.sim_vectors.append(np.array(sim_vec))
        
        # create 2D matrix and mask 
        sim_matrix = np.reshape(a=np.array(self.sim_vectors), newshape=[len(self.words), len(self.words)])
        mask = np.zeros_like(sim_matrix, dtype=np.bool)
        mask[np.tril_indices_from(mask)] = True
        sim_matrix = np.where(mask, sim_matrix, 0)
        # create plot
        # Labels
        xlabs = self.words
        ylabs = self.words 
        # Heat map
        _, ax = plt.subplots()
        ax.imshow(sim_matrix, cmap='gist_heat')
        # Add the labels
        ax.set_xticks(np.arange(len(xlabs)), labels = xlabs)
        ax.set_yticks(np.arange(len(ylabs)), labels = ylabs)
        plt.show()
                