from typing import List
from multi_lingual_models import m2m100, mbart
import torch
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

class similar:
    def __init__(self, _model: str, _words: List[str], _lang: str):
        
        if _model == 'm2m': self.model = m2m100()
        elif _model == 'mbart': self.model = mbart()
        
        self.words = _words
        self.lang = _lang
        self.vectors = []
        self.sim_vectors = []
        
    def cosine_sim(self, A, B):
        return np.dot(A, B) / (norm(A)*norm(B))
        
    def generate_semantic_relation_matrix(self):
        # get embedding vector for each word
        for i in range(len(self.words)):
            res = self.model.embed(self.words[i], self.lang)
            # remove first dim
            res = torch.squeeze(res, dim=0)
            print ('word: ', self.words[i], ', embed shape: ', res.shape)
            # sum vectors together into one [1024] vector
            res = res.sum(dim=0)
            self.vectors.append(res.detach().numpy())
            
        # generate similarity vectors for each word
        for i in range(len(self.words)):
            sim_vec = []
            for j in range(len(self.words)):
                sim_vec.append(self.cosine_sim(self.vectors[i], self.vectors[j]))
            
            print ('word: ', self.words[i], ' sim_vec: ', sim_vec)
            self.sim_vectors.append(np.array(sim_vec))
        
        # create 2D matrix and mask 
        sim_matrix = np.reshape(a=np.array(self.sim_vectors), newshape=[len(self.words), len(self.words)])
        mask = np.zeros_like(sim_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        sim_matrix = np.where(mask, sim_matrix, 0)
        # create plot
        # Labels
        xlabs = self.words
        ylabs = self.words 
        # Heat map
        fig, ax = plt.subplots()
        im = ax.imshow(sim_matrix, cmap='gist_heat')
        # Add the labels
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('right')
        ax.set_xticks(np.arange(len(xlabs)), labels=xlabs)
        ax.set_yticks(np.arange(len(ylabs)), labels=ylabs)
        # add color bar
        cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])  # This is the position for the colorbar
        cbar = ax.figure.colorbar(im, cax=cbaxes)
        cbar.ax.set_ylabel('', rotation=-90, va='bottom')
        ax.yaxis.set_label_position('left')
        # Add the values to each cell
        for i in range(len(xlabs)):
            for j in range(len(ylabs)):
                if (sim_matrix[i, j] > 0.0):
                    text = ax.text(j, i, round(sim_matrix[i, j], 3), ha = "center", va = "center", color = "black")
        # rotate y-axis labels
        plt.setp(ax.get_xticklabels(), rotation=-40, ha='right', rotation_mode='anchor')
        plt.show()
                