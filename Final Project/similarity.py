import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
import swadesh
from typing import List
from multi_lingual_models import m2m100, mbart

''' used to compute similarity between multiple languages in a NxN matrix '''
class multi_sim:
    def __init__(self, _model: str, _langs: List[str]):
        self.model = _model
        self.langs = _langs
        N = len(_langs)
        self.sim_matrix = np.zeros(shape=[N, N])
        self.norm_matrix = np.zeros(shape=[N, N])
        
    def compute_similarity_matrix(self, sim_func: str):
        prev_combos = []
        # compare each language with every other language
        for i in range(len(self.langs)):
            for j in range(len(self.langs)):
                # compute duo similarity if 
                # - both languages are not equal
                # - language pair has not already been calculated
                if self.langs[i] != self.langs[j] and [self.langs[i], self.langs[j]] not in prev_combos:
                    words0 = swadesh.get_swadesh_words(self.langs[i], 'swadesh-207')
                    words1 = swadesh.get_swadesh_words(self.langs[j], 'swadesh-207')
                    d_sim = duo_sim(self.model, self.langs[i], self.langs[j], words0, words1)
                    similarity = d_sim.compute_similarity(sim_func, True)
                    self.sim_matrix[i, j] = similarity
                    self.sim_matrix[j, i] = similarity
                    print ('similarity between \'%s\' and \'%s\' is %f.' % (self.langs[i], self.langs[j], similarity))
                    prev_combos.append([self.langs[i], self.langs[j]])
                    prev_combos.append([self.langs[j], self.langs[i]])
                # set value in matrix to 1 if same languages
                elif self.langs[i] == self.langs[j]:
                    self.sim_matrix[i, j] = 1
                    
        # normalize similarity matrix
        self.norm_matrix = mono_sim.normalize_results(self.sim_matrix)
        # reshape
        self.sim_matrix = np.reshape(a=self.sim_matrix, newshape=[len(self.langs), len(self.langs)])
        self.norm_matrix = np.reshape(a=self.norm_matrix, newshape=[len(self.langs), len(self.langs)])
        
        # mask bottom triangle
        mask = np.zeros_like(self.norm_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        masked_sim_matrix = np.where(mask, self.norm_matrix, 0)
             
        # create plot
        # Labels
        xlabs = self.langs
        ylabs = self.langs 
        # Heat map
        fig, ax = plt.subplots()
        im = ax.imshow(masked_sim_matrix, cmap='gist_heat')
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
                if (mask[i, j] == True):
                    text_color = ''
                    if self.norm_matrix[i, j] <= 0.3: text_color='white'
                    else: text_color='black'
                    text = ax.text(j, i, round(self.sim_matrix[i, j], 3), ha="center", va="center", color=text_color, size='x-small')
        # rotate y-axis labels
        plt.setp(ax.get_xticklabels(), rotation=-40, ha='right', rotation_mode='anchor')
        plt.rcParams.update({'font.size': 8})
        plt.show()

''' used to compute similarity between two languages '''
class duo_sim:
    def __init__(self, _model: str, _lang0: str, _lang1, _words0: List[str], _words1: List[str]):
        assert len(_words0) == len(_words1)
        self.lang0 = _lang0
        self.lang1 = _lang1
        self.model = _model
        self.words0 = _words0
        self.words1 = _words1
        self.similarity = -1
        
    # computes the similarity of the two languages
    def compute_similarity(self, sim_func: str, normalize: bool):
        # determine valid similarity function
        if (sim_func != 'spearman' and 
            sim_func != 'cosine-dist'):
            print ('[ERROR]: Invalid similarity function provided \'%s\'.' % sim_func)
            return -1
        
        # compute vector embeddingds and similarity vectors
        mono_0 = mono_sim(self.model, self.words0, None, self.lang0)
        mono_1 = mono_sim(self.model, self.words1, None, self.lang1)
        mono_0.create_vector_embeddings()
        mono_1.create_vector_embeddings()
        vecs_0 = mono_0.generate_similarity_vectors()
        vecs_1 = mono_1.generate_similarity_vectors()
        sim_0 = mono_sim.normalize_results(vecs_0)
        sim_1 = mono_sim.normalize_results(vecs_1)
        sim_0 = np.reshape(a=sim_0, newshape=[len(self.words0), len(self.words0)])
        sim_1 = np.reshape(a=sim_1, newshape=[len(self.words0), len(self.words0)])
        '''
        use spearman correlation OR cosine distance ?
        '''
        sim_sum = 0
        for i in range(len(self.words0)):
            word_sim = 0
            if sim_func == 'spearman':
                word_sim = utils.spearman(sim_0[i,:], sim_1[i,:])
            elif sim_func == 'cosine-dist':
                word_sim = 1 - utils.cosine_dist(sim_0[i,:], sim_1[i,:])
            #print ('word_sim: ', word_sim)
            sim_sum += word_sim
        self.similarity = sim_sum / len(self.words0)
        # normalize bewteen 0 and 1
        if normalize: self.similarity = utils.inverse_interpolation(-1.0, 1.0, self.similarity)
        return self.similarity

''' used to compute embedding and similarity vectors for one language '''
class mono_sim:
    def __init__(self, _model: str, _words: List[str], _eng_words: List[str], _lang: str):
        if _model == 'm2m': self.model = m2m100()
        elif _model == 'mbart':  self.model = mbart()
        self.lang = self.model.get_language_id(_lang)
        self.words = _words
        self.eng_words = _eng_words
        self.vectors = []
        self.sim_vectors = []
        
    def create_vector_embeddings(self):
        # get embedding vector for each word
        for i in range(len(self.words)):
            res = self.model.embed(self.words[i], self.lang)
            # remove first dim
            res = torch.squeeze(res, dim=0)
            #print ('word: ', self.words[i], ', embed shape: ', res.shape)
            # sum vectors together into one [1024] vector
            res = res.sum(dim=0)
            # normalize vector
            res = res.detach().numpy()
            res = res / np.linalg.norm(res)
            # add to vectors
            self.vectors.append(res)
        return self.vectors
      
    def generate_similarity_vectors(self):
        # generate similarity vectors for each word
        for i in range(len(self.words)):
            sim_vec = []
            for j in range(len(self.words)):
                sim_vec.append(utils.cosine_sim(self.vectors[i], self.vectors[j]))
            self.sim_vectors.append(np.array(sim_vec))
        self.sim_vectors = np.array(self.sim_vectors)
        return self.sim_vectors
            
    def normalize_results(sim_vectors):
        # normalize sim matrix using largest and smallest value
        sim_array = np.array(sim_vectors).flatten()
        min_val = 999.0
        max_val = -999.0
        for i in range(len(sim_array)):
            if sim_array[i] < min_val:
                min_val = sim_array[i]
            if sim_array[i] > max_val and sim_array[i] < 0.999:
                max_val = sim_array[i]
        #print ('min: ', min_val, ', max: ', max_val)
        for i in range(len(sim_array)):
            if sim_array[i] < 0.999:
                sim_array[i] = utils.inverse_interpolation(min_val, max_val, sim_array[i])
        return sim_array
                
    def semantic_relation_matrix(self):
        # words -> vector embeddings
        self.vectors = self.create_vector_embeddings()
        # vector embeddings -> similarity vectors
        self.sim_vectors = self.generate_similarity_vectors()
        # create 2D matrix and mask 
        sim_array = mono_sim.normalize_results(self.sim_vectors)
        sim_matrix = np.reshape(a=sim_array, newshape=[len(self.words), len(self.words)])
        mask = np.zeros_like(sim_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        sim_matrix = np.where(mask, sim_matrix, 0)
        
        # create plot
        # Labels
        labels = []
        for i in range(len(self.words)): labels.append(self.words[i] + '(' + self.eng_words[i] + ')')
        xlabs = labels
        ylabs = labels
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
        '''
        for i in range(len(xlabs)):
            for j in range(len(ylabs)):
                if (sim_matrix[i, j] > 0.0):
                    text = ax.text(j, i, round(sim_matrix[i, j], 3), ha = "center", va = "center", color = "black")
        '''
        # rotate y-axis labels
        plt.setp(ax.get_xticklabels(), rotation=-40, ha='right', rotation_mode='anchor')
        plt.show()
                