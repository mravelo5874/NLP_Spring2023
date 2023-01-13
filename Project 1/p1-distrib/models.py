# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter


NLTK_STOP_WORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]
    
    def update_positive(self, indexes: List[int]):
        raise Exception("Don't call me, call my subclasses")
    
    def update_negative(self, indexes: List[int]):
        raise Exception("Don't call me, call my subclasses")

class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    
    def get_indexer(self) -> Indexer:
        raise Exception("Don't call me, call my subclasses")
    
    def get_counter(self) -> Counter:
        raise Exception("Don't call me, call my subclasses")
    
    def clean_word(self, word: str) -> str:
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")

# [PART 1]
class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.my_indexer = indexer
        self.my_counter = Counter()
    
    # from FeatureExtractor superclass
    def get_indexer(self):
        return self.my_indexer
    
    # from FeatureExtractor superclass
    def get_counter(self):
        return self.my_counter
    
    # cleans word for featurization and training
    def clean_word(self, word):
        # convert to all lowercase characters
        word = word.lower()
        
        # remove all non-alphabetical characters from string
        clean_word = ''
        for character in word:
            if character in 'abcdefghijklmnopqrstuvwxyz':
                clean_word += character
        word = clean_word
        
        # skip if word contains no characters
        if len(word) <= 0:
            return None
        
        # skip if word is in stop words list
        if word in NLTK_STOP_WORDS:
            #print ('stop word: ', word)
            return None
        
        # TODO stemming
        # TODO lemmanting
        
        return word
    
    # from FeatureExtractor superclass
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        feature_counter = Counter()
        
        # for each word in a sentence
        for word in sentence:
            # clean word
            word = self.clean_word(word)
            
            # continue if word is None
            if word == None:
                continue
            
            # get index of word
            index = -1
            if add_to_indexer:
                index = self.my_indexer.add_and_get_index(word)
            else:
                index = self.my_indexer.index_of(word)
            #print ('word: ', word, ' index: ', index) 
            
            # only add to counter iff index not -1
            if index != -1:
                feature_counter[index] += 1
                self.my_counter[index] += 1
                
        # return counter
        return feature_counter
    

# [PART 1 exploration]
class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")
    
    # from FeatureExtractor superclass
    def get_indexer(self):
        return super().get_indexer()
    
    # from FeatureExtractor superclass
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        return super().extract_features(sentence, add_to_indexer)


# [PART 1 exploration]
class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")
    
    # from FeatureExtractor superclass
    def get_indexer(self):
        return super().get_indexer()
    
    # from FeatureExtractor superclass
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        return super().extract_features(sentence, add_to_indexer)
    

# [PART 1]
class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor: FeatureExtractor):
        self.featurizer = feat_extractor
        self.weights = None
        
    def sigmoid(self, value: float):
        exp_val = np.exp(value)
        return exp_val / (1 + exp_val)
        
    def update_positive(self, indexes: List[int]):
        for index in indexes:
            w = self.weights[index]
            val = 1 - self.sigmoid(w)
            self.weights[index] = w + val
    
    def update_negative(self, indexes: List[int]):
        for index in indexes:
            w = self.weights[index]
            val = self.sigmoid(w)
            self.weights[index] = w - val
    
    # predict method from SentimentClassifier superclass
    def predict(self, ex_words: List[str]) -> int:
        # extract features
        feature_counter = self.featurizer.extract_features(ex_words, False)
        
        # TODO prediction is being done incorrectly
        # add weights
        sum = 0
        for index in list(feature_counter):
            sum += self.weights[index]
            
        # determine sentiment based on sum
        if sum >= 0:
            return 1
        else: 
            return 0
    

# [PART 1]
def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # create model
    model = LogisticRegressionClassifier(feat_extractor)
    
    # extract features
    for example in train_exs:
        model.featurizer.extract_features(sentence=example.words, add_to_indexer=True)
    
    # get X most common words
    counter = model.featurizer.get_counter()
    # indexer = model.featurizer.get_indexer()
    # most_common = counter.most_common(100)
    # for index, count in most_common:
    #     word = indexer.get_object(index)
    #     print ('word: ', word, ' = ', count)
    
    # init weights
    print ('individual words: ', len(counter))
    model.weights = np.zeros(len(counter))
        
    # train
    for count, example in enumerate(train_exs):
        words = example.words
        label = example.label
        
        # extract features
        feature_counter = model.featurizer.extract_features(words, False)
        #print (count, '\tlabel: ', label, '\twords: ', words, '\tcounter: ', sorted(feature_counter.elements()))
        
        # logistic regression
        if label == 0:
            model.update_negative(list(feature_counter))
        elif label == 1:
            model.update_positive(list(feature_counter))
    
    # return trained classifier
    return model


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        raise NotImplementedError


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    raise NotImplementedError
