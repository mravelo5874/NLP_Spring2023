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

from tqdm import tqdm


NLTK_STOP_WORDS = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]
CUSTOM_STOP_WORDS = ["rrb", "lrb", "nt", "re", "ve", "ll", "mr", "ms", "th", ]
COMMON_SUFFIX_LIST = ["able", "acy", "al", "ance", "ation", "ate", "dom", "ed", "en", "ence", "er", "es", "est", "esque", "ful", "fy", "ial", "ible", "ic", "ical", "ify", "ing", "ion", "tion", "ious", "ise", "ish", "ism", "ist", "ity", "ive", "ize", "less", "ly", "ment", "ness", "or", "ous", "s", "ship", "sion", "tion", "ty", "y"]

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
        
        # skip if word is in stop words list
        if word in NLTK_STOP_WORDS:
            #print ('stop word: ', word)
            return None
        
        # remove all non-alphabetical characters from string
        clean_word = ''
        for character in word:
            if character in 'abcdefghijklmnopqrstuvwxyz':
                clean_word += character
        word = clean_word
        
        # skip if word is in custom stop words list
        if word in CUSTOM_STOP_WORDS:
            #print ('stop word: ', word)
            return None
        
        # remove any suffix found in the word
        # for suffix in COMMON_SUFFIX_LIST:
        #     if word.endswith(suffix):
        #         # only remove suffix iff suffix is less than 50% of the word's characters
        #         suffix_percent = len(suffix) / len(word)
        #         if (suffix_percent <= 0.5):
        #             word = word.removesuffix(suffix)
        
        # skip if word contains no characters
        if len(word) <= 1:
            return None
        
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
    def __init__(self, args, feat_extractor: FeatureExtractor):
        self.args = args
        self.args.num_epochs = 10 # override num epochs value
        self.args.lr = 0.1 # override lerning rate value
        self.featurizer = feat_extractor
        self.weights = None
        self.loss_list = []

    def fit(self, train_exs):
        # get feature counter (featire vector)
        counter = self.featurizer.get_counter()

        # get X most common words
        top_x_words = 1000
        indexer = self.featurizer.get_indexer()
        most_common = counter.most_common(top_x_words)
        print ('[Top ', top_x_words, ' Words]')
        i = 1
        for index, count in most_common:
            word = indexer.get_object(index)
            print (i, ': ', word, ' = ', count)
            i += 1

        # init weights
        print ('individual words: ', len(counter))
        self.weights = np.zeros(len(counter))
            
        # set random seed for data shuffling
        random.seed(100)
            
        # train over x epochs
        print ('[starting epochs]')
        print ('overridden epochs: ', self.args.num_epochs)
        print ('overridden learning rate: ', self.args.lr)
        for _ in range(self.args.num_epochs):
            # randomly shuffle training data
            random.shuffle(train_exs)
            # iterate through each example and update weights
            for count, example in enumerate(train_exs):
                # get data
                words = example.words
                label = example.label
                # extract features
                feature_counter = self.featurizer.extract_features(words, False)
                #print (count, '\tlabel: ', label, '\twords: ', words, '\tcounter: ', sorted(feature_counter.elements()))
                self.update_weights(label, list(feature_counter))
    
    # sigmoid function
    def logistic(self, value: float):
        exp_val = np.exp(value)
        return exp_val / (1 + exp_val)

    def update_weights(self, y: int, indexes: List[int]):
        # iterate through each individual index
        for index in indexes:
            w = self.weights[index]
            pred = self.logistic(w)
            # calculate loss and gradient
            loss = self.calculate_loss(y, pred)
            self.loss_list.append(loss)
            grad = self.calculate_gradient(y, pred)
            # update weight based on y value
            self.weights[index] = w - (self.args.lr * grad)
    
    # binary cross entropy loss function
    def calculate_loss(self, y, pred):
        return -np.mean(y*(np.log(pred))-(1-y)*np.log(1-pred))
     
    def calculate_gradient(self, y, pred):
        difference = pred - y
        return difference
    
    # predict method from SentimentClassifier superclass
    def predict(self, ex_words: List[str]) -> int:
        # extract features
        feature_counter = self.featurizer.extract_features(ex_words, False)
        
        # add weights
        sum = 0
        for index in list(feature_counter):
            sum += self.weights[index]

        # calculate probability
        prob = self.logistic(sum)
        
        #print ('predict: ', ex_words, '\tprob: ', prob)
            
        # determine sentiment based on calculation
        if prob >= 0.5:
            return 1
        else: 
            return 0
    

# [PART 1]
def train_logistic_regression(args, train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # create model
    model = LogisticRegressionClassifier(args, feat_extractor)
    
    # extract features
    for example in train_exs:
        model.featurizer.extract_features(sentence=example.words, add_to_indexer=True)

    # fit model using training data
    model.fit(train_exs)
    
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
    model = train_logistic_regression(args, train_exs, feat_extractor)
    return model


class SentimentIndexListExample:
    """
    Data wrapper for a single example for sentiment analysis for DAN.
    Attributes:
        index_list (List[int]): list of indexes for each word
        label (int): 0 or 1 (0 = negative, 1 = positive)
    """
    def __init__(self, index_list, label):
        self.index_list = index_list
        self.label = label


class DAN(nn.Module):
    
    def __init__(self, num_input, num_hidden, num_output, word_embeddings: WordEmbeddings):
        super(DAN, self).__init__()
        
        # create DAN layers
        self.embedding = word_embeddings.get_initialized_embedding_layer(True)
        self.linear_1 = nn.Linear(num_input, num_hidden)
        self.tanh_1 = nn.Tanh()
        self.linear_2 = nn.Linear(num_hidden, num_output)
        self.log_softmax = nn.LogSoftmax(dim=0)
        
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        
    def forward(self, x):   
        x = self.embedding(x)  
        x = torch.mean(input=x, dim=0)
        x = self.linear_1(x)
        x = self.tanh_1(x) 
        x = self.linear_2(x)  
        out = self.log_softmax(x)
        return out

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, args, model: DAN, word_embeddings: WordEmbeddings):
        self.args = args
        self.args.num_epochs = 5 # override num epochs value
        self.args.lr = 0.001 # override lerning rate value
        self.model = model
        self.word_embeddings = word_embeddings
    
    
    # converts a list of SentimentExample to SentimentIndexListExample
    def to_index_list_example(self, sentiment_exs: List[SentimentExample]) -> List[SentimentIndexListExample]:
        # list of SentimentIndexListExample to return
        sediment_index_list_examples = []
        # iterate through each sentiment example
        for example in sentiment_exs:
            # add to list
            sediment_index_list_examples.append(SentimentIndexListExample(self.to_index_list(example.words), example.label))
        return sediment_index_list_examples


    # takes a list of words and converts to list of indexes
    def to_index_list(self, words: List[str]) -> List[int]:
        # get indexer from word embeddings
        indexer = self.word_embeddings.get_indexer()
        # get index each word
        index_list = []
        for word in words:
            index = indexer.index_of(word)
            if index == -1:
                index = 1 # UNK token for unknown words
            index_list.append(index)
        return index_list
        
        
    def fit(self, train_exs: List[SentimentIndexListExample]):
        train_exs = self.to_index_list_example(train_exs)
        # set random seed for data shuffling
        random.seed(100)
        # optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        # train over x epochs
        print ('[starting epochs]')
        print ('overridden epochs: ', self.args.num_epochs)
        print ('overridden learning rate: ', self.args.lr)
        for epoch in range(self.args.num_epochs):
            # randomly shuffle training data
            random.shuffle(train_exs)
            # iterate through each training example
            print ('[epoch ', epoch + 1, ']')
            for example in tqdm(train_exs):
                # prepare x and y for training
                x = torch.IntTensor(example.index_list)
                y = example.label
                y_onehot = torch.zeros(2)
                y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
                # Zero out the gradients
                self.model.zero_grad()
                log_probs = self.model.forward(x)
                # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
                loss = torch.neg(log_probs).dot(y_onehot)
                # Computes the gradient and takes the optimizer step
                loss.backward()
                optimizer.step()
                
            # test each epoch
            
    
    def predict(self, ex_words: List[str]) -> int:
        index_list = self.to_index_list(ex_words)
        x = torch.IntTensor(index_list)
        log_probs = self.model.forward(x)
        pred = torch.argmax(log_probs)
        return pred
            
    
def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """

    # create DAN
    model = DAN(300, 500, 2, word_embeddings)
    
    # create classifier
    classifier = NeuralSentimentClassifier(args, model, word_embeddings)

    # train 
    classifier.fit(train_exs)
    
    return classifier