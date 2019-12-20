# Classifier Processor

import time
import pickle
import pandas as pd

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor

from sklearn.model_selection import train_test_split

# CLASSIFIERS
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report, confusion_matrix

class ClassifierProcessor():

    def __init__(self, lda_model, doc_to_topic_matrix):
        self.lda_model = lda_model
        self.doc_to_topic_matrix = doc_to_topic_matrix

    
