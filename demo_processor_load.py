import time
import pickle
import pandas as pd

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor
from classifier_processor import ClassifierProcessor
from display_notes import DisplayNotes

from sklearn.model_selection import train_test_split

# CLASSIFIERS
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn import svm

from sklearn.metrics import classification_report, confusion_matrix

# main file to run the program

def main():
    # load vectorizer, doc to word matrix, and topic model
    with open('c_finalized_model.pkl', 'rb') as f:
        vectorizer, doc_to_word_matrix, lda_model, doc_to_topic_matrix = pickle.load(f)

    topic_model_creator = LDAProcessor(doc_to_word_matrix, 7, vectorizer, True, lda_model)

    # Create an excel file showing the top 15 words for each topic
    words = 100
    filename = 'movie_reviews_topic_to_word_matrix' + str(words) + '.csv'
    topic_to_word_matrix = topic_model_creator.topic_to_word_matrix_n_words(words, filename)
    

    '''Predict topics for a new piece of text'''
    path_neg_test = 'C:/Users/johnm/Documents/Tutoring/CLASSES/MASTERS_PROJECT/code/txt_sentoken/neg_test/*.txt'
    #path_pos_test = 'C:/Users/johnm/Documents/Tutoring/CLASSES/MASTERS_PROJECT/code/txt_sentoken/pos_test/*.txt'

    doc_neg = CorpusProcessor(path_neg_test)
    doc_neg_text = doc_neg.create_one_doc()
    
    # one_string = " ".join(doc_neg_text)
    

    ''' input used for predicting classification (good or bad) for unseen document!!!!!!!!!!!!'''
    unseen_doc_features = topic_model_creator.show_topics_for_unseen(doc_neg_text).reshape(1, -1)
    
    classifying_movie_reviews = ClassifierProcessor(doc_to_topic_matrix, unseen_doc_features)
    classifying_movie_reviews.train_classifier()
    classifying_movie_reviews.predict_class_for_doc()

    

    display = DisplayNotes(doc_neg_text, unseen_doc_features, topic_to_word_matrix)


    display.display_threshold_topics_m_words(0.05, 30)
    # display.display_top_n_topics_m_words(5, 100)


main()


