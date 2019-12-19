import time
import pickle
import pandas as pd

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor

# main file to run the program

def main():
    # load vectorizer, doc to word matrix, and topic model
    with open('a_finalized_model.pkl', 'rb') as f:
        vectorizer, doc_to_word_matrix, lda_model, doc_to_topic_matrix = pickle.load(f)

    topic_model_creator = LDAProcessor(doc_to_word_matrix, 10, vectorizer, True, lda_model)



    
    # Create an excel file showing the top 15 words for each topic
    topic_model_creator.topic_to_word_matrix_n_words(15)

    '''Predict topics for a new piece of text'''
    path_neg_test = 'C:/Users/johnm/Documents/Tutoring/CLASSES/MASTERS_PROJECT/code/txt_sentoken/neg_test/*.txt'
    #path_pos_test = 'C:/Users/johnm/Documents/Tutoring/CLASSES/MASTERS_PROJECT/code/txt_sentoken/pos_test/*.txt'

    doc_neg = CorpusProcessor(path_neg_test)
    doc_neg_text = doc_neg.create_one_doc()

    topic_model_creator.show_topics_for_unseen(doc_neg_text)

    # returns Document to Topic matrix and creates a csv file of the matrix
    #####doc_topic_matrix = topic_model_creator.create_doc_to_topic_matrix()
   
    print(doc_to_topic_matrix)

    classifications = ['bad', 'good']
    n = 999
    list_of_classifications = [item for item in classifications for i in range(n)]
    print(list_of_classifications)

    doc_to_topic_matrix['Classification'] = list_of_classifications

    print(doc_to_topic_matrix)
    











    







main()


