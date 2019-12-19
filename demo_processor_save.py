import time
import pickle

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor

# main file to run the program

def main():
    start_time = time.time()

    path_neg = 'C:/Users/johnm/Documents/Tutoring/CLASSES/MASTERS_PROJECT/code/txt_sentoken/neg/*.txt'
    path_pos = 'C:/Users/johnm/Documents/Tutoring/CLASSES/MASTERS_PROJECT/code/txt_sentoken/pos/*.txt'

    corpus = CorpusProcessor(path_neg, path_pos)  
    list_of_list_of_words = corpus.create_list_of_list_of_words()

    doc_to_word_matrix = corpus.create_doc_to_word_matrix(list_of_list_of_words)
    vectorizer = corpus.get_vectorizer()

    # We are choosing 20 topics as second parameter
    topic_model_creator = LDAProcessor(doc_to_word_matrix, 20, vectorizer, False)

    lda_model = topic_model_creator.get_lda_model()

    # returns Document to Topic matrix and creates a csv file of the matrix
    doc_to_topic_matrix = topic_model_creator.create_doc_to_topic_matrix()

    # save vectorizer, doc to word matrix, and topic model
    with open('a_finalized_model.pkl', 'wb') as fout:
        pickle.dump((vectorizer, doc_to_word_matrix, lda_model, doc_to_topic_matrix), fout)

    
    











    print("My program took", time.time() - start_time, "to run")







main()


