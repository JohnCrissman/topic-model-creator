import time

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor

# main file to run the program

def main():
    start_time = time.time()

    path_neg = 'C:/Users/johnm/Documents/Tutoring/CLASSES/MASTERS_PROJECT/code/txt_sentoken/neg/*.txt'
    path_pos = 'C:/Users/johnm/Documents/Tutoring/CLASSES/MASTERS_PROJECT/code/txt_sentoken/pos/*.txt'

    corpus = CorpusProcessor(path_neg, path_pos)
    list_of_docs = corpus
    list_of_list_of_words = corpus.create_list_of_list_of_words()
    doc_to_word_matrix = corpus.create_doc_to_word_matrix(list_of_list_of_words)

    # We are choosing 20 topics as second parameter
    topic_model_creator = LDAProcessor(doc_to_word_matrix, 20, corpus.get_vectorizer())

    # Create an excel file showing the top 9 words for each topic
    topic_model_creator.topic_to_word_matrix_n_words(9)

    '''Predict topics for a new piece of text'''
    path_neg_test = 'C:/Users/johnm/Documents/Tutoring/CLASSES/MASTERS_PROJECT/code/txt_sentoken/neg_test/*.txt'
    #path_pos_test = 'C:/Users/johnm/Documents/Tutoring/CLASSES/MASTERS_PROJECT/code/txt_sentoken/pos_test/*.txt'

    new_doc_neg = CorpusProcessor(path_neg_test)
    new_doc_neg_text = new_doc_neg.create_one_doc()

    topic_model_creator.show_topics_for_unseen(new_doc_neg_text)

    # save the model with a getter
    # load the model here as well!

    











    print("My program took", time.time() - start_time, "to run")







main()