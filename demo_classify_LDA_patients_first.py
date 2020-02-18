# demo_classify_LDA_patients_first.py
'''In this file we are loading objects that were saved in
    demo_LDA_patients_first.py

    We are going to use the doc_to_topic matrices to test 
    various classifiers.  We are interested whether the topic distribution
    of documents have some impact on the label/class/barrier.

    Latent Dirichlet Allocation (topic models) will be the only attributes
    for the tuples used to train and test the varous supervised learning
    classifiers used.

    Below are the objects that we are loading in the beginning of main
        1. vectorizer
            - a CountVectorizer
              https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

        2. all_lda_processors
            - This is a list that contains the following:
                0. lda_processor_5_topics
                1. lda_processor_10_topics
                2. lda_processor_15_topics
                3. lda_processor_20_topics
                4. lda_processor_25_topics
                5. lda_processor_30_topics

        3. all_lda_models
            - This is a list that contains the following:
                0. lda_model_5_topics
                1. lda_model_10_topics
                2. lda_model_15_topics
                3. lda_model_20_topics
                4. lda_model_25_topics
                5. lda_model_30_topics

        4. all_doc_to_topic_matrices
            - This is a list that contains the following: (with barriers)
                0. doc_to_5_topic_matrix
                1. doc_to_10_topic_matrix
                2. doc_to_15_topic_matrix
                3. doc_to_20_topic_matrix
                4. doc_to_25_topic_matrix
                5. doc_to_30_topic_matrix

        5. list_of_documents
            - list of strings such that each string are the comments
              from a patient navigator for the first visit for a given patient

        6. list_of_barriers
            - list of numpy.float64 that represents the barrier (social determinant of health)
              that was determined by the patient navigator after the visit/session.

        7. doc_to_word_matrix
            - document to work matrix
        Note:  indices for list_of_documents, list_of_barriers, and row number
                for doc_to_topic matrices are in line with each other.

        
'''
from pprint import pprint
import time
import pickle
import pandas as pd 

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor
from classifier_processor import ClassifierProcessor
from display_notes import DisplayNotes

def train_and_test_classifier(input_for_classifier, tuple_to_predict=[]):
    print("Training and testing multiple classifiers classifiers for", len(input_for_classifier.columns)-1,"topics")
    classifier = ClassifierProcessor(doc_to_topic_matrix= input_for_classifier, unseen_doc_features= tuple_to_predict)
    classifier.train_and_test_classifier_k_fold()


def show_top_n_words_for_each_topic(num_words, lda_processor_object, filename):
    topic_to_word_matrix = lda_processor_object.topic_to_word_matrix_n_words(num_words, filename)
    pprint(topic_to_word_matrix)

# takes in a dataframe and returns a dataframe with only the rows that != label for the Classfication column
def remove_rows_with_this_label(label, df):
    new_df = df[df.Classification != label]
    new_df.reset_index(drop=True, inplace=True) # reset index
    return new_df

def main():
    with open('china_LDA_patients_first_5_10_15_20_25_30.pkl', 'rb') as f:
        vectorizer, all_lda_processors, all_lda_models, all_doc_to_topic_matrices, list_of_documents, list_of_barriers, doc_to_word_matrix = pickle.load(f)

    # Show 5 topics and top 10 words
    show_top_n_words_for_each_topic(num_words= 10, lda_processor_object= all_lda_processors[0], filename= "topic_5_to_word_10_1st_visit.csv")
    
    # Show 10 topics and top 10 words
    show_top_n_words_for_each_topic(num_words= 10, lda_processor_object= all_lda_processors[1], filename= "topic_10_to_word_10_1st_visit.csv")

    # print("These are the 5 topics distribution for the first document: \n")
    # print(all_lda_models[0].transform(doc_to_word_matrix)[0])

    # print("These are the 5 topics distribution for the second document: \n")
    # print(all_lda_models[0].transform(doc_to_word_matrix)[1])

    # print("There are the 5 topics distribution for the third document: \n")
    # print(all_lda_models[0].transform(doc_to_word_matrix)[2])

    # print(all_doc_to_topic_matrices[0])

    # testing 5 topics
    # train_and_test_classifier(input_for_classifier= all_doc_to_topic_matrices[0])

    # # testing 10 topics
    # train_and_test_classifier(input_for_classifier= all_doc_to_topic_matrices[1])

    # # testing 15 topics
    # train_and_test_classifier(input_for_classifier= all_doc_to_topic_matrices[2])

    # # testing 20 topics
    # train_and_test_classifier(input_for_classifier= all_doc_to_topic_matrices[3])

    # # testing 25 topics
    # train_and_test_classifier(input_for_classifier= all_doc_to_topic_matrices[4])

    # # testing 30 topics
    # train_and_test_classifier(input_for_classifier= all_doc_to_topic_matrices[5])  

    '''
        In the following tests we are taking out label/classification
        "5.0" or "language/interpreter" from our data set.  
    '''
    pprint(all_doc_to_topic_matrices[0])
    my_df = remove_rows_with_this_label(label=5.0, df = all_doc_to_topic_matrices[0])
    
    pprint(my_df)
    # testing 5 topics without tuples with language/interpreter label
    # df = remove_rows_with_this_label(label=5.0, df = all_doc_to_topic_matrices[0])
    # train_and_test_classifier(input_for_classifier= df)

    # testing 10 topics without tuples with language/interpreter label
    # df = remove_rows_with_this_label(label=5.0, df = all_doc_to_topic_matrices[1])
    # train_and_test_classifier(input_for_classifier= df)

    # testing 15 topics without tuples with language/interpreter label
    # df = remove_rows_with_this_label(label=5.0, df = all_doc_to_topic_matrices[2])
    # train_and_test_classifier(input_for_classifier= df)

    # testing 20 topics without tuples with language/interpreter label
    # df = remove_rows_with_this_label(label=5.0, df = all_doc_to_topic_matrices[3])
    # train_and_test_classifier(input_for_classifier= df)

    # testing 25 topics without tuples with language/interpreter label
    # df = remove_rows_with_this_label(label=5.0, df = all_doc_to_topic_matrices[4])
    # train_and_test_classifier(input_for_classifier= df)

    # testing 30 topics without tuples with language/interpreter label
    df = remove_rows_with_this_label(label=5.0, df = all_doc_to_topic_matrices[5])
    train_and_test_classifier(input_for_classifier= df)


    # let's visuallize a document while highlighting most prevalent topics and their words

    topic_distribution_for_doc = all_lda_processors[3].show_topics_for_unseen([list_of_documents[2]]).reshape(1,-1)
    distribution = []
    distribution.append(topic_distribution_for_doc)
    document = []
    document.append(list_of_documents[2])
    words = 1000
    filename = 'patients_topic_to_word_matrix' + str(words) + '.csv'
    topic_to_word_matrix = all_lda_processors[3].topic_to_word_matrix_n_words(words, filename)
    display = DisplayNotes(document, distribution, topic_to_word_matrix)


    # display.display_threshold_topics_m_words(0.01, 100)
    display.display_top_n_topics_m_words(5, 1000)
main()