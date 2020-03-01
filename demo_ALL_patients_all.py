# demo_ALL_patients_all.py

# composition of this demo file is similar to demo_ALL_patients_first.py
""" This demonstraction file will create LDA models for the following number of topics:
        5, 10, 15, 20, 25, 30

       We will use the document to topic matrices along with the barrier (label) other 
       visit information and patients demographics to classify 
        them in demo_classify_ALL_patients_all.py.  We will save all the models in this file and 
        load them in demo_classify_ALL_patients_all.py
        
       The comments from the patient navigator for a patients visit will be one document.
       We will be looking at all patient's visits that have comments and barriers(classifications)
       
        Each visit is one data point.


        Input to make LDA comes from this csv:
        df_each_visit_one_hot_encoding.csv
"""

from pprint import pprint
import pandas as pd
import csv
import math
import time
import pickle

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor

def main():

    # create all lda_models and use pickle to save
    # the models and other objects to load somewhere else.

    # creating dataframe from csv file
    df = pd.read_csv('df_each_visit_one_hot_encoding.csv')
    print(df) # [1486 rows x 42 columns]

    # creating new dataframe file with columns Classification
    # comments and record_id from df in the previous line.
    dfObj = pd.DataFrame(columns=['record_id', 'navigation_comments', 'barrier'])
    dfObj['record_id'], dfObj['navigation_comments'], dfObj['barrier'] = df['record_id'], df['comments'], df['Classification']
    print(dfObj) # [1486 rows x 3 columns]

    del df['comments']
    del df['Classification']
    del df['record_id']
    other_patient_visit_data = df
    
    print('Here is the dataframe!!!!!')
    print(other_patient_visit_data)
    # create list of strings s.t. each string is a comment
    # create list of strings s.t. each string is a barrier
    list_of_documents = []
    list_of_barriers = []
    list_of_record_ids = []
    for i in range(len(dfObj)):
        list_of_documents.append(dfObj.iloc[i,1])
        list_of_barriers.append(dfObj.iloc[i,2])
        list_of_record_ids.append(dfObj.iloc[i,0])
    
    print(len(list_of_documents))
    print(len(list_of_barriers))
    print(len(list_of_record_ids))

    # create a CorpusProcessor object to transform our lists into input for LDA
    corpus = CorpusProcessor()

    list_of_list_of_words = corpus.create_list_of_list_of_words(list_of_documents)
    doc_to_word_matrix = corpus.create_doc_to_word_matrix(list_of_list_of_words)

    vectorizer = corpus.get_vectorizer()

    # creating distinct LDA models for different number of topics (5, 10, 15, 20, 25, 30)
    lda_processor_5_topics = LDAProcessor(doc_to_word_matrix= doc_to_word_matrix, num_topics=5, vectorizer= vectorizer, exists= False)
    lda_processor_10_topics = LDAProcessor(doc_to_word_matrix= doc_to_word_matrix, num_topics=10, vectorizer= vectorizer, exists= False)
    lda_processor_15_topics = LDAProcessor(doc_to_word_matrix= doc_to_word_matrix, num_topics=15, vectorizer= vectorizer, exists= False)
    lda_processor_20_topics = LDAProcessor(doc_to_word_matrix= doc_to_word_matrix, num_topics=20, vectorizer= vectorizer, exists= False)
    lda_processor_25_topics = LDAProcessor(doc_to_word_matrix= doc_to_word_matrix, num_topics=25, vectorizer= vectorizer, exists= False)
    lda_processor_30_topics = LDAProcessor(doc_to_word_matrix= doc_to_word_matrix, num_topics=30, vectorizer= vectorizer, exists= False)

    all_lda_processors = [lda_processor_5_topics, lda_processor_10_topics, lda_processor_15_topics, lda_processor_20_topics,
                            lda_processor_25_topics, lda_processor_30_topics]

    # storing all the LDA models in variables in order to save
    lda_model_5_topics = lda_processor_5_topics.get_lda_model()
    lda_model_10_topics = lda_processor_10_topics.get_lda_model()
    lda_model_15_topics = lda_processor_15_topics.get_lda_model()
    lda_model_20_topics = lda_processor_20_topics.get_lda_model()
    lda_model_25_topics = lda_processor_25_topics.get_lda_model()
    lda_model_30_topics = lda_processor_30_topics.get_lda_model()

    all_lda_models = [lda_model_5_topics, lda_model_10_topics, lda_model_15_topics, lda_model_20_topics,
                        lda_model_25_topics, lda_model_30_topics]

    # creates a doc_to_topic_matrix and appends the barriers (labels) and the record_id to them
    doc_to_5_topic_matrix = lda_processor_5_topics.create_doc_to_topic_matrix(list_of_barriers, list_of_record_ids)
    doc_to_10_topic_matrix = lda_processor_10_topics.create_doc_to_topic_matrix(list_of_barriers, list_of_record_ids)
    doc_to_15_topic_matrix = lda_processor_15_topics.create_doc_to_topic_matrix(list_of_barriers, list_of_record_ids)
    doc_to_20_topic_matrix = lda_processor_20_topics.create_doc_to_topic_matrix(list_of_barriers, list_of_record_ids)
    doc_to_25_topic_matrix = lda_processor_25_topics.create_doc_to_topic_matrix(list_of_barriers, list_of_record_ids)
    doc_to_30_topic_matrix = lda_processor_30_topics.create_doc_to_topic_matrix(list_of_barriers, list_of_record_ids)

    # concatenating the rest of the data from the patients visits with the patient navigator
    doc_to_5_topic_matrix = pd.concat([doc_to_5_topic_matrix, other_patient_visit_data], axis=1)
    doc_to_10_topic_matrix = pd.concat([doc_to_10_topic_matrix, other_patient_visit_data], axis=1)
    doc_to_15_topic_matrix = pd.concat([doc_to_15_topic_matrix, other_patient_visit_data], axis=1)
    doc_to_20_topic_matrix = pd.concat([doc_to_20_topic_matrix, other_patient_visit_data], axis=1)
    doc_to_25_topic_matrix = pd.concat([doc_to_25_topic_matrix, other_patient_visit_data], axis=1)
    doc_to_30_topic_matrix = pd.concat([doc_to_30_topic_matrix, other_patient_visit_data], axis=1)

    print(doc_to_5_topic_matrix)

    all_doc_to_topic_matrices = [doc_to_5_topic_matrix, doc_to_10_topic_matrix, doc_to_15_topic_matrix,
                                doc_to_20_topic_matrix, doc_to_25_topic_matrix, doc_to_30_topic_matrix]

    # saves the following objects to a pickle file
    '''
        1. vectorizer
        2. all_lda_processors
        3. all_lda_models
        4. all_doc_to_topic_matrices
        5. list_of_documents
        6. list_of_barriers
        7. document_to_word matrix
    '''
    with open('china_ALL_patients_ALL_5_10_15_20_25_30.pkl', 'wb') as fout:
        pickle.dump((vectorizer, all_lda_processors, all_lda_models, all_doc_to_topic_matrices, list_of_documents, list_of_barriers, doc_to_word_matrix, other_patient_visit_data), fout)

if __name__ == "__main__":
    main()