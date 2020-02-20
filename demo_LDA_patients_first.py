""" This demonstraction file will create LDA models for the following number of topics:
        5, 10, 15, 20, 25, 30

       We will use the document to topic matrices along with the barrier (label) to classify 
        them in demo_classify_LDA_first.py.  We will save all the models in this file and 
        load them in demo_classify_LDA_first.py
        
       The comments from the patient navigator for a patients visit will be one document.
       We will be looking at only the patient's first visit and those with comments.
       
       Only 283 out of 330 (all patients) have patient navigator comments for the first visit
       so we will only look at these 284 patients in these tests.

       For the 283 patients with comments in the first visit:
          - 199 of them are classified as language/interpreter

       For the 330 patients (all patients) in our data:
          - 239 of them are classified as language/interpreter

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

    # creating dataframe from excel file
    df = pd.read_excel('Tracking_Log_1-10_for_NEIU_excel.xlsx')

    # creating new dataframe file with columns barrier1 and 
    #    navigation_comments1 from df in the previous line.
    # size[330 X 2]
    dfObj = pd.DataFrame(columns=['navigation_comments', 'barrier'])
    dfObj['navigation_comments'], dfObj['barrier'] = df['navigation_comments1'], df['barrier1'] 

    # remove rows with missing comments
    # 283 with comments.  We eliminated 47 rows.
    dfObj = dfObj[dfObj.navigation_comments.notnull()]
    dfObj.reset_index(drop=True, inplace=True) # reset index

    # remove rows with missing barriers
    # 279 with comments and barriers.  We eliminated 4 rows.
    dfObj = dfObj[dfObj.barrier.notnull()]
    dfObj.reset_index(drop=True, inplace=True) # reset index
    
    # create list of strings s.t. each string is a comment 
    # create a list of numpy.float64 for each barrier
    # lists will be created in order of dataframe (row 0 to row n)

    list_of_documents = []
    list_of_barriers = []
    for i in range(len(dfObj)):
        list_of_documents.append(dfObj.iloc[i,0])
        list_of_barriers.append(dfObj.iloc[i,1])

    # create a CorpusProcessor object to transform our lists into input for LDA
    corpus = CorpusProcessor()
    
    list_of_list_of_words = corpus.create_list_of_list_of_words(list_of_documents)
    doc_to_word_matrix = corpus.create_doc_to_word_matrix(list_of_list_of_words)
    
    # use the two lines below to create a .csv file for doc_to_word_matrix
    # df_doc_to_word_matrix = pd.DataFrame((doc_to_word_matrix).toarray())
    # df_doc_to_word_matrix.to_csv('china_1st_visit_LDA_doc_to_word_matrix.csv', encoding='utf-8', index=False)
    
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

    # creates a doc_to_topic_matrix and appends the barriers (labels) to them
    doc_to_5_topic_matrix = lda_processor_5_topics.create_doc_to_topic_matrix(list_of_barriers)
    doc_to_10_topic_matrix = lda_processor_10_topics.create_doc_to_topic_matrix(list_of_barriers)
    doc_to_15_topic_matrix = lda_processor_15_topics.create_doc_to_topic_matrix(list_of_barriers)
    doc_to_20_topic_matrix = lda_processor_20_topics.create_doc_to_topic_matrix(list_of_barriers)
    doc_to_25_topic_matrix = lda_processor_25_topics.create_doc_to_topic_matrix(list_of_barriers)
    doc_to_30_topic_matrix = lda_processor_30_topics.create_doc_to_topic_matrix(list_of_barriers)

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
    '''
    with open('china_LDA_patients_first_5_10_15_20_25_30.pkl', 'wb') as fout:
        pickle.dump((vectorizer, all_lda_processors, all_lda_models, all_doc_to_topic_matrices, list_of_documents, list_of_barriers, doc_to_word_matrix), fout)



if __name__ == "__main__":
    main()