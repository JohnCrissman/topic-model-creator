# demo_classify_ALL_patients_first.py

# How is this file different from demo_classify_LDA_patients_first.py?
#   In this file we are not only using LDA on the patients first visit.
#   We will be using other information gathered from the first visit (like type of service)
#     as well as the patients demographics!

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
            - This is a list that contains the following:  # objects of class LDAProcessor from lda_processor.py
                0. lda_processor_5_topics
                1. lda_processor_10_topics
                2. lda_processor_15_topics
                3. lda_processor_20_topics
                4. lda_processor_25_topics
                5. lda_processor_30_topics

        3. all_lda_models
            - This is a list that contains the following:
                0. lda_model_5_topics   # objects of class LatentDirichletAllocation from sklearn 
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
import numpy as np

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor
from classifier_processor import ClassifierProcessor
from display_notes import DisplayNotes

from sklearn.datasets import load_iris

def train_and_test_classifier(input_for_classifier, tuple_to_predict=[]):
    print("Training and testing multiple classifiers classifiers for", len(input_for_classifier.columns)-1,"topics")
    classifier = ClassifierProcessor(doc_to_topic_matrix= input_for_classifier, unseen_doc_features= tuple_to_predict)
    print(input_for_classifier)
    classifier.train_and_test_classifier_k_fold()


def show_top_n_words_for_each_topic(num_words, lda_processor_object, filename):
    topic_to_word_matrix = lda_processor_object.topic_to_word_matrix_n_words(num_words, filename)
    pprint(topic_to_word_matrix)

# takes in a dataframe and returns a dataframe with only the rows that != label for the Classfication column
def remove_rows_with_this_label(label, df):
    new_df = df[df.Classification != label]
    new_df.reset_index(drop=True, inplace=True) # reset index
    return new_df

''' add demographics and types of service to document_to_topic_matrix '''
def add_demographics_and_other_to_doc_to_topic(document_to_topic_matrix):

    df_demographics = pd.read_excel('PN_demographics_neiu.xlsx')
    df_all_INFO = df_demographics.merge(document_to_topic_matrix, on='record_id', how='inner')
    df_all_INFO.rename(columns={df_all_INFO.columns[4]: "education", df_all_INFO.columns[5]: "born_in_US",
                                df_all_INFO.columns[6]: "year_entered_US", df_all_INFO.columns[7]: "english_fluency",
                                df_all_INFO.columns[8]: "native_land", df_all_INFO.columns[9]: "zip_code",
                                df_all_INFO.columns[10]: "family_members_in_household", df_all_INFO.columns[11]: "household_income"}, inplace = True)

    ## one hot encoding for all attributes that need it
    # First, create a dataframe that is a one hot encoding of a column
    # Second, merge the two dataframes
    # Third, delete the original column (no longer needed)
    # Below, this is performed on each relevant column
    dummy = pd.get_dummies(df_all_INFO['education'])
    df = df_all_INFO.merge(dummy, left_index=True, right_index=True)
    del df['education']
    del df['born_in_US']
    dummy = pd.get_dummies(df['occupational status'])
    df = df.merge(dummy, left_index=True, right_index=True)
    del df['occupational status']
    dummy = pd.get_dummies(df['english_fluency'])
    df = df.merge(dummy, left_index=True, right_index=True)
    del df['english_fluency']

    # I could probably delete this one because there are over 100 values for this attribute
    dummy = pd.get_dummies(df['native_land'])
    df = df.merge(dummy, left_index=True, right_index=True)
    del df['native_land']

    dummy = pd.get_dummies(df['family_members_in_household'])
    df = df.merge(dummy, left_index=True, right_index=True)
    del df['family_members_in_household']

    dummy = pd.get_dummies(df['marital status '])
    df = df.merge(dummy, left_index=True, right_index=True)
    del df['marital status ']

    dummy = pd.get_dummies(df['household_income'])
    df = df.merge(dummy, left_index=True, right_index=True)
    del df['household_income']
    
    cols = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    df_type_of_service = pd.read_excel('Tracking_Log_1-10_for_NEIU_excel.xlsx', usecols=cols)
    
    df_ready_for_classification = df.merge(df_type_of_service, on='record_id', how='inner')

    # take out record id for classification purposes
    del df_ready_for_classification['record_id']
    
    df_ready_for_classification.at[53,'year_entered_US'] = 2000
    df_ready_for_classification.fillna(method='pad', inplace=True)
    
    df_ready_for_classification.to_csv('df_ALL_patients_first.csv', encoding='utf-8', index=False)
    return df_ready_for_classification

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['Classification'] = pd.Series(sklearn_dataset.target)
    return df

def main():
    with open('china_ALL_patients_first_5_10_15_20_25_30.pkl', 'rb') as f:
        vectorizer, all_lda_processors, all_lda_models, all_doc_to_topic_matrices, list_of_documents, list_of_barriers, doc_to_word_matrix = pickle.load(f)

    # # Show 5 topics and top 10 words
    # show_top_n_words_for_each_topic(num_words= 10, lda_processor_object= all_lda_processors[0], filename= "topic_5_to_word_10_1st_visit.csv")
    
    # # Show 10 topics and top 10 words
    # show_top_n_words_for_each_topic(num_words= 10, lda_processor_object= all_lda_processors[1], filename= "topic_10_to_word_10_1st_visit.csv")
    

    
    # print("These are the 5 topics distribution for the first document: \n")
    # print(all_lda_models[0].transform(doc_to_word_matrix)[0])

    # print("These are the 5 topics distribution for the second document: \n")
    # print(all_lda_models[0].transform(doc_to_word_matrix)[1])

    # print("There are the 5 topics distribution for the third document: \n")
    # print(all_lda_models[0].transform(doc_to_word_matrix)[2])

    # print(all_doc_to_topic_matrices[0])

    ''' testing my code on a dataset that I know will perform well'''
    iris = sklearn_to_df(load_iris())
    print(iris)
    print(type(iris))
    iris.to_csv('IRIS_TESTING.csv', encoding='utf-8', index=False)
    ##### TEST MY CLASSIFIER ON DATA THAT I KNOW SHOULD PERFORM WELL
    train_and_test_classifier(input_for_classifier= iris)

    # testing 5 topics
    # matrix_for_classifier = add_demographics_and_other_to_doc_to_topic(document_to_topic_matrix= all_doc_to_topic_matrices[0])
    # train_and_test_classifier(input_for_classifier= matrix_for_classifier)

    # # testing 10 topics
    # matrix_for_classifier = add_demographics_and_other_to_doc_to_topic(document_to_topic_matrix= all_doc_to_topic_matrices[1])
    # train_and_test_classifier(input_for_classifier= matrix_for_classifier)

    # # testing 15 topics
    # matrix_for_classifier = add_demographics_and_other_to_doc_to_topic(document_to_topic_matrix= all_doc_to_topic_matrices[2])
    # train_and_test_classifier(input_for_classifier= matrix_for_classifier)

    # # testing 20 topics
    # matrix_for_classifier = add_demographics_and_other_to_doc_to_topic(document_to_topic_matrix= all_doc_to_topic_matrices[3])
    # train_and_test_classifier(input_for_classifier= matrix_for_classifier)

    # # testing 25 topics
    # matrix_for_classifier = add_demographics_and_other_to_doc_to_topic(document_to_topic_matrix= all_doc_to_topic_matrices[4])
    # train_and_test_classifier(input_for_classifier= matrix_for_classifier)

    # # testing 30 topics
    # matrix_for_classifier = add_demographics_and_other_to_doc_to_topic(document_to_topic_matrix= all_doc_to_topic_matrices[5])
    # train_and_test_classifier(input_for_classifier= matrix_for_classifier)

    '''
        In the following tests we are taking out label/classification
        "5.0" or "language/interpreter" from our data set.  
    '''
    # pprint(all_doc_to_topic_matrices[0])
    # my_df = remove_rows_with_this_label(label=5.0, df = all_doc_to_topic_matrices[0])
    
    # pprint(my_df)



    # testing 5 topics without tuples with language/interpreter label
    # matrix_for_classifier = add_demographics_and_other_to_doc_to_topic(document_to_topic_matrix= all_doc_to_topic_matrices[0])
    # df = remove_rows_with_this_label(label=5.0, df = matrix_for_classifier)
    # train_and_test_classifier(input_for_classifier= df)

    # testing 10 topics without tuples with language/interpreter label
    # matrix_for_classifier = add_demographics_and_other_to_doc_to_topic(document_to_topic_matrix= all_doc_to_topic_matrices[1])
    # df = remove_rows_with_this_label(label=5.0, df = matrix_for_classifier)
    # train_and_test_classifier(input_for_classifier= df)

    # testing 15 topics without tuples with language/interpreter label
    # matrix_for_classifier = add_demographics_and_other_to_doc_to_topic(document_to_topic_matrix= all_doc_to_topic_matrices[2])
    # df = remove_rows_with_this_label(label=5.0, df = matrix_for_classifier)
    # train_and_test_classifier(input_for_classifier= df)

    # testing 20 topics without tuples with language/interpreter label
    # matrix_for_classifier = add_demographics_and_other_to_doc_to_topic(document_to_topic_matrix= all_doc_to_topic_matrices[3])
    # df = remove_rows_with_this_label(label=5.0, df = matrix_for_classifier)
    # train_and_test_classifier(input_for_classifier= df)

    # testing 25 topics without tuples with language/interpreter label
    # matrix_for_classifier = add_demographics_and_other_to_doc_to_topic(document_to_topic_matrix= all_doc_to_topic_matrices[4])
    # df = remove_rows_with_this_label(label=5.0, df = matrix_for_classifier)
    # train_and_test_classifier(input_for_classifier= df)

    # testing 30 topics without tuples with language/interpreter label
    # matrix_for_classifier = add_demographics_and_other_to_doc_to_topic(document_to_topic_matrix= all_doc_to_topic_matrices[5])
    # df = remove_rows_with_this_label(label=5.0, df = matrix_for_classifier)
    # train_and_test_classifier(input_for_classifier= df)


    # # let's visuallize a document while highlighting most prevalent topics and their words
    # "-----------------------------------------------------------------------------"
    # topic_distribution_for_doc = all_lda_processors[3].show_topics_for_unseen([list_of_documents[2]]).reshape(1,-1)
    # distribution = []
    # distribution.append(topic_distribution_for_doc)

    # document = []
    # document.append(list_of_documents[2])
    # words = 1000
    # filename = 'patients_topic_to_word_matrix' + str(words) + '.csv'
    # topic_to_word_matrix = all_lda_processors[3].topic_to_word_matrix_n_words(words, filename)
    # npa = np.asarray(distribution, dtype=np.float64)
    
    
    
    # display = DisplayNotes(document, npa[0], topic_to_word_matrix)
    # "-----------------------------------------------------------------------------"

    # display.display_threshold_topics_m_words(0.0000000000001, 5)
    # # display.display_top_n_topics_m_words(5, 1000)

if __name__ == "__main__":
    main()