# demo_each_patient_is_a_data_point.py

# composition of this demo file is similar to demo_ALL_patients_first.py and demo_ALL_patients_all
""" This demonstraction file will create LDA models for the following number of topics:
        5, 10, 15, 20, 25, 30

       We will use the document to topic matrices along with the barrier (label) other 
       visit information and patients demographics to classify 
        them in demo_classify_each_patient_is_a_data_point.py.  We will save all the models in this file and 
        load them in demo_classify_ALL_patients_all.py
        
        Each patient is one data point.
       We will be looking at all patient's visits that have comments and barriers(classifications)
       
       We aggregate all patient visits from the first visit up to a visit that is not
       labeled as barrier = Language/interpreter.  Therefore, we classify the data point as the barrier
       that is not equal to Language/interpreter.  If all visits are labeled as barrier = Language/interpreter, 
       then we classify the data point as Language/interpreter and only look at the first visit.
        


        Input to make LDA comes from this csv:
        df_each_visit_one_hot_encoding_sorted_by_id.csv
"""

from pprint import pprint
import pandas as pd
import csv
import math
import time
import pickle
import numpy as np

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor

''' Input is a list of patients visits sorted by patient number and then for
each patient the list is sorted from 1st visit to the their last visit'''

def set_equal_to_one(x):
    if x > 1:
        return 1
    else:
        return x
''' Strategy 1'''
def convert_df_using_technique(df):
    how_many_records = [0] * len(df)
    can_we_append = True # flag to let us know when to skip rows we are not using (when flag is false)
    input_data = pd.DataFrame() # creates a new dataframe that is empty
    
    for i in range(len(df)):
        x = df.loc[i, 'record_id']
        
        if i < (len(df) - 1):
            if df.loc[i+1, 'record_id'] == x:
                continue
            else:
                new_data = df.loc[df['record_id'] == x]
                if len(new_data) is 1:
                    input_data = input_data.append(new_data, ignore_index=True)
                else:
                    if len(np.unique(new_data.Classification)) == 1 or new_data.iloc[0]['Classification'] != 'Language/interpreter':
                        input_data = input_data.append(new_data.iloc[[0]],ignore_index=True)
                    
                        
                         
                    else:
                        # print(np.where(new_data['Classification'] != 'Language/interpreter'))
                        # print(type(np.where(new_data['Classification'] != 'Language/interpreter')))
                        # print(np.where(new_data['Classification'] != 'Language/interpreter')[0])
                        # print(type(np.where(new_data['Classification'] != 'Language/interpreter')[0]))
                        # print(np.where(new_data['Classification'] != 'Language/interpreter')[0].size)
                        # print(np.where(new_data['Classification'] != 'Language/interpreter')[0][0])

                        # last_index_of_new_data = np.where(new_data['Classification'] != 'Language/interpreter')[0][0]
                        new_data = new_data[0:np.where(new_data['Classification'] != 'Language/interpreter')[0][0]+1]
                        new_data = new_data.reset_index(drop=True)
                        
                        # barrier = new_data.iloc[0, 'Classification']
                        # print(type(np.where(new_data['Classification'] != 'Language/interpreter')[0][0]))
                        # print(new_data)   # rows up to barrier != Language/interpreter
                        # print(new_data.iloc[len(new_data) - 1]['Classification'])   # barrier for the last row (will not be language/interpreter)

                        # trying to aggregating the rows.. first taking the sum
                        # pprint(new_data.append(new_data.sum().rename('Total')))
                        # print("HEY!!!!!!!!!!!!!!")
                        new_new_data = new_data.append(new_data.sum().rename('000'))
                        new_new_data.iloc[len(new_new_data)-1, new_new_data.columns.get_loc('record_id')] = x
                        barrier = new_data.loc[len(new_data)-1, 'Classification']
                        new_new_data.iloc[len(new_new_data)-1, new_new_data.columns.get_loc('Classification')] = barrier
                        length_of_action_taken = new_data.loc[len(new_data)-1, 'length_of_action_taken']
                        new_new_data.iloc[len(new_new_data)-1, new_new_data.columns.get_loc('length_of_action_taken')] = length_of_action_taken

                        ### ISSUE:  summing up barriers as well as one long string
                        ### 1. first, check to see if that is the only issue (compare new csv with one_hot_encoding csv)
                        ### 2. fix the issue.

                        # print(new_new_data)
                        # print(new_new_data[len(new_new_data)-1])
                        input_data = input_data.append(new_new_data.iloc[[len(new_new_data)-1]], ignore_index=True)
                        # print(new_new_data[len(new_new_data)-1]['Classification'])


                        
        else:
            
            new_data = df.loc[df['record_id'] == x]
            if len(new_data) is 1:
                input_data = input_data.append(new_data, ignore_index=True)
            else:
                if len(np.unique(new_data.Classification)) == 1 or new_data.iloc[0]['Classification'] != 'Language/interpreter':
                        input_data = input_data.append(new_data.iloc[[0]], ignore_index=True)         
                else:
                    # print(np.where(new_data['Classification'] != 'Language/interpreter'))
                    # print(type(np.where(new_data['Classification'] != 'Language/interpreter')))
                    # print(np.where(new_data['Classification'] != 'Language/interpreter')[0])
                    # print(type(np.where(new_data['Classification'] != 'Language/interpreter')[0]))
                    # print(np.where(new_data['Classification'] != 'Language/interpreter')[0].size)
                    # print(np.where(new_data['Classification'] != 'Language/interpreter')[0][0])

                    # last_index_of_new_data = np.where(new_data['Classification'] != 'Language/interpreter')[0][0]
                    new_data = new_data[0:np.where(new_data['Classification'] != 'Language/interpreter')[0][0]+1]
                    new_data = new_data.reset_index(drop=True)
                    
                    # print(type(np.where(new_data['Classification'] != 'Language/interpreter')[0][0]))
                    # print(new_data)   # rows up to barrier != Language/interpreter
                    # print(new_data.iloc[len(new_data) - 1]['Classification'])   # barrier for the last row (will not be language/interpreter)

                    # trying to aggregating the rows.. first taking the sum
                    # pprint(new_data.append(new_data.sum().rename('Total')))
                    # print("HEY!!!!!!!!!!!!!!")
                    new_new_data = new_data.append(new_data.sum().rename('000'))
                    new_new_data.iloc[len(new_new_data)-1, new_new_data.columns.get_loc('record_id')] = x
                    barrier = new_data.loc[len(new_data)-1, 'Classification']
                    new_new_data.iloc[len(new_new_data)-1, new_new_data.columns.get_loc('Classification')] = barrier
                    length_of_action_taken = new_data.loc[len(new_data)-1, 'length_of_action_taken']
                    new_new_data.iloc[len(new_new_data)-1, new_new_data.columns.get_loc('length_of_action_taken')] = length_of_action_taken
                    
                    ## if value is greater than 1, set value to 1.
                    

                    # new_new_data['All Languages (select all that apply) (choice=Cantonese)_x'].loc[(new_new_data['All Languages (select all that apply) (choice=Cantonese)_x'] > 1)] = 1
                    # new_new_data['All Languages (select all that apply) (choice=Mandarin)_x'].loc[(new_new_data['All Languages (select all that apply) (choice=Mandarin)_x'] > 1)] = 1
                    # new_new_data['All Languages (select all that apply) (choice=Toishanese)_x'].loc[(new_new_data['All Languages (select all that apply) (choice=Toishanese)_x'] > 1)] = 1
                    # new_new_data['All Languages (select all that apply) (choice=English)_x'].loc[(new_new_data['All Languages (select all that apply) (choice=English)_x'] > 1)] = 1

                    # new_new_data['Type of service (select all that apply) (choice=Breast - Screening)_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Breast - Screening)_x'] > 1)] = 1
                    # new_new_data['Type of service (select all that apply) (choice=Breast - Dx (MMG/US/MRI))_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Breast - Dx (MMG/US/MRI))_x'] > 1)] = 1
                    # new_new_data['Type of service (select all that apply) (choice=Breast - Bx (US/Stereotactic))_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Breast - Bx (US/Stereotactic))_x'] > 1)] = 1
                    # new_new_data['Type of service (select all that apply) (choice=Breast - F/u (results))_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Breast - F/u (results))_x'] > 1)] = 1

                    # new_new_data['Type of service (select all that apply) (choice=Breast - Tx (Surgery))_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Breast - Tx (Surgery))_x'] > 1)] = 1
                    # new_new_data['Type of service (select all that apply) (choice=Breast - Other)_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Breast - Other)_x'] > 1)] = 1
                    # new_new_data['Type of service (select all that apply) (choice=Cervical - Screening)_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Cervical - Screening)_x'] > 1)] = 1
                    # new_new_data['Type of service (select all that apply) (choice=Cervical - Dx (COLPO))_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Cervical - Dx (COLPO))_x'] > 1)] = 1

                    # new_new_data['Type of service (select all that apply) (choice=Cervical - Bx (COLPO))_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Cervical - Bx (COLPO))_x'] > 1)] = 1
                    # new_new_data['Type of service (select all that apply) (choice=Cervical - F/u (results))_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Cervical - F/u (results))_x'] > 1)] = 1
                    # new_new_data['Type of service (select all that apply) (choice=Cervical - Tx (Surgery/LEEP/CONE))_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Cervical - Tx (Surgery/LEEP/CONE))_x'] > 1)] = 1
                    # new_new_data['Type of service (select all that apply) (choice=Cervical - Other)_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Cervical - Other)_x'] > 1)] = 1

                    # new_new_data['Type of service (select all that apply) (choice=Cervical - F/u (results))_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Cervical - F/u (results))_x'] > 1)] = 1
                    # new_new_data['Type of service (select all that apply) (choice=Cervical - Tx (Surgery/LEEP/CONE))_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Cervical - Tx (Surgery/LEEP/CONE))_x'] > 1)] = 1
                    # new_new_data['Type of service (select all that apply) (choice=Cervical - Other)_x'].loc[(new_new_data['Type of service (select all that apply) (choice=Cervical - Other)_x'] > 1)] = 1
                    

                    # print(new_new_data)
                    # print(new_new_data[len(new_new_data)-1])
                    input_data = input_data.append(new_new_data.iloc[[len(new_new_data)-1]], ignore_index=True)
                    # print(new_new_data[len(new_new_data)-1]['Classification'])
                
                    

            

    input_data['length_of_action_taken'].fillna((input_data['length_of_action_taken'].mean()), inplace=True)
    # modDfObj = input_data.apply(lambda x: 1 if (x.name != 'record_id') & (x.name != 'Classification') & (x.name != 'length_of_action_taken') & (x.name != 'comments') & (x > 1) else x)

    # if a value is greater than 1, set it to 1
    # input_data[['All Languages (select all that apply) (choice=Cantonese)_x', 'All Languages (select all that apply) (choice=Mandarin)_x',
    #             'All Languages (select all that apply) (choice=Toishanese)_x', 'All Languages (select all that apply) (choice=English)_x',
    #             'Type of service (select all that apply) (choice=Breast - Screening)_x', 'Type of service (select all that apply) (choice=Breast - Dx (MMG/US/MRI))_x',
    #             'Type of service (select all that apply) (choice=Breast - Bx (US/Stereotactic))_x', 'Type of service (select all that apply) (choice=Breast - F/u (results))_x',
    #             'Type of service (select all that apply) (choice=Breast - Tx (Surgery))_x', 'Type of service (select all that apply) (choice=Breast - Other)_x',
    #             'Type of service (select all that apply) (choice=Cervical - Screening)_x', 'Type of service (select all that apply) (choice=Cervical - Dx (COLPO))_x',
    #             'Type of service (select all that apply) (choice=Cervical - Bx (COLPO))_x', 'Type of service (select all that apply) (choice=Cervical - F/u (results))_x',
    #             'Type of service (select all that apply) (choice=Cervical - Tx (Surgery/LEEP/CONE))_x', 'Type of service (select all that apply) (choice=Cervical - Other)_x',
    #             'Type of service (select all that apply) (choice=Other - health insurance)_x', 'Type of service (select all that apply) (choice=Other - medical)_x',
    #             'Type of service (select all that apply) (choice=Other - other)_x', 'Cantonese', 'Mandarin', 'Toishanese', 'Email', 'In person', 'Online instant message',
    #             'Phone call', 'Social/practical support_x', 'Text message', 'Accompaniment', 'Action pending / No action', 'Arrangements', 'Education', 'Language/interpreter_y',
    #             'Records/Recordkeeping', 'Referrals/ Direct Contact', 'Scheduling appointment', 'Social/practical support_y', 'Support']].apply(set_equal_to_one)
    
    return input_data

def main():

    # create all lda_models and use pickle to save
    # the models and other objects to load somewhere else.

    # creating dataframe from csv file
    dataframe = pd.read_csv('df_each_visit_one_hot_encoding_sorted_by_id.csv')
    print(dataframe) # [1486 rows x 42 columns]


    df = convert_df_using_technique(df = dataframe)
    
    
    df.to_csv('df_each_patient_is_a_data_point.csv', encoding='utf-8', index=False)

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

    print(list_of_documents)
    print(list_of_barriers)
    print(list_of_record_ids)

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
        8. other_patient_visit_data
    '''
    with open('china_ALL_patients_each_patient_one_data_point_ALL_5_10_15_20_25_30.pkl', 'wb') as fout:
        pickle.dump((vectorizer, all_lda_processors, all_lda_models, all_doc_to_topic_matrices, list_of_documents, list_of_barriers, doc_to_word_matrix, other_patient_visit_data), fout)

if __name__ == "__main__":
    main()