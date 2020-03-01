# preprocess_raw_data.py
''' Opening the following excel files in current directory:
        1. visits_1-10.csv
        2. visits_11-20.csv
        3. visits_21-30.csv
        4. visits_31-40.csv
        5. visits_41-50.csv

        This file creates the csv used by 
        demo_classify_ALL_patients_first.py   (taking entire matrix as input into classification algorithms)
        demo_LDA_patients_all.py   (creating topics distribution from comments)

'''
from pprint import pprint
import time
import pickle
import pandas as pd
import numpy as np

def main():
    ## Reading in demographics (will need to merge with others based on record_id)
    df_demographics = pd.read_excel('PN_demographics_neiu.xlsx')
    
    df_each_visit_is_one_tuple = pd.read_csv('df_each_visit_is_one_tuple.csv')
    
    ## implement one hot encoding for preferred_language, channel, and action_taken
    dummy = pd.get_dummies(df_each_visit_is_one_tuple['preferred_language'])
    df = df_each_visit_is_one_tuple.merge(dummy, left_index=True, right_index=True)
    del df['preferred_language']
    dummy = pd.get_dummies(df['channel'])
    df = df.merge(dummy, left_index=True, right_index=True)
    del df['channel']
    dummy = pd.get_dummies(df['action_taken'])
    df = df.merge(dummy, left_index = True, right_index = True)
    del df['action_taken']

    print(df_demographics)
    print(df)

    # remove rows with missing barriers (classifications)
    df = df[df.Classification.notnull()]
    df.reset_index(drop=True, inplace=True)

    # remove rows with missing comments
    df = df[df.comments.notnull()]
    df.reset_index(drop=True, inplace=True)

    # df.to_csv('df_each_visit_one_hot_encoding.csv', encoding='utf-8', index=False)





    
    # ### Reading in .csv files and turning the into pandas dataframes
    # df_visits_1_through_10 = pd.read_csv('visits_1-10.csv')
    # df_visits_11_through_20 = pd.read_csv('visits_11-20.csv')
    # df_visits_21_through_30 = pd.read_csv('visits_21-30.csv')
    # df_visits_31_through_40 = pd.read_csv('visits_31-40.csv')
    # df_visits_41_through_50 = pd.read_csv('visits_41-50.csv')
    
    # ### Replacing Checked and Unchecked with 1s and 0s
    # df_visits_1_through_10.replace('Unchecked',0, inplace=True)
    # df_visits_1_through_10.replace('Checked',1, inplace=True)
    # df_visits_11_through_20.replace('Unchecked',0, inplace=True)
    # df_visits_11_through_20.replace('Checked',1, inplace=True)
    # df_visits_21_through_30.replace('Unchecked',0, inplace=True)
    # df_visits_21_through_30.replace('Checked',1, inplace=True)
    # df_visits_31_through_40.replace('Unchecked',0, inplace=True)
    # df_visits_31_through_40.replace('Checked',1, inplace=True)
    # df_visits_41_through_50.replace('Unchecked',0, inplace=True)
    # df_visits_41_through_50.replace('Checked',1, inplace=True)

    
    # ### Sending these updated pandas dataframes to CSV for viewing
    # df_visits_1_through_10.to_csv('df_visits_1_10_TESTING.csv', encoding='utf-8', index=False)
    # df_visits_11_through_20.to_csv('df_visits_11_20_TESTING.csv', encoding='utf-8', index=False)
    # df_visits_21_through_30.to_csv('df_visits_21_30_TESTING.csv', encoding='utf-8', index=False)
    # df_visits_31_through_40.to_csv('df_visits_31_40_TESTING.csv', encoding='utf-8', index=False)
    # df_visits_41_through_50.to_csv('df_visits_41_50_TESTING.csv', encoding='utf-8', index=False)

    # ### joining all 5 files on record_id
    # df_all_INFO = df_visits_1_through_10.merge(df_visits_11_through_20, on='Record ID (automatically assigned)', how='outer')
    # df_all_INFO = df_all_INFO.merge(df_visits_21_through_30, on='Record ID (automatically assigned)', how='outer')
    # df_all_INFO = df_all_INFO.merge(df_visits_31_through_40, on='Record ID (automatically assigned)', how='outer')
    # df_all_INFO = df_all_INFO.merge(df_visits_41_through_50, on='Record ID (automatically assigned)', how='outer')
    # print(df_all_INFO)

    # ### sending the one big joined file to csv
    # df_all_INFO.to_csv('df_visits_all_together.csv', encoding='utf-8', index=False)

    ### TESTING ######## YES!! to the question below. 
    ''' Yes, it works!!!!!! '''
    ### Can I join on a record_id even if I have multiple rows with the same record_id???
    # lst0 = [1, 2, 3]
    # lst1 = ['A1', 'A2', 'A3']
    # lst2 = ['B1', 'B2', 'B3']
    # # Calling DataFrame constructor after ziping
    # # both lists, with columns specified
    # dfAB = pd.DataFrame(list(zip(lst0,lst1,lst2)),
    #                 columns =['record_id', 'A', 'B'])
    # print(dfAB)
    # lst3 = [1, 2, 3, 1, 2, 3]
    # lst4 = ['C1', 'C2', 'C3', 'C11', 'C21', 'C31']
    # lst5 = ['D1', 'D2', 'D3', 'D11', 'D21', 'D31']
    # # Calling DataFrame constructor after ziping
    # # both lists, with columns specified
    # dfCD = pd.DataFrame(list(zip(lst3, lst4, lst5)),
    #                 columns =['record_id', 'C', 'D'])
    # print(dfCD)
    # dfABCD = dfAB.merge(dfCD, on='record_id', how='inner')
    # print(dfABCD)





if __name__ == "__main__":
    main()