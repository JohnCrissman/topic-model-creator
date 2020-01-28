## reading in medical dataset and manipulating it.
## using this for testing and getting the program working

import csv
import pandas as pd  
import math
import time
import pickle

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor

# read by default 1st sheet of an excel file
df = pd.read_excel('Tracking_Log_1-10_for_NEIU_excel.xlsx')

df.to_csv('aa_testing1.csv', encoding='utf-8', index=False)

dfObj1 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj2 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj3 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj4 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj5 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj6 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj7 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj8 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj9 = pd.DataFrame(columns=['navigation_comments', 'barrier'])
dfObj10 = pd.DataFrame(columns=['navigation_comments', 'barrier'])

dfObj1['navigation_comments'], dfObj1['barrier'] = df['navigation_comments1'], df['barrier1']
dfObj2['navigation_comments'], dfObj2['barrier'] = df['navigation_comments2'], df['barrier2']
dfObj3['navigation_comments'], dfObj3['barrier'] = df['navigation_comments3'], df['barrier3']
dfObj4['navigation_comments'], dfObj4['barrier'] = df['navigation_comments4'], df['barrier4']
dfObj5['navigation_comments'], dfObj5['barrier'] = df['navigation_comments5'], df['barrier5']
dfObj6['navigation_comments'], dfObj6['barrier'] = df['navigation_comments6'], df['barrier6']
dfObj7['navigation_comments'], dfObj7['barrier'] = df['navigation_comments7'], df['barrier7']
dfObj8['navigation_comments'], dfObj8['barrier'] = df['navigation_comments8'], df['barrier8']
dfObj9['navigation_comments'], dfObj9['barrier'] = df['navigation_comments9'], df['barrier9']
dfObj10['navigation_comments'], dfObj10['barrier'] = df['navigation_comments10'], df['barrier10']

frames = [dfObj1, dfObj2, dfObj3, dfObj4, dfObj5, dfObj6, dfObj7, dfObj8, dfObj9, dfObj10]

# this is the dataframe I need.
# when creating the docs for LDA, ignore the rows with no comments
result = pd.concat(frames)

result.to_csv('aaaaa_testing1.csv', encoding='utf-8', index=False)

# remove rows with missing comments
df_no_missing = result[result.navigation_comments.notnull()]

# remove rows with missing barriers
df_no_missing = df_no_missing[df_no_missing.barrier.notnull()]

# reset index
df_no_missing.reset_index(drop=True, inplace=True)

#printing dataframe I will be reading from for LDA-Classification
print(df_no_missing)

# saving dataframe to df_no_missing_values.csv
df_no_missing.to_csv('df_no_missing_values.csv', encoding='utf-8', index=False)

# create a list of strings from iterating through the dataframe

list_of_documents = []
list_of_barriers = []
for i in range(len(df_no_missing)):
    list_of_documents.append(df_no_missing.iloc[i,0])
    list_of_barriers.append(df_no_missing.iloc[i,1])

print("Here is the list of documents: ")
print(list_of_documents)
print()
print("Here is the list of barriers: ")
print(list_of_barriers)

corpus = CorpusProcessor()
list_of_list_of_words = corpus.create_list_of_list_of_words(list_of_documents)

doc_to_word_matrix = corpus.create_doc_to_word_matrix(list_of_list_of_words)
df_doc_to_word_matrix = pd.DataFrame((doc_to_word_matrix).toarray())
df_doc_to_word_matrix.to_csv('china_doc_to_word_matrix.csv', encoding='utf-8', index=False)

vectorizer = corpus.get_vectorizer()

# We are choosing 10 topics as second parameter
topic_model_creator = LDAProcessor(doc_to_word_matrix, 10, vectorizer, False)

lda_model = topic_model_creator.get_lda_model()

# returns Document to Topic matrix and creates a csv file of the matrix
doc_to_topic_matrix = topic_model_creator.create_doc_to_topic_matrix(list_of_barriers)
    
df_doc_to_topic_matrix = pd.DataFrame(doc_to_topic_matrix)
df_doc_to_topic_matrix.to_csv('china_doc_to_topic_matrix.csv', encoding='utf-8', index=False)

# save vectorizer, doc to word matrix, and topic model
with open('china_finalized_model.pkl', 'wb') as fout:
    pickle.dump((vectorizer, doc_to_word_matrix, lda_model, doc_to_topic_matrix, list_of_barriers), fout)