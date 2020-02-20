# demo_china_save2.py
# merge excel files together into a pandas dataframe

# first make j_testing.csv merge with m_testing.csv

import csv
import time
import pickle
import numpy as np
import pandas as pd

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor
from classifier_processor import ClassifierProcessor
from display_notes import DisplayNotes

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# CLASSIFIERS
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn import svm

from sklearn.metrics import classification_report, confusion_matrix

df_j = pd.read_excel('j_testing.xlsx')
df_m = pd.read_excel('m_testing.xlsx')

print("Here is df_j:")
print(df_j)

# print("here is df_m:")
# print(df_m)

# df_j_m = df_j.join(df_m.set_index('record_id'), on='record_id')
# print(df_j_m)

# df_tracking_log = pd.read_excel('Tracking_Log_1-10_for_NEIU_excel.xlsx')
# df_demographics = pd.read_excel('PN_demographics_neiu.xlsx')
# print(df_tracking_log)
# print(df_demographics)
# # df_all_INFO = df_tracking_log.merge(df_demographics, on='record_id', how='outer')
# df_all_INFO = df_demographics.merge(df_tracking_log, on='record_id', how='inner')



# # df_all_INFO = df_all_INFO.loc[:,'record_id':'navigation_comments1']
# df_all_INFO = df_all_INFO.loc[:,'record_id':'barrier1']


# df_all_INFO.columns = df_all_INFO.columns.str.replace('barrier1','Classification')

# print(df_all_INFO)
# df_all_INFO.to_csv('df_all_INFO.csv', encoding='utf-8', index=False)

# print(type(df_all_INFO.loc[0][0])) #numpy.float64
# print(type(df_all_INFO.loc[0][1])) #numpy.float64
# print(type(df_all_INFO.loc[0][2])) # str
# print(type(df_all_INFO.loc[0][3])) # str
# print(type(df_all_INFO.loc[0][4])) # str
# print(type(df_all_INFO.loc[0][5])) # str
# print(type(df_all_INFO.loc[0][6])) # int
# print(type(df_all_INFO.loc[0][7])) # str
# print(type(df_all_INFO.loc[0][8])) # str
# print(type(df_all_INFO.loc[0][9])) # int
# print(type(df_all_INFO.loc[0][10])) # numpyfloat64
# print(type(df_all_INFO.loc[0][11])) # str
# print(type(df_all_INFO.loc[0][12])) # numpyfloat64
# print(type(df_all_INFO.loc[0][13])) # pandas time stamp
# print(type(df_all_INFO.loc[0][14])) # numpy int 64
# print(type(df_all_INFO.loc[0][15])) # numpy int 64
# print(type(df_all_INFO.loc[0][16])) # .. 
# print(type(df_all_INFO.loc[0][17])) # .. 
# print(type(df_all_INFO.loc[0][18]))
# print(type(df_all_INFO.loc[0][19]))
# print(type(df_all_INFO.loc[0][20]))
# print(type(df_all_INFO.loc[0][21]))
# print(type(df_all_INFO.loc[0][22]))
# print(type(df_all_INFO.loc[0][23]))
# print(type(df_all_INFO.loc[0][24]))
# print(type(df_all_INFO.loc[0][25]))
# print(type(df_all_INFO.loc[0][26]))
# print(type(df_all_INFO.loc[0][27]))
# print(type(df_all_INFO.loc[0][28])) # numpy int 64
# print(type(df_all_INFO.loc[0][29])) # numpy.float64

# del df_all_INFO['date_of_service1']
# del df_all_INFO['record_id']

# print(df_all_INFO.head())  # 30 columns
# print(df_all_INFO.select_dtypes(include=[object]))   # 9 columns
# df_all_INFO.select_dtypes(include=[object]).to_csv('df_all_CATEGORICAL.csv', encoding='utf-8', index=False)
# print(df_all_INFO.select_dtypes(exclude=[object]))   # 21 columns
# df_all_INFO.select_dtypes(exclude=[object]).to_csv('df_all_NUMERICAL.csv', encoding='utf-8', index=False)

# imp = SimpleImputer(strategy='most_frequent')
# # imp1 = SimpleImputer(missing_values=None, strategy='constant')

# X_numerical = df_all_INFO.select_dtypes(exclude=[object])
# X_categorical = df_all_INFO.select_dtypes(include=[object])

# # X_categorical = imp1.fit_transform(X_categorical)
# X_imputed = imp.fit_transform(X_numerical)
# print(X_imputed)
# print(type(X_imputed))
# X_2 = pd.DataFrame(data=X_imputed[1:,1:],
#                     columns=['1','2','3','4','5','6','7','8','9','10','11','12','14','15','16','17','18','Classification'])
# print(X_2)
# print(type(X_2))
# print(X_2.info())
# X_2.to_csv('df_all_TESTIING.csv', encoding='utf-8', index=False)

# classifying_patients = ClassifierProcessor(doc_to_topic_matrix=X_2, unseen_doc_features=[], classifier='DT')
# classifying_patients.train_classifier()
# # X_categorical3 = pd.DataFrame(data=X_c[0:,0:])

# print(X_categorical)
# print(type(X_categorical))

# print(X_categorical.shape)
# X_categorical.dropna(axis=0, how='any', thresh=None, subset=None)
# X_categorical.to_csv('df_all_TESTIING2.csv', encoding='utf-8', index=False)
# # encoding numerical here before we do one hot encoding
# # le = LabelEncoder()

# print(pd.get_dummies(X_categorical))
# X_categorical_1hot = pd.get_dummies(X_categorical)
# X_categorical_1hot.to_csv('df_all_TESTIING3.csv', encoding='utf-8', index=False)

# # X_categorical4 = le.fit_transform(X_categorical)
# # enc = preprocessing.OneHotEncoder()
# # enc.fit(X_categorical)
# # onehotlabels = enc.transform(X_categorical).toarray()
# # print(onehotlabels.shape)
