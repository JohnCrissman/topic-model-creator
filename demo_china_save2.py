# demo_china_save2.py
# merge excel files together into a pandas dataframe

# first make j_testing.csv merge with m_testing.csv

import csv
import pandas as pd

df_j = pd.read_excel('j_testing.xlsx')
df_m = pd.read_excel('m_testing.xlsx')

print("Here is df_j:")
print(df_j)

print("here is df_m:")
print(df_m)

df_j_m = df_j.join(df_m.set_index('record_id'), on='record_id')
print(df_j_m)

df_tracking_log = pd.read_excel('Tracking_Log_1-10_for_NEIU_excel.xlsx')
df_demographics = pd.read_excel('PN_demographics_neiu.xlsx')
print(df_tracking_log)
print(df_demographics)
# df_all_INFO = df_tracking_log.merge(df_demographics, on='record_id', how='outer')
df_all_INFO = df_demographics.merge(df_tracking_log, on='record_id', how='inner')



df_all_INFO = df_all_INFO.loc[:,'record_id':'navigation_comments1']

df_all_INFO.columns = df_all_INFO.columns.str.replace('barrier1','Classification')

print(df_all_INFO)
df_all_INFO.to_csv('df_all_INFO.csv', encoding='utf-8', index=False)
# classifying_patients = ClassifierProcessor(df_tracking_log)