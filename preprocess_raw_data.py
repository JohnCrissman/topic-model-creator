# preprocess_raw_data.py
''' Opening the following excel files in current directory:
        1. visits_1-10.csv
        2. visits_11-20.csv
        3. visits_21-30.csv
        4. visits_31-40.csv
        5. visits_41-50.csv

'''
from pprint import pprint
import time
import pickle
import pandas as pd
import numpy as np

def main():
    df_demographics = pd.read_excel('PN_demographics_neiu.xlsx')




    df_visits_1_through_10 = pd.read_csv('visits_1-10.csv')
    df_visits_11_through_20 = pd.read_csv('visits_11-20.csv')
    df_visits_21_through_30 = pd.read_csv('visits_21-30.csv')
    df_visits_31_through_40 = pd.read_csv('visits_31-40.csv')
    df_visits_41_through_50 = pd.read_csv('visits_41-50.csv')

    print(df_visits_1_through_10)
    





if __name__ == "__main__":
    main()