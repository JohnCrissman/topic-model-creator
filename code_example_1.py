# code_example_1_save.py

from pprint import pprint
import pandas as pd   
import csv
import pickle

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor

def main():

    # converting code_example_1_data into a pandas dataframe
    df = pd.read_csv('code_example_1_data.csv')
    print(df)

    '''
        Adding column names to the data.
        The column with the labels must be called 'Classification' 
            in order to run classification algorithms later on.
    '''
    df.columns = ['documents', 'Classification']
    print(df)

    '''
        Creating a separate list of the documents for CorpusProcessor()
        Creating a separate list of the classifications to add to the
            document to topic matrix that comes from LDAProcessor()
    '''
    list_of_documents = []
    list_of_classifications = []
    for i in range(len(df)):
        list_of_documents.append(df.iloc[i,0])
        list_of_classifications.append(df.iloc[i,1])

    print(list_of_documents)
    print(list_of_classifications)



if __name__ == "__main__":
    main()