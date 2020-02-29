# demo_LDA_patients_all.py
""" This demonstraction file will create LDA models for the following number of topics:
        5, 10, 15, 20, 25, 30

       We will use the document to topic matrices along with the barrier (label) other 
       visit information and patients demographics to classify 
        them in demo_classify_ALL_patients_all.py.  We will save all the models in this file and 
        load them in demo_classify_ALL_patients_all.py
        
       The comments from the patient navigator for a patients visit will be one document.
       We will be looking at all patient's visits that have comments and barriers(classifications)
       
        Each visit is one data point.

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
    

if __name__ == "__main__":
    main()