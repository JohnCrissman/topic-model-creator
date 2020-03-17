# Topic Model Creator
*Developed by John Crissman

1. Focus (reason I created this project)
 >Predicting social determinants of health in patients
 
2. Data
 >On china town patients and produced by patient navigators at Northwestern University.  Data includes visit information, demographics, comments left by patient navigators and the barrier (social determinant of health)

3. Algorithms
 >Latent Dirichlet Allocation (Topic Modeling) is used to convert text to numerical data.  Random Forests, Artificial Neural Networks, Logistic Regression, Support Vector Machines, and Gaussian Naive Bayes were used for classification.
 
4. Other uses for this program.
 >Users can turn text into topic models and concatenate this data with other numerica data for predicting (classification) purposes.  
 

# Setup

## Dependencies

This application was developed in Python 3 and HTML, using Visual Studio Code

 - Python 3.6.4
 - Visual Studio Code.  February 2020 (version 1.43)
 - Python packages (including several sub-packages)
	 - `pandas`
	 - `numpy`
	 - `pickle`
	 - `time`
	 - `pprint`
	 - `glob`
	 - `gensim`
	 - `sklearn`
	 - `spacy`
	 - `statistics`
	 - `seaborn`
	 - `itertools`
	 - `matplotlib`
	 - `re`
	 - `webbrowser`
	 - `heapq`
	 - `csv`
	 - `math`

# API Reference

## Classes 

* CorpusProcessor() (**corpus_processor.py**):  This class prepares a collection of documents to use a vectorizer in order to make a document to word matrix.  The doc to word matrix is input into LDA.

* LDAProcessor() (**lda_processor.py**):  This class creates a Latent Dirichlet Allocation (LDA) model and transforms the output of the LDA in order to use as input for supervised learning.

* ClassifierProcessor() (**classifier_processor.py**):  This class takes a pandas dataframe such that each row is a data point and the columns are attributes of the data point.  The values in the column with the column name "Classification" are the labels associated with that respective row/data point.

* DisplayNotes() (**display_notes.py**):  This class displays one document and highlights words different colors that are associated with topics.


## Demo files that use the above classes
### Each demo file below creates LDA models for the following topic numbers: 5, 10, 15, 20, 25, 30

* **demo_ALL_patients_all.py**:  using data from _df_each_visit__one_hot_encoding.csv_, this demo file will create LDA models and the appropriate matrices and save them into _china_ALL_patients_ALL_5_10_15_20_25_30.pkl_.  We are considering each visit from a patient as a data point and the barrier given by the patient navigator will be the label for that data point.  

* **demo_classify_ALL_patients_all.py**:

* **demo_each_patient_is_a_data_point.py**:

* **demo_classify_each_patient_is_a_data_point.py**:

* **demo_LDA_patients_first.py**:

* **demo_classify_LDA_patients_first.py**:





## Data used for project


## Pickle files and their contents

- **china_ALL_patients_ALL_5_10_15_20_25_30.pkl**:  These objects are representative of the data when considering each visit from a patient as a data point and the barrier given by the patient navigator will be the label for that data point.  
	 - `vectorizer`
	 - `all_lda_processors`
	 - `all_lda_models`
	 - `vectorizer`
	 - `all_lda_processors`
	 - `all_lda_models`
	 - `vectorizer`
	 - `all_lda_processors`
	 - `all_lda_models`
	 


## Code Examples





