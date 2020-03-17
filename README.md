# Topic Model Creator
*Developed by John Crissman

1. Focus (reason I created this project)
 >Predicting social determinants of health in patients.  Also, using Latent Dirichlet Allocation topic models as input into classification algorithms were another focus.
 
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

* **demo_ALL_patients_all.py**:  Using data from _df_each_visit__one_hot_encoding.csv_, this demo file will create LDA models (topic #s are 5, 10, 15, 20, 25, 30) and the appropriate matrices and save them into _china_ALL_patients_ALL_5_10_15_20_25_30.pkl_.  We are considering each visit from a patient as a data point and the barrier given by the patient navigator will be the label for that data point.  

* **demo_classify_ALL_patients_all.py**:  This class loads the objects from the pickle file _china_ALL_patients_ALL_5_10_15_20_25_30.pkl_ and concatenates the data with demographics data associated with each patient.  This demo file also runs machine learning algorithms from ClassifierProcessor() and displays the text with DisplayNotes().

* **demo_each_patient_is_a_data_point.py**: Using data from _df_each_visit_one_hot_encoding_sorted_by_id.csv_, we transform the data to a each visit is a data point strategy to each patient is a data point strategy.  We create LDA models (topic #s are 5, 10, 15, 20, 25, 30) and the appropriate matrices and save them into _china_ALL_patients_each_patient_one_data_point_ALL_5_10_15_20_25_30.pkl_.  We are considering each patient as a data point and considering all the visits up to the first occurence when their barrier is not _language/interpreter_.

* **demo_classify_each_patient_is_a_data_point.py**:  This class loads the objects from the pickle file _china_ALL_patients_each_patient_one_data_point_ALL_5_10_15_20_25_30.pkl_ and concatenates the data with demographics data associated with each patient.  This demo file also runs machine learning algorithms from ClassifierProcessor() and displays the text with DisplayNotes().

* **demo_processor_load**:  This class loads the objects from the pick file _c_finalized_model.pkl_.  These objects in _c_finalized_model.pkl_ are representative of movie review data.  Each movie review is labelled as either a positive review or a negative review.  1000 positive reviews and 1000 negative reviews were used.  Only the document to topic matrix and the label (positive or negative) were used for classification.  This file uses ClassifierProcessor() and DisplayNotes().   




## Data used for project

* **df_each_visit_one_hot_encoding.csv**:  This data shows each row as a visit from a patient and each column is an attribute.  Because of one-hot-encoding, there will be many columns/attributes.

* **PN_demographics_neiu.xlsx**:  This data has some demographics information for each patient such as age, occupational status, marital status, education level, whether or not they were born in the United States, in what year they came to the U.S., how well they speak english, where they are from, their current zip code, and household income.

* **df_each_visit_one_hot_encoding_sorted_by_id.csv**:  Same as _df_each_visit_one_hot_encoding_.csv_ above with the exception that this file is sorted by record_id.



## Pickle files and their contents

- **china_ALL_patients_ALL_5_10_15_20_25_30.pkl**:  These objects are representative of the data when considering each visit from a patient as a data point and the barrier given by the patient navigator will be the label for that data point.  
	 - `vectorizer`
	 - `all_lda_processors`
	 - `all_lda_models`
	 - `all_doc_to_topic_matrices`
	 - `list_of_documents`
	 - `list_of_barriers`
	 - `document_to_word_matrix`
	 - `other_patient_visit_data`

- **china_ALL_patients_each_patient_one_data_point_ALL_5_10_15_20_25_30.pkl**:  These objects are representative of the data when considering each patient as a data point.  We are focusing on barriers that are not language/interpreter.  If a patient only has barriers of language/interpreter from their visits, then they will be labeled as language/interpreter  
	 - `vectorizer`
	 - `all_lda_processors`
	 - `all_lda_models`
	 - `all_doc_to_topic_matrices`
	 - `list_of_documents`
	 - `list_of_barriers`
	 - `document_to_word_matrix`
	 - `other_patient_visit_data`
	 
- **c_finalized_model.pkl**:  These objects are representative of movie review data.  Each movie review is labelled as either a positive review or a negative review.  1000 positive reviews and 1000 negative reviews were used.  Only the document to topic matrix and the label (positive or negative) were used for classification.
	 - `vectorizer`
	 - `doc_to_word_matrix`
	 - `lda_model`
	 - `doc_to_topic_matrix`
	 


## Code Examples





