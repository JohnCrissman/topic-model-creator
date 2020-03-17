# Topic Model Creator
*Developed by John Crissman

1. Focus (reason I created this project)
 Predicting social determinants of health in patients
 
2. Data
 On china town patients and produced by patient navigators at Northwestern University.  Data includes visit information, demographics, comments left by patient navigators and the barrier (social determinant of health)

3. Algorithms
 Latent Dirichlet Allocation (Topic Modeling) is used to convert text to numerical data.  Random Forests, Artificial Neural Networks, Logistic Regression, Support Vector Machines, and Gaussian Naive Bayes were used for classification.
 
4. Other uses for this program.
 Users can turn text into topic models and concatenate this data with other numerica data for predicting (classification) purposes.  
 

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
* CorpusProcessor() (corpus_processor.py):  This class prepares a collection of documents to use a vectorizer in order to make a document to word matrix.  The doc to word matrix is input into LDA.
* LDAProcessor() (lda_processor.py):  This class creates a Latent Dirichlet Allocation (LDA) model and transforms the output of the LDA in order to use as input for supervised learning.
* ClassifierProcessor() (classifier_processor.py):  This class takes a pandas dataframe such that each row is a data point and the columns are attributes of the data point.  The values in the column with the column name "Classification" are the labels associated with that respective row/data point.

## Demo files that use the above classes


## Data used for project


## Pickle files and their contents


## Code Examples




## Instructions to Use

The application window contains 3 main parts:

- Top section: search menu
- Middle section: view crimes as a map, list, or set of bar graphs
- Bottom section: change between different views, or exit the application

To use:

- Type in an address (full or partial - this works just like Google Maps to fill in incomplete addresses)
- Select a search radius from the drop-down menu.
- Click "Search" to run your query.

Repeat these steps as many times as desired.

Some crimes are missing latitude and longitude information -- these crimes are excluded from our application, since location information is very important to this application performing as expected!  These crimes are saved in a `logFile_[date information].log` file at application launch, located within `build/resources/main`.
