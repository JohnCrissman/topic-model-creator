# Classifier Processor
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import seaborn as sns

# CLASSIFIERS
import tensorflow as tf 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import itertools

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_predict
        

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import re

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                See http://matplotlib.org/examples/color/colormaps_reference.html
                
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

class ClassifierProcessor():
    """ Using a document to topic matrix, with each document's label added on, this class 
        can classify documents using supervised learning algorithms.  Each document will 
        have a multinomial distribution of topics amongst them.  This distribution of topics 
        will be the attributes to help with the classifcation.
    """

    def __init__(self, doc_to_topic_matrix, unseen_doc_features = [], classifier = 'RF'):
        self.doc_to_topic_matrix = doc_to_topic_matrix
        self.unseen_doc_features = unseen_doc_features
        if classifier == 'DT':
            self.classifier_name = 'Decision Tree'
            self.classifier = DecisionTreeClassifier()
        elif classifier == 'SVM':
            self.classifier_name = 'Support Vector Machine'
            self.classifier = SVC(gamma='auto')
        elif classifier == 'LR':
            self.classifier_name = 'Logistic Regression'
            self.classifier = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=10000)
        elif classifier == 'RF':
            self.classifier_name = 'Random Forest'
            self.classifier = RandomForestClassifier(max_depth=50, n_estimators=100)
        elif classifier == 'GNB':
            self.classifier_name = 'Gaussian Naive Bayes'
            self.classifier = GaussianNB()
        elif classifier == 'MNB':
            self.classifier_name = 'Multinomial Naive Bayes'
            self.classifier = MultinomialNB()
        elif classifier == 'KM':
            self.classifier_name = 'K-Means'
            self.classifier = KMeans(n_clusters=2, random_state=0)
        elif classifier == 'SGD':
            self.classifier_name = 'Stochastic Gradient Descent'
            self.classifier = SGDClassifier(loss = "hinge", penalty = "l2", max_iter = 10)
        elif classifier == 'CNB':
            self.classifier_name = 'Complement Naive Bayes'
            self.classifier = ComplementNB()
        else:
            self.classifier_name = 'ANN - Multilayer Perceptron'
            self.classifier = MLPClassifier(max_iter=1000)
   
    
        
    def train_classifier(self, title):
        X = self.doc_to_topic_matrix.drop('Classification', axis=1)
        y = self.doc_to_topic_matrix['Classification']

        y_pred = cross_val_predict(self.classifier, X, y, cv=10)
        conf_mat = confusion_matrix(y, y_pred)
        
        report = classification_report(y, y_pred)
        print(report)
        report = report.replace('\n','   ')
        report_array = re.split(r'\s{2,}', report)
        
        # print(report_array[5])
        # print(report_array[10])
        # print(report_array[15])
        # print(report_array[20])
        # print(report_array[25])
        # print(report_array[30])
        # print(report_array[35])
        # print(report_array[40])
        # print(report_array[45])
        # print(report_array[50])
        # print(report_array[55])
        # print(report_array[60])
        # print(report_array[65])
        # print(report_array[70])
        # print(report_array[75])
        # print(report_array[80])

        categories = [report_array[5], report_array[10], report_array[15], report_array[20],
                    report_array[25], report_array[30], report_array[35], report_array[40],
                    report_array[45], report_array[50], report_array[55], report_array[60],
                    report_array[65], report_array[70], report_array[75], report_array[80]]
        
        make_confusion_matrix(conf_mat, categories = categories, percent = False, xyplotlabels = True,
                            sum_stats = True, figsize = (30,20), title = title)
        
        # sns.set(font_scale=0.75) # for label size
        # sns.heatmap(conf_mat/np.sum(conf_mat), annot=True, fmt='.2%', cmap='Blues') # font size
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.subplots_adjust(left=0.28, bottom=0.54, right=1.00,
                            top=0.95, wspace=0.20, hspace=0.20)

        plt.show()


    def train_and_test_classifier_k_fold(self, num_folds = 10):
        X = self.doc_to_topic_matrix.drop('Classification', axis=1)
        y = self.doc_to_topic_matrix['Classification']

        scores_lr = cross_val_score(LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=10000), X, y, cv=num_folds)
        scores_svm = cross_val_score(SVC(gamma='auto'), X, y, cv=num_folds)
        scores_rf = cross_val_score(RandomForestClassifier(max_depth=50, n_estimators=100), X, y, cv=num_folds)
        scores_dt = cross_val_score(DecisionTreeClassifier(), X, y, cv=num_folds)
        scores_gnb = cross_val_score(GaussianNB(), X, y, cv=num_folds)
        scores_ann = cross_val_score(MLPClassifier(max_iter=1000), X, y, cv=num_folds)
        

        print("Average score for Logistic Regression:", round(mean(scores_lr),2))
        print("Average score Random Forest:", round(mean(scores_rf),2))
        print("Average score Support Vector Machine:", round(mean(scores_svm),2))
        print("Average score Decision Tree:", round(mean(scores_dt),2))
        print("Average score Gaussian Naive Bayes:", round(mean(scores_gnb),2))
        print("Average score ANN:", round(mean(scores_ann),2))

    def predict_class_for_doc(self):
        print(self.classifier.predict(self.unseen_doc_features))

        
