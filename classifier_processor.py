# Classifier Processor
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import seaborn as sn

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

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


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

    
   

    def train_classifier(self):
        X = self.doc_to_topic_matrix.drop('Classification', axis=1)
        y = self.doc_to_topic_matrix['Classification']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
        # plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

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

        
