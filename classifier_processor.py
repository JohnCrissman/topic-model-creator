# Classifier Processor
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

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
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import numpy 

class ClassifierProcessor():
    """ Using a document to topic matrix, with each document's label added on, this class 
        can classify documents using supervised learning algorithms.  Each document will 
        have a multinomial distribution of topics amongst them.  This distribution of topics 
        will be the attributes to help with the classifcation.
    """

    def __init__(self, doc_to_topic_matrix, unseen_doc_features = [], classifier = 'ANN'):
        self.doc_to_topic_matrix = doc_to_topic_matrix
        self.unseen_doc_features = unseen_doc_features
        if classifier == 'DT':
            self.classifier_name = 'Decision Tree'
            self.classifier = DecisionTreeClassifier()
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
            self.classifier = MLPClassifier(solver='lbfgs', alpha = 1e-5, 
                                hidden_layer_sizes=(10,4), random_state =1)

    

    def train_classifier(self):
        X = self.doc_to_topic_matrix.drop('Classification', axis=1)
        y = pd.factorize( self.doc_to_topic_matrix['Classification'] )[0]
        
        # y = self.doc_to_topic_matrix['Classification']
        print(y)
        print(numpy.unique(y).size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        # self.classifier.fit(X_train, y_train)

        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

        classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[100, 200, 100], n_classes=18)
        classifier.fit(X_train, y_train, steps=2000)

        accuracy_score = classifier.evaluate(x=X,
                                     y=y)["accuracy"]
        print('Accuracy: {0:f}'.format(accuracy_score))


        

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

        
