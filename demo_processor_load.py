import time
import pickle
import pandas as pd

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor
from classifier_processor import ClassifierProcessor

from sklearn.model_selection import train_test_split

# CLASSIFIERS
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn import svm


from sklearn.metrics import classification_report, confusion_matrix


# main file to run the program

def main():
    # load vectorizer, doc to word matrix, and topic model
    with open('c_finalized_model.pkl', 'rb') as f:
        vectorizer, doc_to_word_matrix, lda_model, doc_to_topic_matrix = pickle.load(f)

    topic_model_creator = LDAProcessor(doc_to_word_matrix, 7, vectorizer, True, lda_model)



    
    # Create an excel file showing the top 15 words for each topic
    topic_model_creator.topic_to_word_matrix_n_words(15)

    '''Predict topics for a new piece of text'''
    path_neg_test = 'C:/Users/johnm/Documents/Tutoring/CLASSES/MASTERS_PROJECT/code/txt_sentoken/neg_test/*.txt'
    #path_pos_test = 'C:/Users/johnm/Documents/Tutoring/CLASSES/MASTERS_PROJECT/code/txt_sentoken/pos_test/*.txt'

    doc_neg = CorpusProcessor(path_neg_test)
    doc_neg_text = doc_neg.create_one_doc()
    print("testing unseen doc")
    print(doc_neg_text)
    print("testing finished")

    ''' input used for predicting classification (good or bad) for unseen document!!!!!!!!!!!!'''
    unseen_doc_features = topic_model_creator.show_topics_for_unseen(doc_neg_text).reshape(1, -1)
    print("\nThese are the unseen document features: \n")
    print(unseen_doc_features)
    print("\n")

    print("These are the topics for the first document: \n")
    print(lda_model.transform(doc_to_word_matrix)[0])

    print("\nThese are the topics for the second document: \n")
    print(lda_model.transform(doc_to_word_matrix)[1])

    print("\nThese are the topics for the third document: \n")
    print(lda_model.transform(doc_to_word_matrix)[2])

    # returns Document to Topic matrix and creates a csv file of the matrix
    #####doc_topic_matrix = topic_model_creator.create_doc_to_topic_matrix()
   
    #print(doc_to_topic_matrix)

    '''
    classification of 0 means bad
    classification of 1 means good
    '''
    # classifications = ['0', '1']
    # n = 999
    # list_of_classifications = [item for item in classifications for i in range(n)]
    # #print(list_of_classifications)

    # doc_to_topic_matrix['Classification'] = list_of_classifications

    print(doc_to_topic_matrix)

    # #doc_to_topic_matrix.to_csv('doc_to_topic_w_classifier.csv', encoding='utf-8', index=False)
    
    # print(doc_to_topic_matrix.shape)

    # print(doc_to_topic_matrix.head())

    # X = doc_to_topic_matrix.drop('Classification', axis=1)
    # y = doc_to_topic_matrix['Classification']

    # print(X)
    # print(y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # #############
    # # Testing different types of classifiers here:  results for 20 topics!!
    # #############
    # '''Decision Tree: ~57% accuracy'''
    # # classifier = DecisionTreeClassifier()

    # '''Multi-Layer-Perceptron: ~60% accuracy'''
    # classifier = MLPClassifier(solver='lbfgs', alpha = 1e-5, 
    #                             hidden_layer_sizes=(5,2), random_state =1)

    # '''Gaussian Naive Bayes: ~52% accuracy'''
    # # classifier = GaussianNB()

    # '''Multinomial Naive Bayes: ~50% accuracy'''
    # # classifier = MultinomialNB()

    # '''Nearest Neighbors (Nearest Centroid): 
    #     20 topics:  ~50% accuracy'''
    # # classifier = NearestCentroid()

    # '''Stochastic Gradient Descent Classifier: ~60% accuracy'''
    # # classifier = SGDClassifier(loss = "hinge", penalty = "l2", max_iter = 5)

    # '''Support Vector Machines (SVM)'''
    # # classifier = svm.SVC()
    
    

    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    # # SHOWING THE PREDICTION (GOOD OR BAD) OF THE UNSEEN DOCUMENT
    # print("Here is the prediction: ")
    # print(classifier.predict(unseen_doc_features))

    # print("looking at the tests: ")
    # print(classifier.predict(X_test))

    classifying_movie_reviews = ClassifierProcessor(doc_to_topic_matrix, unseen_doc_features)
    classifying_movie_reviews.train_classifier()
    classifying_movie_reviews.predict_class_for_doc()





    







main()


