# demo china load
import time
import pickle
import pandas as pd

from corpus_processor import CorpusProcessor
from lda_processor import LDAProcessor
from classifier_processor import ClassifierProcessor
from display_notes import DisplayNotes

def main():
    # load vectorizer, doc_to_word_matrix, lda_model, doc_to_topic_matrix, and list_of_barriers
    with open('china_finalized_model.pkl', 'rb') as f:
        vectorizer, doc_to_word_matrix, lda_model, doc_to_topic_matrix, list_of_barriers = pickle.load(f)

    topic_model_creator = LDAProcessor(doc_to_word_matrix, 7, vectorizer, True, lda_model)

    # Create an excel file showing the top n words for each topic
    words = 10
    filename = 'china_topic_to_word_matrix' + str(words) + '.csv'
    topic_to_word_matrix = topic_model_creator.topic_to_word_matrix_n_words(10, filename)

    print("These are the topics for the first document: \n")
    print(lda_model.transform(doc_to_word_matrix)[0])

    print("\nThese are the topics for the second document: \n")
    print(lda_model.transform(doc_to_word_matrix)[1])

    print("\nThese are the topics for the third document: \n")
    print(lda_model.transform(doc_to_word_matrix)[2])

    print(doc_to_topic_matrix)

    classifying_movie_reviews = ClassifierProcessor(doc_to_topic_matrix)
    classifying_movie_reviews.train_classifier()
    # classifying_movie_reviews.predict_class_for_doc() ## I do not have unseen doc features (check demo_processor_load.py)

    # display = DisplayNotes(doc_neg_text, unseen_doc_features, topic_to_word_matrix)

if __name__ == "__main__":
    main()

    