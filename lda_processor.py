from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd

class LDAProcessor():

    def __init__(self, doc_to_word_matrix, num_topics, vectorizer):
        self.doc_to_word_matrix = doc_to_word_matrix
        self.num_topics = num_topics
        self.vectorizer = vectorizer
        self.lda_model = self.build_lda_model()
    
    def build_lda_model(self):
        '''Build LDA model.'''
        lda_model = LatentDirichletAllocation(n_components = self.num_topics, # Number of topics
                                      max_iter=1000, # Max learning iterations
                                      learning_method='online',
                                      random_state=100, # Random state
                                      batch_size=1800,#  docs in each learning iter 
                                      n_jobs=-1, # Use all available CPUs
                                      )

        lda_output = lda_model.fit_transform(self.doc_to_word_matrix)
        print(lda_output)
        return lda_model

    def show_topics(self, n_words):
        '''Show top n keywords for each topic.'''
        lda_model = self.lda_model
        keywords = np.array(self.vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    def topic_to_word_matrix_n_words(self, n_words):
        '''Creates a topic - Keywords Dataframe as an excel file.'''
        topic_keywords = self.show_topics(n_words)
        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
        df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
        #print(df_topic_keywords)

        df_topic_keywords.to_csv('movie_reviews_neg_pos_' + str(n_words) + '.csv', encoding='utf-8', index=False)

    def show_topics_for_unseen(self, data_test):
        unseen_document_topics = self.lda_model.transform(self.vectorizer.transform(data_test))[0]
        print(unseen_document_topics)





        