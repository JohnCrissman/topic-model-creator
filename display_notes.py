import webbrowser
import pandas as pd
import numpy as np  
import heapq

class DisplayNotes():
    """ This class displays one document and highlights words different
        colors that are associated with topics.
        
        Maximum ammount of topics that can be highlighted is 5.  This is ok
        because most of the document will be associated with either 1 or 2 topics.

        Two options:
            1. Highlight the top m words from the top n topics.  Each word from
                different topics will be highlighed with different colors.
            2. Highlight the top m words from the topics that meet a specific
                threshold.  Words from different topics will be highlighed differently.
    """

    def __init__(self, notes, unseen_doc_features, topic_to_word_matrix):

        self.unseen_doc_features = unseen_doc_features
        self.topic_to_word_matrix = topic_to_word_matrix
        self.notes = notes 

    def highlight_words_associated_with_topics(self, lists_of_words_from_topics):

        string_notes = self.notes = " ".join(self.notes)
        notes = string_notes.split(" ")
        notes1 = ""

        blue_words, green_words, yellow_words, orange_words, red_words = [],[],[],[],[]
        if len(lists_of_words_from_topics) >= 1:
            blue_words = lists_of_words_from_topics[0]
        if len(lists_of_words_from_topics) >= 2:
            green_words = lists_of_words_from_topics[1]
        if len(lists_of_words_from_topics) >= 3:
            yellow_words = lists_of_words_from_topics[2]
        if len(lists_of_words_from_topics) >= 4:
            orange_words = lists_of_words_from_topics[3]
        if len(lists_of_words_from_topics) >= 5:
            red_words = lists_of_words_from_topics[4]
        
        for word in notes:
            if(word in blue_words):
                notes1 = notes1 + '<span class="highlighted-blue">'+word+'</span>'
            elif(word in green_words):
                notes1 = notes1 + '<span class="highlighted-green">'+word+'</span>'
            elif(word in yellow_words):
                notes1 = notes1 + '<span class="highlighted-yellow">'+word+'</span>'
            elif(word in orange_words):
                notes1 = notes1 + '<span class="highlighted-orange">'+word+'</span>'
            elif(word in red_words):
                notes1 = notes1 + '<span class="highlighted-red">'+word+'</span>'
            else:
                notes1 = notes1 + '<span>'+word+'</span>'
            notes1 = notes1 + " "

        self.notes = notes1
        f = open('helloworld.html','w')

        message = """<!DOCTYPE>
        <html>
        <head>
        <style>
        .highlighted-blue{
            background: #98c9d4;
        }
        .highlighted-green{
            background: #bbd48d;
        }
        .highlighted-yellow{
            background: #f0ec97;
        }
        .highlighted-orange{
            background: #f6cd69;
        }
        .highlighted-red{
            background: #f09fc8;
        }
        </style>
        </head>
        <body><p>""" + self.notes + """</p></body>
        </html>"""

        f.write(message)
        f.close()
        webbrowser.open_new_tab('helloworld.html')


    def display_threshold_topics_m_words(self, topic_threshold, m_words):

        lists_of_words_from_topics = self.display_doc_threshold_m_words(threshold= topic_threshold, num_words= m_words)  # showing top topics over threshold
        self.highlight_words_associated_with_topics(lists_of_words_from_topics)


    def display_top_n_topics_m_words(self, n_topics, m_words):

        lists_of_words_from_topics = self.display_doc_n_topics_m_words(num_topics= n_topics, num_words= m_words)   # showing top n topics
        self.highlight_words_associated_with_topics(lists_of_words_from_topics)
        

    def display_doc_n_topics_m_words(self,num_topics = 1, num_words = 5):
        # Input: number of topics and number of words as natural numbers
        # Output: A list of lists.  Each element in the list is a list representing the top m words from that topic.
        #         The first element is the top topic, the second element is the 2nd topic, and so on up to n topics.

        topics = self.unseen_doc_features
        matrix = self.topic_to_word_matrix
        indices_of_n_top_topics = heapq.nlargest(num_topics, range(len(topics[0])), topics[0].take) # [3, 6, 0, 5]
        list_of_list_of_words = []
        for topic_num in indices_of_n_top_topics:
            top_m_words_in_topic = matrix.iloc[topic_num,0:num_words].tolist()
            list_of_list_of_words.append(top_m_words_in_topic)
        return list_of_list_of_words

        
    def display_doc_threshold_m_words(self, threshold = 0.2, num_words = 10):
        topics = self.unseen_doc_features[0]
        matrix = self.topic_to_word_matrix
        indices_over_threshold = [idx for idx, val in enumerate(topics) if val > threshold]
        list_of_list_of_words = []
        for topic_num in indices_over_threshold:
            top_m_words_in_topic = matrix.iloc[topic_num,0:num_words].tolist()
            list_of_list_of_words.append(top_m_words_in_topic)
        return list_of_list_of_words


        # Will only display topics that are equal to or over the threshold
        # For example:  if threshold is 0.1, then all topics that make up 10% (0.1)
        #               or more of the document will be displayed.  If none of the topics
        #               make up atleast 10% of the topics then no topics with their
        #               corresponding words will be displayed.
        # The top m words will be shown for each topic that makes it within the threshold.
        # Default for threshold is 0.2 (20 %) and default for m is 10


        