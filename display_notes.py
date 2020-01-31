import webbrowser
import pandas as pd
import numpy as np  
import heapq

class DisplayNotes():
    def __init__(self, notes, unseen_doc_features, topic_to_word_matrix):
        self.unseen_doc_features = unseen_doc_features
        print(type(notes))
        print(len(notes))
        print(type(unseen_doc_features))
        print(len(unseen_doc_features))
        string_notes = notes = " ".join(notes)
        notes = string_notes.split(" ")
        notes1 = ""
        red_words = ["movie", "film", "the"] # highlighted red
        yellow_words = ["the", "on"] # highlighted yellow
        green_words = ["kid", "dad"] # highlighted green
        blue_words = ["that", "get"] # highlighted blue

        print(type(notes))
        print(len(notes))
        print("Finding the greatest value in the unseen doc features:  ")
        print(unseen_doc_features)
        self.notes = notes 
        self.topic_to_word_matrix = topic_to_word_matrix

        for word in notes:
            if(word in yellow_words):
                notes1 = notes1 + '<span class="highlighted-yellow">'+word+'</span>'
            elif(word in green_words):
                notes1 = notes1 + '<span class="highlighted-green">'+word+'</span>'
            elif(word in red_words):
                notes1 = notes1 + '<span class="highlighted-red">'+word+'</span>'
            elif(word in blue_words):
                notes1 = notes1 + '<span class="highlighted-blue">'+word+'</span>'
            else:
                notes1 = notes1 + '<span>'+word+'</span>'
            notes1 = notes1 + " "

        self.notes = notes1
        f = open('helloworld.html','w')

        message = """<!DOCTYPE>
        <html>
        <head>
        <style>
        .highlighted-green{
            background: #66df66;
        }
        .highlighted-yellow{
            background:yellow;
        }
        .highlighted-red{
            background:red;
        }
        .highlighted-blue{
            background:blue;
        }
        </style>
        </head>
        <body><p>""" + self.notes + """</p></body>
        </html>"""


        f.write(message)

        f.close()


        webbrowser.open_new_tab('helloworld.html')
        result_of_function = self.display_doc_n_topics_m_words(4,5)

        print(result_of_function)
        print(result_of_function)

    def display_top_topic(self, topics):
        print("Here is the value for the top topic: ")
        print(np.max(topics))
        print("Here is the index for the top topic: ")
        argmax_top_topic = np.argmax(topics)
        print(argmax_top_topic)
        print("This would be topic number " + str(argmax_top_topic + 1))
        
        print(self.topic_to_word_matrix)
        print()
        return 1


    def display_doc_n_topics_m_words(self,n = 1, m = 5):
        topics = self.unseen_doc_features
        matrix = self.topic_to_word_matrix
        a = self.display_top_topic(self.unseen_doc_features)
        top_topic = np.argmax(topics) # which topic between topic 0 and topic n-1
        print(top_topic)
        print(type(matrix))  # pandas dataframe
        list_of_words_in_topic = matrix.iloc[top_topic,0:m]
        print(list_of_words_in_topic)
        print(type(list_of_words_in_topic))
        print(list_of_words_in_topic.size)
        print(list_of_words_in_topic.tolist())
        print(type(list_of_words_in_topic.tolist()))
        print(len(list_of_words_in_topic.tolist()))
        array = np.array(list_of_words_in_topic)
        print(array)
        print(type(array))
        print(array.shape)
        print(topics)
        print(topics[0])
        indices_of_n_top_topics = heapq.nlargest(n, range(len(topics[0])), topics[0].take) # [3, 6, 0, 5]
        print(indices_of_n_top_topics) # list of indices where the highest value of topics are (sorted from largest to smallest)
        print(type(indices_of_n_top_topics))
        m_words_first_topic = matrix.iloc[indices_of_n_top_topics[0],0:m] # m = 5
        print(m_words_first_topic)
        print(type(m_words_first_topic))
        print(m_words_first_topic.tolist())
        print(type(m_words_first_topic.tolist()))

        list_of_list_of_words = []
        for topic_num in indices_of_n_top_topics:
            top_m_words_in_topic = matrix.iloc[topic_num,0:m].tolist()
            list_of_list_of_words.append(top_m_words_in_topic)

        print(list_of_list_of_words)
        print(type(list_of_list_of_words))





        return list_of_list_of_words
        # display document to show the top n topics and
        # the top m words in each topic
        # Default for n = 2 and default for m = 10


        # Create a method that finds the top n topics (input: n, output: list of numbers that represent topics)
            # will use doc_neg_text (notes private instance variable for this class) and unseen_doc_features from demo_processor_load
        # Then given those topics (probably a list of numbers) get the list of m words. (input: list of numbers, output: shown below)
            # This will probably be a nested list.. because there are n topics.
            # Thus, there will be n elements in the list such that each element is a list of length m.
            # Once I have this list of lists, I should be able to display doc with highlighted sections.
            # Example of output:  [[movie, review, cinema, shoot], [batman, superman, catwoman, penguin], [shark, fish, boat, water]]


    def display_doc_threshold_m_words(self, threshold = 0.2, m = 10):
        return 1
        # Will only display topics that are equal to or over the threshold
        # For example:  if threshold is 0.1, then all topics that make up 10% (0.1)
        #               or more of the document will be displayed.  If none of the topics
        #               make up atleast 10% of the topics then no topics with their
        #               corresponding words will be displayed.
        # The top m words will be shown for each topic that makes it within the threshold.
        # Default for threshold is 0.2 (20 %) and default for m is 10


        # Create a method that finds topics given a threshold (input: threshold, output: list of numbers that represent topics)
            # will use doc_neg_text and unseen_doc_features from demo_processor_load
        # THEN I CAN USE THE METHOD I MADE BEFORE FOR THIS AS WELL!!!!!

        # ONCE I HAVE THE ABOVE METHODS WORKING, HAVE THEM SET UP TO HANDLE UP TO 5 TOPICS.