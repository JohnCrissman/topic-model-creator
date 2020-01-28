import webbrowser
import pandas as pd

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

        print(type(notes))
        print(len(notes))
        self.notes = notes
        self.topic_to_word_matrix = topic_to_word_matrix
        
        for word in notes:
            if(word in yellow_words):
                notes1 = notes1 + '<span class="highlighted-yellow">'+word+'</span>'
            elif(word in green_words):
                notes1 = notes1 + '<span class="highlighted-green">'+word+'</span>'
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
        .highlighted-green{
            background: #66df66;
        }
        .highlighted-yellow{
            background:yellow;
        }
        .highlighted-red{
            background:red;
        }
        </style>
        </head>
        <body><p>""" + self.notes + """</p></body>
        </html>"""


        f.write(message)

        f.close()

        webbrowser.open_new_tab('helloworld.html')

    def display_doc_n_topics_m_words(self, n = 2, m = 10):
        return 1
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