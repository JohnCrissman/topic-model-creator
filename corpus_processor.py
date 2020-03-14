import glob
import gensim
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import re

class CorpusProcessor():
    """ This class prepares a collection of documents to use a 
        vectorizer in order to make a document to word matrix
    """

    def __init__(self, fileName1 = None, fileName2 = None):
        """ Initializes instance variables:
            path_1 = path name to get all the txt files in some directory
            path_2 = path name to get all the txt files in another directory
            vectorizer = return a CountVectorizer with specific parameters
                            shown in method below.
        """
        self.path_1 = fileName1
        self.path_2 = fileName2
        self.vectorizer = self.create_vectorizer()

    def get_vectorizer(self):
        return self.vectorizer

    def remove_punctuation(self, line):
        '''Remove punctuation from a line of text.'''
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        newLine = ""
        for char in line:
            if char not in punctuations:
                newLine = newLine + char
            elif char in punctuations:
                newLine = newLine + " "
        return newLine

    def lemmatization(self, texts, allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        nlp = spacy.load('en', disable=['parser', 'ner'])
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
        return texts_out

    def sent_to_words(self, sentences):
        """Tokenize each sentence into a list of words."""
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
            # deacc = True removes punctuations

    def create_one_doc(self):
        files_neg_test = glob.glob(self.path_1)
        # files_pos = glob.glob(path_2)

        data_test = []
        one_document = ""
        text = ""

        for name in files_neg_test:
            try:
                with open(name, 'r', encoding = 'utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        text = self.remove_punctuation(line.strip())
                        one_document = one_document + text
                    data_test.append(one_document)
                    
            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise

        f.close()
        return data_test

    def create_list_of_docs(self):
        """Create a list of strings such that each string is a document in the corpus."""

        files_neg = glob.glob(self.path_1)
        files_pos = glob.glob(self.path_2)
        data = []
        one_document = ""
        text = ""
        for name in files_neg:
            try:
                with open(name, 'r', encoding = 'utf-8') as f:
                    
                    lines = f.readlines()
                    for line in lines:
                        text = self.remove_punctuation(line.strip())
                        one_document = one_document + text
                    data.append(one_document)
                    one_document=""
                    
            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise
        f.close()
        for name in files_pos:
            try:
                with open(name, 'r', encoding = 'utf-8') as f:
                    
                    lines = f.readlines()
                    for line in lines:
                        text = self.remove_punctuation(line.strip())
                        one_document = one_document + text
                    data.append(one_document)
                    one_document=""
                    
            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise
        f.close()

        return data

    def create_list_of_list_of_words(self, list_of_docs = []):
        '''Each element in the list is a document.
           Each document is represented as a list of words.
           This will be our input to create a document - word matrix.'''
        if(len(list_of_docs) is 0):
            data = self.create_list_of_docs()
        else:
            data = list_of_docs
        # Remove Emails
        data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

        # Remove new line characters
        data = [re.sub('\s+', ' ', sent) for sent in data]

        # Remove distracting single quotes
        data = [re.sub("\'", "", sent) for sent in data]

        """ data is a list of documents such that each document is a string.  (list of strings)"""
        # Tokenize each sentence into a list of words
        data_words = self.sent_to_words(data)
    
        """ data_words is an iterator.  Thus we need to convert it to a list like we did below."""
        
        data_words_list = list(data_words)

        # Do lemmatization keeping only Noun, Adj, Verb, Adverb
        data_lemmatized = self.lemmatization(data_words_list, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        return data
    
    def create_vectorizer(self):
        vectorizer = CountVectorizer(analyzer='word',
                                    min_df = 5,
                                    max_df = 0.9,
                                    #stop_words='english', # remove stop words
                                    lowercase=True,
                                    stop_words='english')  # convert all words to lowercase
                                    # token_pattern='[a-zA-Z0-9]{1,}')# num chars >= 1
                                    # max_features = 50000, # max number of unique words
        return vectorizer

    def create_doc_to_word_matrix(self, list_of_list_of_words):
        # Create the Document-Word matrix
        data_vectorized = self.vectorizer.fit_transform(list_of_list_of_words)
        return data_vectorized

    




