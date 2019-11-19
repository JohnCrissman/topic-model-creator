import glob
import gensim
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import re

class CorpusProcessor():

    def __init__(self, fileName1, fileName2 = None):
        self.path_neg = fileName1
        self.path_pos = fileName2
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
        '''Tokenize each sentence into a list of words.'''
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
            # deacc = True removes punctuations

    def create_one_doc(self):
        files_neg_test = glob.glob(self.path_neg)
        # files_pos = glob.glob(path_pos)

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
        '''Create a list of strings that has each document in the corpus.'''
        files_neg = glob.glob(self.path_neg)
        files_pos = glob.glob(self.path_pos)
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

    def create_list_of_list_of_words(self):
        '''Each element in the list is a document.
           Each document is represented as a list of words.
           This will be our input to create a document - word matrix.'''
        data = self.create_list_of_docs()

        # Remove Emails
        data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

        # Remove new line characters
        data = [re.sub('\s+', ' ', sent) for sent in data]

        # Remove distracting single quotes
        data = [re.sub("\'", "", sent) for sent in data]

        # Tokenize each sentence into a list of words
        data_words = self.sent_to_words(data)
    
        data_words_list = list(data_words)

        # Do lemmatization keeping only Noun, Adj, Verb, Adverb
        data_lemmatized = self.lemmatization(data_words_list, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        
        return data_lemmatized
    
    def create_vectorizer(self):
        vectorizer = CountVectorizer(#analyzer='word',
                                    min_df = 5,
                                    max_df = 0.9,
                                    stop_words='english', # remove stop words
                                    lowercase=True,  # convert all words to lowercase
                                    token_pattern='[a-zA-Z0-9]{3,}')# num chars >= 3
                                    # max_features = 50000, # max number of unique words
        return vectorizer

    def create_doc_to_word_matrix(self, list_of_list_of_words):
        # Create the Document-Word matrix
        data_vectorized = self.vectorizer.fit_transform(list_of_list_of_words)
        return data_vectorized




