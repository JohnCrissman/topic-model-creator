# testing simple functions
# '''
# classifications = ['good', 'bad']
# n = 5
# list_of_classifications = [item for item in classifications for i in range(n)]
# print(list_of_classifications)
# '''

# classifications = ['bad', 'good']
# n = 999
# list_of_classifications = [item for item in classifications for i in range(n)]
# print(list_of_classifications)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

corpus = [
    'This is the first document.',
    'This is the second second document',
    'And the third one.',
    'Is this the first document?',
]

X = vectorizer.fit_transform(corpus)

analyze = vectorizer.build_analyzer()
print(analyze("This is a text document to analyze.") == (
        ['this', 'is', 'text', 'document', 'to', 'analyze']))

print(X.toarray())