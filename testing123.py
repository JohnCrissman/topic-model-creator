# testing123.py

import pandas as pd   

# initialize list of lists 
data = [['tom', 10], ['nick', 15], ['juli', 14], ['bob', 24], ['john', 21]] 
  
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['Name', 'Age']) 

print(df)
print(type(df))
# print dataframe. 
print(df.loc[[0]])

for i in range(len(df)):
    print(df.loc[[i]])
    print(i)
df1 = pd.DataFrame()
df1.append(df[0])
print(df1)
# import tensorflow as tf  
# from sklearn import datasets, metrics

# iris = datasets.load_iris()

# feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10, 20, 10], n_classes=3)
# classifier.fit(iris.data, iris.target, steps=2000)

# accuracy_score = classifier.evaluate(x=iris.data,
#                                      y=iris.target)["accuracy"]
# print('Accuracy: {0:f}'.format(accuracy_score))

#hhjhjd


