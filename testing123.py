# testing123.py
import tensorflow as tf  
from sklearn import datasets, metrics

iris = datasets.load_iris()

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10, 20, 10], n_classes=3)
classifier.fit(iris.data, iris.target, steps=2000)

accuracy_score = classifier.evaluate(x=iris.data,
                                     y=iris.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))



