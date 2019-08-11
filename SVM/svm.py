import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('~/Downloads/mnist-in-csv/mnist_train.csv')

y = data['label']
target = np.sort(data['label'].unique())
print(target)

#preprocessing the data
X = data.drop(columns = 'label')    
X_scaled = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.20,  random_state = 10)
clf =svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print('Accuracy: ', accuracy_score(pred, y_test))
print(classification_report(pred, y_test))
