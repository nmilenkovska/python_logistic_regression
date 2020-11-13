import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_predict)
print(cm)

accuracy = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print('Accuracy: {:.2f} %'.format(accuracy.mean()*100))
print('Standard Deviation: {:.2f}'.format(accuracy.std()*100))
