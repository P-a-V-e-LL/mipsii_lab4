import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

#df = pd.read_csv('./data/IRIS.csv')

#features  = list(df.columns)[:-1]
#target = list(df.columns)[-1:][0]

#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 0)

iris = pd.read_csv("./data/train.csv")
#train = iris.iloc[:, [0, 1, 2, 3]].values

X_train = iris.drop(['species'], axis=1)
y_train = iris['species']

iris1 = pd.read_csv("./data/test.csv")
#test = iris.iloc[:, [0, 1, 2, 3]].values

X_test = iris1.drop(['species'], axis=1)
# y_test = iris1['species']

classifier1 = DecisionTreeClassifier(criterion='entropy')
classifier1.fit(X_train, y_train)
#classifier1.fit(train)

y_pred_1 = classifier1.predict(X_test)

#print(y_pred_1)

f = open("./data/predict.txt", "w+")
f.write(str(y_pred_1))
f.close()

# acc_1 = accuracy_score(y_test,y_pred_1)
# print("Accuracy {} %".format(acc_1*100))
#print(confusion_matrix(y_test, y_pred_1))
#print(classification_report(y_test, y_pred_1))
