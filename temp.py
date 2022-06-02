# !pip install xgboost
# Load libraries
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np
import csv
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle
##*********************************question 01 program ********************************************************************************
# Load dataset
url_train = "Titanic_train.csv"
url_test = "Titanic_test.csv"
dataset1 = pd.read_csv(url_train)
dataset2 = pd.read_csv(url_test)

dataset1['Sex'] = np.where(dataset1['Sex'] == 'male',1,0)
non_missed_values = dataset1.loc[dataset1['Age'].notnull(),['Age','Survived','Pclass','Sex']]
missing_values = dataset1.loc[dataset1['Age'].isnull(),['Age','Survived','Pclass','Sex']]

X_train = non_missed_values.drop(["Age"], axis=1)
y_train = non_missed_values['Age']
X_train_missing_values = missing_values.drop(["Age"], axis=1)

lm = LinearRegression()
lm.fit(X_train,y_train)
y_train_missing_values_pred = lm.predict(X_train_missing_values)
missing_values['Age'] = y_train_missing_values_pred

dataset1_New = pd.concat([non_missed_values,missing_values])
#Dataset1 is cleaned now and can be further used - called as dataset1_New
dataset1_New['AdultOrChild'] = np.where(dataset1_New['Age'] >= 18,'Adult','Child')
#print(dataset1_New.head())
Survived_New=dataset1_New[dataset1_New['Survived']==1].count()


dataset2['Sex'] = np.where(dataset2['Sex'] == 'male',1,0)
non_missed_values2 = dataset2.loc[dataset2['Age'].notnull(),['Age','Survived','Pclass','Sex']]
missing_values2 = dataset2.loc[dataset2['Age'].isnull(),['Age','Survived','Pclass','Sex']]

X_test2 = missing_values2.drop(["Age"], axis=1)
y_test2 = missing_values2['Age']
y_test2_pred = lm.predict(X_test2)

missing_values2['Age'] = y_test2_pred
#Dataset2 is cleaned now and can be further used - called as dataset2_New
dataset2_New = pd.concat([non_missed_values2,missing_values2])
#print(dataset2_New.head())

#Apply logistic regression from here to get confusion matrix
X_train_New = dataset1_New.drop(["Survived","AdultOrChild"], axis=1)
y_train_New = dataset1_New['Survived']
X_test2_New = dataset2_New[['Age','Pclass','Sex']]
y_test2_New = dataset2_New['Survived']

lgm = LogisticRegression()
lgm.fit(X_train_New,y_train_New)
y_test2_pred_New = lgm.predict(X_test2_New)

cm=confusion_matrix(np.array(y_test2_New),y_test2_pred_New)
print(cm)
Accuracy = ((cm[0,0]+cm[1,1])/np.sum(cm)*100).astype(int)
print('Accuracy from confusion matrix:',Accuracy)
#print('Accuracy from confusion matrix:',(accuracy_score(np.array(y_test2_New),y_test2_pred_New)*100).round(0))

list=[]
list.extend([Survived_New[0],cm.sum(),Accuracy])
with open("output.csv", "w", newline='') as out:
# with open("C:/Users/smd5aq1/Desktop/TCOE/NITW/June_Batch/Assignments05/output.csv", "w", newline='') as out:
    writer = csv.writer(out, delimiter="\n")
    print(list)
    writer.writerow(list)

# 28.500000       3    1
print('X_train_New are',X_train_New.columns)
print('y_train_New are \n',y_train_New)
print('X_test2_New are',X_test2_New.columns)
print('X_test2_New are \n \n',X_test2_New)

print('y_PRED are \n', lgm.predict(X_test2_New))
print('y_PRED are \n', lgm.predict([[28.500000,3,1]]))
print('y_PRED are \n', lgm.predict([[80.000000,1,1]]))
print('y_PRED are \n', lgm.predict([[28.500000,2,0]])[0])


pickle.dump(lgm,open('temp.pkl','wb'))
pickle.dump(lgm,open('temp.sav','wb'))
# pickle.dump(lgm,open('temp.sav','wb'))
# pickle.dump(lgm,open('temp.sav','wb'))