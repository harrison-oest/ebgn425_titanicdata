# -*- coding: utf-8 -*-
"""TitanicData.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_BbfvSH-bupeDUT9iFqNL1VEV0-dQQMc
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

global titanic_df
titanic_df = pd.read_table('train.csv', sep=',')
titanic_df.describe()

global test_df
test_df = pd.read_table('test.csv', sep=',')
test_df.describe()

# --------------------------------------------------
# Creates a pivot table and then a bar graph of the comparison between gender and survival rate
sex_pivot = titanic_df.pivot_table(index='Sex', values='Survived')
sex_pivot.plot.bar()
# plt.show()

# --------------------------------------------------
# Creates a pivot table and then a bar graph of the comparison between Pclass and survival rate
pclass_pivot = titanic_df.pivot_table(index='Pclass', values='Survived')
pclass_pivot.plot.bar()
# plt.show()

# --------------------------------------------------
titanic_df['Age'].describe()  # Compares the frequency that people of each age survived or did not survive

survived = titanic_df[titanic_df["Survived"] == 1]
didNotSurvive = titanic_df[titanic_df["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5, color='red', bins=50)
didNotSurvive["Age"].plot.hist(alpha=0.5, color='green', bins=50)
plt.legend(['Survived', 'Did Not Survive'])
# plt.show()

# --------------------------------------------------
# This function groups ranges of ages together so the data is easier to look at
# def process_age(df,cutPoints,ageType):
#     df["Age"] = df["Age"].fillna(-0.5) #Takes any name that does not have an age and sets it to -0.5 so we can sort
#     df["Age Classification"] = pd.cut(df["Age"],cutPoints,labels=ageType)
#     #df["Age Classification"] = pd.Categorical(titanic_df["Age Classification"], ["Child or Teen","Adult","Elderly"])
#     #df["Age Classification"] = df.sort("Age Classification")
#     return df

# cutPoints = [0,18,60,100] #Sets the age ranges
# ageType = ["Child or Teen","Adult","Elderly"]

# titanic_df = process_age(titanic_df,cutPoints,ageType)

# pivot = titanic_df.pivot_table(index="Age Classification",values='Survived')
# pivot.plot.bar()
# plt.show()

"""To make the classifications easier, we need to clean up the data:"""

# Convert the Sex column to female and male columns with binary values
dummies_df = pd.get_dummies(titanic_df["Sex"])
titanic_df = pd.concat([titanic_df, dummies_df], axis=1)
titanic_df.drop("Sex", axis=1, inplace=True)

test_dummies_df = pd.get_dummies(test_df["Sex"])
test_df = pd.concat([test_df, test_dummies_df], axis=1)
test_df.drop("Sex", axis=1, inplace=True)

# Since the Ticket column is not useful for classification, we will drop it from the dataframe
titanic_df = titanic_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)

titanic_df = titanic_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)

# The Fare column does not contain the right type for the number, so we need to convert it

datasets = [titanic_df, test_df]

for dataset in datasets:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

# Change NaN values in the 'Age' column to the average age
mean_age_train = titanic_df['Age'].mean(skipna=True)
mean_age_test = test_df['Age'].mean(skipna=True)

titanic_df['Age'].fillna(int(mean_age_train), inplace=True)
test_df['Age'].fillna(int(mean_age_test), inplace=True)

# Similar to the 'Sex' column, we need to assign numeric values in the 'Embarked' column


def process_embarked():
    global titanic_df
    global test_df
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    titanic_df.Embarked.fillna('S', inplace=True)
    test_df.Embarked.fillna('S', inplace=True)
    # dummy encoding 
    embarked_dummies_train = pd.get_dummies(titanic_df['Embarked'], prefix='Embarked')
    embarked_dummies_test = pd.get_dummies(test_df['Embarked'], prefix='Embarked')
    titanic_df = pd.concat([titanic_df, embarked_dummies_train], axis=1)
    test_df = pd.concat([test_df, embarked_dummies_test], axis=1)
    titanic_df.drop('Embarked', axis=1, inplace=True)
    test_df.drop('Embarked', axis=1, inplace=True)
    return titanic_df, test_df


titanic_df, test_df = process_embarked()

titanic_df.drop('Name', axis=1, inplace=True)
test_df.drop('Name', axis=1, inplace=True)

"""K-Nearest Neighbors ML"""

inputs = ['Pclass', 'Age', 'female']
target = ['Survived']

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

X_train = titanic_df.drop("Survived", axis=1)
X_train = preprocessing.scale(X_train)
Y_train = titanic_df["Survived"]
X_test = test_df.copy()

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

df_output = pd.DataFrame()
aux = pd.read_csv('test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = Y_pred
df_output[['PassengerId', 'Survived']].to_csv('output.csv', index=False)

start_df = pd.read_table('train.csv', sep=',')
start_df = start_df['Survived']
df_output = df_output['Survived']

iter = 0
total = 0

for i in range(len(start_df)):
    if start_df.get(i, 'Survived') == df_output.get(i, 'Survived'):
        total += 1

print("\nThe Logistic Regresssion model predicted {:2.2%}".format(total/start_df.sum()),"cases correctly!")

# KNN
X_train = titanic_df.drop("Survived", axis=1)
X_train = preprocessing.scale(X_train)
Y_train = titanic_df["Survived"]
X_test = test_df.copy()

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

df_output = pd.DataFrame()
aux = pd.read_csv('test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = Y_pred
df_output[['PassengerId', 'Survived']].to_csv('output.csv', index=False)

start_df = pd.read_table('train.csv', sep=',')
start_df = start_df['Survived']
df_output = df_output['Survived']

iter = 0
total = 0

for i in range(len(start_df)):
    if start_df.get(i, 'Survived') == df_output.get(i, 'Survived'):
        total += 1

print("\nThe K-Nearest Neighbors model predicted {:2.2%}".format(total/start_df.sum()),"cases correctly!")



