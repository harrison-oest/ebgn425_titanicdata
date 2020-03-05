import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

plt.style.use('ggplot')


# This function groups ranges of ages together so the data is easier to look at
def process_age(df, cut_points, age_type):
    df["Age"] = df["Age"].fillna(-0.5)  # Takes any name that does not have an age and sets it to -0.5 so we can sort
    df["Age Classification"] = pd.cut(df["Age"], cut_points, labels=age_type)
    return df


def process_embarked(train, test):
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    train.Embarked.fillna('S', inplace=True)
    test.Embarked.fillna('S', inplace=True)
    # dummy encoding
    embarked_dummies_train = pd.get_dummies(train['Embarked'], prefix='Embarked')
    embarked_dummies_test = pd.get_dummies(test['Embarked'], prefix='Embarked')
    train = pd.concat([train, embarked_dummies_train], axis=1)
    test = pd.concat([test, embarked_dummies_test], axis=1)
    train.drop('Embarked', axis=1, inplace=True)
    test.drop('Embarked', axis=1, inplace=True)

    return train, test


def create_plots(train):
    # Creates a pivot table and then a bar graph of the comparison between gender and survival rate
    sex_pivot = train.pivot_table(index='Sex', values='Survived')
    sex_pivot.plot.bar()
    plt.show()

    # Creates a pivot table and then a bar graph of the comparison between Pclass and survival rate
    pclass_pivot = train.pivot_table(index='Pclass', values='Survived')
    pclass_pivot.plot.bar()
    plt.show()

    train['Age'].describe()  # Compares the frequency that people of each age survived or did not survive

    survived = train[train["Survived"] == 1]
    not_survive = train[train["Survived"] == 0]
    survived["Age"].plot.hist(alpha=0.5, color='red', bins=50)
    not_survive["Age"].plot.hist(alpha=0.5, color='green', bins=50)
    plt.legend(['Survived', 'Did Not Survive'])
    plt.show()

    cut_points = [0, 18, 60, 100]  # Sets the age ranges
    age_type = ["Child or Teen", "Adult", "Elderly"]

    train = process_age(train, cut_points, age_type)

    pivot = train.pivot_table(index="Age Classification", values='Survived')
    pivot.plot.bar()
    plt.show()


def clean_data(train, test):
    # Convert the Sex column to female and male columns with binary values
    dummies_df = pd.get_dummies(train["Sex"])
    train = pd.concat([train, dummies_df], axis=1)
    train.drop("Sex", axis=1, inplace=True)

    test_dummies_df = pd.get_dummies(test["Sex"])
    test = pd.concat([test, test_dummies_df], axis=1)
    test.drop("Sex", axis=1, inplace=True)

    # Since the Ticket column is not useful for classification, we will drop it from the dataframe
    train = train.drop(['Ticket'], axis=1)
    test = test.drop(['Ticket'], axis=1)

    # Note using the Cabin column so that is dropped as well
    train = train.drop(['Cabin'], axis=1)
    test = test.drop(['Cabin'], axis=1)

    # Figure out who was alone and who had family members with them
    datasets = [train, test]
    for dataset in datasets:
        dataset['family'] = dataset['SibSp'] + dataset['Parch']
        dataset.loc[dataset['family'] > 0, 'alone'] = 0
        dataset.loc[dataset['family'] == 0, 'alone'] = 1
        dataset['alone'] = dataset['alone'].astype(int)

    # The Fare column does not contain the right type for the number, so we need to convert it
    datasets = [train, test]
    for dataset in datasets:
        dataset['Fare'] = dataset['Fare'].fillna(0)
        dataset['Fare'] = dataset['Fare'].astype(int)

    # Divie up the Fare category into more manageable sections
    for dataset in datasets:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare'] = 3
        dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare'] = 4
        dataset.loc[dataset['Fare'] > 250, 'Fare'] = 5
        dataset['Fare'] = dataset['Fare'].astype(int)

    # Change NaN values in the 'Age' column to the average age
    median_age_train = train['Age'].median(skipna=True)
    median_age_test = test['Age'].median(skipna=True)

    train['Age'].fillna(int(median_age_train), inplace=True)
    test['Age'].fillna(int(median_age_test), inplace=True)

    # Split up where they embarked to binary values
    train, test = process_embarked(train, test)

    # Drop the name column since it doesn't matter unless we factor in titles
    train.drop('Name', axis=1, inplace=True)
    test.drop('Name', axis=1, inplace=True)

    # Don't need the passenger IDs so that's removed too
    train.drop('PassengerId', axis=1, inplace=True)

    # Process the Pclass to make it binary
    pclass_dummies_train = pd.get_dummies(train['Pclass'], prefix='Pclass')
    pclass_dummies_test = pd.get_dummies(test['Pclass'], prefix='Pclass')
    train = pd.concat([train, pclass_dummies_train], axis=1)
    test = pd.concat([test, pclass_dummies_test], axis=1)
    train.drop('Pclass', axis=1, inplace=True)
    test.drop('Pclass', axis=1, inplace=True)

    # Simplify the age range into more manageable sections
    datasets = [train, test]
    for dataset in datasets:
        dataset['Age'] = dataset['Age'].astype(int)
        dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
        dataset.loc[dataset['Age'] > 66, 'Age'] = 7

    train.to_csv('train_out.csv', index=False)

    return train, test


def create_models(train, test, start):
    features = ['Age', 'SibSp', 'Parch', 'Fare', 'female', 'male', 'family', 'alone', 'Embarked_C', 'Embarked_Q',
                'Embarked_S', 'Pclass_1', 'Pclass_2', 'Pclass_3']
    x_train = train[features]
    y_train = train["Survived"]
    x_test = test[features]
    x_train = preprocessing.scale(x_train)

    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)

    y_pred = logreg.predict(x_test)

    cross_score_log = cross_val_score(logreg, x_train, y_train, cv=10, scoring="accuracy")

    df_output = pd.DataFrame()
    aux = pd.read_csv('test.csv')
    df_output['PassengerId'] = aux['PassengerId']
    df_output['Survived'] = y_pred
    df_output[['PassengerId', 'Survived']].to_csv('logreg_output.csv', index=False)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    cross_score_knn = cross_val_score(knn, x_train, y_train, cv=10, scoring="accuracy")

    df_output = pd.DataFrame()
    aux = pd.read_csv('test.csv')
    df_output['PassengerId'] = aux['PassengerId']
    df_output['Survived'] = y_pred
    df_output[['PassengerId', 'Survived']].to_csv('knn_output.csv', index=False)

    # Linear SVM
    linear_svc = LinearSVC(dual=False)
    linear_svc.fit(x_train, y_train)

    y_pred = linear_svc.predict(x_test)

    cross_score_svc = cross_val_score(linear_svc, x_train, y_train, cv=10, scoring="accuracy")

    df_output = pd.DataFrame()
    aux = pd.read_csv('test.csv')
    df_output['PassengerId'] = aux['PassengerId']
    df_output['Survived'] = y_pred
    df_output[['PassengerId', 'Survived']].to_csv('linsvm_output.csv', index=False)

    # Random Forest
    random_forest = RandomForestClassifier(max_depth=None, random_state=1, n_estimators=500)
    random_forest.fit(x_train, y_train)

    y_pred = random_forest.predict(x_test)

    cross_score_rf = cross_val_score(random_forest, x_train, y_train, cv=10, scoring="accuracy")

    df_output = pd.DataFrame()
    aux = pd.read_csv('test.csv')
    df_output['PassengerId'] = aux['PassengerId']
    df_output['Survived'] = y_pred
    df_output[['PassengerId', 'Survived']].to_csv('rf_output.csv', index=False)

    # Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(x_train, y_train)
    y_pred = decision_tree.predict(x_test)

    df_output = pd.DataFrame()
    aux = pd.read_csv('test.csv')
    df_output['PassengerId'] = aux['PassengerId']
    df_output['Survived'] = y_pred
    df_output[['PassengerId', 'Survived']].to_csv('tree_output.csv', index=False)

    cross_score_dt = cross_val_score(decision_tree, x_train, y_train, cv=10, scoring="accuracy")

    print("\nCROSS FOLD VALIDATION SCORES:")
    print("The Logistic Regression Model had a score of {:2.2%}".format(cross_score_log.mean()))
    print("The K-Nearest Neighbors Model had a score of {:2.2%}".format(cross_score_knn.mean()))
    print("The Linear Support Vector Model had a score of {:2.2%}".format(cross_score_svc.mean()))
    print("The Random Forest Model had a score of {:2.2%}".format(cross_score_rf.mean()))
    print("The Decision Tree Model had a score of {:2.2%}".format(cross_score_dt.mean()))

    # Figure out how many and what features are important for the model
    # from sklearn.feature_selection import RFECV
    # rfecv = RFECV(estimator=RandomForestClassifier(), step=1, cv=10, scoring='accuracy')
    # rfecv.fit(x_train, y_train)
    #
    # opt_features = train[features].columns[(rfecv.get_support())]
    # print("Optimal number of features: %d" % rfecv.n_features_)
    # print('Selected features: %s' % opt_features)
    #
    # # Plot number of features VS. cross-validation scores
    # plt.figure(figsize=(10, 6))
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (nb of correct classifications)")
    # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    # plt.show()


def run_data(train, test, start):
    plot_bool = input("Would you like to plot the data? (y/n)")
    if plot_bool == 'y' or plot_bool == 'Y':
        create_plots(train)
    train, test = clean_data(train, test)
    create_models(train, test, start)


def main():
    train_df = pd.read_table('train.csv', sep=',')
    train_df.describe()

    test_df = pd.read_table('test.csv', sep=',')
    test_df.describe()

    start_df = pd.read_table('train.csv', sep=',')
    start_df = start_df['Survived']

    run_data(train_df, test_df, start_df)


main()
