import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error


def find_missing_features(df):
    num_missing = df.isnull().sum()
    percent = num_missing / df.isnull().count()
    df_missing = pd.concat([num_missing, percent], axis=1, keys=['MissingValues', 'Fraction'])
    df_missing = df_missing.sort_values('Fraction', ascending=False)
    print(df_missing[df_missing['MissingValues'] > 0])
    #keeping the variables with no missing information
    variables_to_keep = df_missing[df_missing['MissingValues'] == 0].index
    df_train = df[variables_to_keep]
    return df_train

def correlation_matrix(matrix):
    f, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(matrix, vmax=0.7, square=True)
    plt.show()

def select_important_variables(matrix):
    important_variables = matrix['SalePrice'].sort_values(ascending=False)
    important_variables = important_variables[abs(important_variables) >= 0.6]
    important_variables = important_variables[important_variables.index != 'SalePrice']
    print(important_variables)
    #OverallQual is the best feature
    return important_variables
def analysis(df):
    #Analysing the trend
    values = np.sort(df['OverallQual'].unique())
    print('Unique values of "OverallQual":', values)
    sns.regplot(x='OverallQual', y = 'SalePrice', data = df)
    plt.show()

def prediction(model,X_test,y_test):
    y_pred = model.predict(X_test)
    # Build a plot
    plt.scatter(y_pred, y_test)
    plt.xlabel('Prediction')
    plt.ylabel('Real value')
    # Now add the perfect prediction line
    diagonal = np.linspace(0, np.max(y_test), 100)
    plt.plot(diagonal, diagonal, '-r')
    plt.show()
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    print('RMSE:\t%.5f' % mse)
def main():
    print("Assignment3- Regression\n")
    df = pd.read_csv('train.csv')
    df_train = find_missing_features(df)
    matrix = df_train.corr()
    correlation_matrix(matrix)
    imp_var = select_important_variables(matrix)
    analysis(df_train)
    #print(imp_var.index)
    #print(df_train[imp_var.index])
    X = df_train[imp_var.index]
    y = df_train['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    prediction(rfc, X_test, y_test)
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    prediction(lr,X_test,y_test)

if __name__ == '__main__':
    main()
