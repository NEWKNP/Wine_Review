# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin    # mix transformers in scilit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder #For text value
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """
        Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def readFirstDataSet():
    temp_data = pd.read_csv("winemag-data_first150k.csv")
    return temp_data

def readSecondDataSet():
    temp_data = pd.read_csv("winemag-data-130k-v2.csv")
    return temp_data

def dataPreprocessing(d):
    d.rename(columns={'Unnamed: 0':'id'}, inplace=True)
    # clear data
    # Imputing = replace missing value
    d = DataFrameImputer().fit_transform(d)
    
    print(d.info())
    
    # Create training and test set
    X = d.loc[:, d.columns != 'points'].values
    Y = d["points"].values
    
    # Preproccesing: Categorical labels as numbers
    le = LabelEncoder()
    X[:, 0] = le.fit_transform(X[:, 0])
    X[:, 1] = le.fit_transform(X[:, 1])
    X[:, 2] = le.fit_transform(X[:, 2])
    X[:, 3] = le.fit_transform(X[:, 3])
    X[:, 4] = le.fit_transform(X[:, 4])
    X[:, 5] = le.fit_transform(X[:, 5])
    X[:, 6] = le.fit_transform(X[:, 6])
    X[:, 7] = le.fit_transform(X[:, 7])
    X[:, 8] = le.fit_transform(X[:, 8])
    X[:, 9] = le.fit_transform(X[:, 9])
    Y = le.fit_transform(Y)
    
    normalizer = Normalizer()
    X = normalizer.fit_transform(X)
    return X,Y
    
def main():
    print("*****\tWelcome to Wine Review\t*****")
    print("\n***\tData Preprocessing\t***\n")
    
    raw = readFirstDataSet()
    #raw = readSecondDataSet()
    #print(raw.head())
    #print(raw.info())
    
    X, Y = dataPreprocessing(raw)
    
    #print(X[0:5])
    #print(Y[0:5])
    
    print("\n***\tData Modelling\t***\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    
    print("\n***\tData Evaluation\t***\n")

if __name__ == "__main__":
    main()