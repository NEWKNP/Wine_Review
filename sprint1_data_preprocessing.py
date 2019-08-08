# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin    # mix transformers in scilit-learn
from sklearn.preprocessing import Imputer
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

def imputer(d):
    d.rename(columns={'Unnamed: 0':'id'}, inplace=True)
    ## clear data
    # Imputing = replace missing value
    nd = DataFrameImputer().fit_transform(d)
    return nd

def splitData(d):
    X = d[['country', 'price', 'province', 'region_1', 'variety']]
    Y = d["points"]
    return X, Y
    
def dataPreprocessing(X, y):
    # Preproccesing: Categorical labels as numbers
    le = LabelEncoder()
    for i in range(0,5):
        X[:, i] = le.fit_transform(X[:, i])
         
    # Re-scale Y-paras
    y = np.ceil(y/10)
    
    normalizer = Normalizer()
    X = normalizer.fit_transform(X)
    return X, y
    
###=========================================================================###
# Main
print("*****\tWelcome to Wine Review\t*****")
print("\n***\tData Preprocessing\t***\n")
    
raw = readFirstDataSet()
#raw = readSecondDataSet()

print("**\tSuccess Reading\t**")

nraw = imputer(raw)
rawX, rawY = splitData(nraw)

print(rawY.min())
print(rawY.max())

X, y = dataPreprocessing(rawX.values, rawY.values)

print(np.amin(y))
print(np.amax(y))
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = None)
    
print("**\tSuccess Preparation\t**")
np.savez_compressed('realpacks', a=X_train, b=X_test, c=y_train, d=y_test)
    
print("**\tSuccess Saving Preparation data\t**")
    
print("***\tSuccess Preprocessing\t***")
###=========================================================================###
    
'''
if __name__ == "__main__":
    main()
'''