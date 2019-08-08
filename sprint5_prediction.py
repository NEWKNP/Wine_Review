# -*- coding: utf-8 -*-

import random

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from joblib import load   

def sampleData():
    df = pd.read_pickle("clean_data")
    magic_number = random.randint(0,df.shape[0])
    d = df.iloc[magic_number, :]
    X = d[['country', 'price', 'province', 'region_1', 'variety']]
    Y = d["points"]
    return X, Y, d[['title']]

def dataPreprocessing(X, y):
    # Preproccesing: Categorical labels as numbers
    le = LabelEncoder()
    for i in range(0,5):
        #print(np.array([X[i]]))
        X[i] = le.fit_transform(np.array([X[i]]))
         
    # Re-scale Y-paras
    y = np.ceil(y/10)
    #print(X)
    #print(y)
    normalizer = Normalizer()
    X = normalizer.fit_transform(np.array([X]))
    return X, y
###=========================================================================###
# Main
print("\n***\tData Prediction\t***\n")

sX, sY, name = sampleData()
nX, nY = dataPreprocessing(sX.values, np.array([sY]))

# Load Model
rfc = load('randomForest_classifier.joblib')
gbc = load('gradientBoosting_classifier.joblib')

# Predict Data
'''
print("RandomForest Classifier")
ypred1 = rfc.predict(nX)
print("Predict Score: ", ypred1[0])
print("Actual Score: ", nY[0])
'''
print("Model: GradientBoosting Classifier")
print("Wines name: ", name.values[0])
ypred2 = gbc.predict(nX)
print("Predict Score: ", ypred2[0])
print("Actual Score: ", nY[0])

print("\n***\tSuccess\t***\n")
###=========================================================================###