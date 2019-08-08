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

from joblib import dump, load   

def sampleData():
    df = pd.read_pickle("clean_data")
    magic_number = random.randint(0,df.shape[0])
    d = df.iloc[magic_number, :]
    X = d[['country', 'price', 'province', 'region_1', 'variety']]
    Y = d["points"]
    return X, Y

def dataPreprocessing(X, y):
    # Preproccesing: Categorical labels as numbers
    le = LabelEncoder()
    for i in range(0,5):
         X[i] = le.fit_transform(X[i])
         
    # Re-scale Y-paras
    y = np.ceil(y/10)
    
    normalizer = Normalizer()
    X = normalizer.fit_transform(X)
    return X, y

class Evaluation:
    def __init__(self, model):
        self.model = model
        self.clf = 0
        self.params = []
    
    def setTunedParameters(self, p):
        self.params = p
        
    def evaluate(self, X, y):
        self.clf = GridSearchCV(self.model, self.params, cv=5)
        self.clf.fit(X, y)
    
    def predict(self, X, y):
        Ypred = self.clf.predict(X)
        return Ypred, self.clf.score(X, y) ,accuracy_score(y, Ypred)
    
    def saveModel(self, filename):
        dump(self.model, filename)

###=========================================================================###
# Main
print("\n***\tData Evaluation\t***\n")

loaded = np.load('realpacks.npz')
X_train, X_test, y_train, y_test = loaded['a'], loaded['b'], loaded['c'], loaded['d']

#knn = load('knn.joblib')
rfc = load('rfc.joblib')
ngbc = load('gbc.joblib')
#svmm = load('svmm.joblib')

print(rfc)
print(ngbc)

# Parametersearch
print("RandomForest Classifier Evaluation")
prs1 = Evaluation(rfc)
n_estimators = np.array([10, 20, 25, 40, 50])
max_depth = np.array([3, 4, 5])
random_state = np.array([0, 1])
params = [{'n_estimators': n_estimators}, {'max_depth': max_depth}, {'random_state': random_state}]
prs1.setTunedParameters(params)
prs1.evaluate(X_train, y_train)
scores1, acc1, compare1 = prs1.predict(X_test, y_test)
print("Accuracy: ", acc1)
print("Comparasion: ", compare1)
print(y_test[0:15])
print(scores1[0:15])

print("GradientBoosting Classifier Evaluation")
prs2 = Evaluation(ngbc)
n_estimators = np.array([120, 130, 140])
max_depth = np.array([6, 8])
learning_rate = np.array([0.4])
params = [{'n_estimators': n_estimators}]
prs2.setTunedParameters(params)
prs2.evaluate(X_train, y_train)
scores2, acc2, compare2 = prs2.predict(X_test, y_test)
print("Accuracy: ", acc2)
print("Comparasion: ", compare2)
print(y_test[0:15])
print(scores2[0:15])

# another source
loaded2 = np.load('realpacks2.npz')

print("Another Source")
X_test2, y_test2 = loaded2['b'], loaded2['d']
scores3, acc3, compare3 = prs2.predict(X_test2, y_test2)
print("Accuracy: ", acc3)
print("Comparasion: ", compare3)
print(y_test2[0:15])
print(scores3[0:15])

# Save model
prs1.saveModel('randomForest_classifier.joblib')
prs2.saveModel('gradientBoosting_classifier.joblib')
###=========================================================================###
'''
if __name__ == "__main__":
    main()
'''