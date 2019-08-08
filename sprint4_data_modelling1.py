# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

from joblib import dump, load

class Data:
    def __init__(self, Xtrain, Xtest, ytrain, ytest):
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest
        
    def getXtrain(self):
        return self.Xtrain
    
    def getYtrain(self):
        return self.ytrain
    
    def getXtest(self):
        return self.Xtest
    
    def getYtest(self):
        return self.ytest

class Modelling:
    def __init__(self, model, data):
        self.model = model.fit(data.getXtrain(), data.getYtrain())
        self.xt = data.getXtest()
        self.yt = data.getYtest()
        
    def evaluate(self):
        return self.model.score(self.xt, self.yt)
        
    def getModel(self):
        return self.model
    
    def saveModel(self, filename):
        dump(self.model, filename)

###=========================================================================###
# Main
print("\n***\tData Modelling\t***\n")

#loaded = np.load('packs.npz')
loaded = np.load('realpacks.npz')

X_train, X_test, y_train, y_test = loaded['a'], loaded['b'], loaded['c'], loaded['d']
#print(X_train.shape)
#print(y_train.shape)

#print(X_test)
#print(y_test)

data = Data(X_train, X_test, y_train, y_test)

# Logistic Regression Model
'''
print("Logistic Classifier Model")
lr = Modelling(LogisticRegression(penalty='l2'), data)
print(lr.evaluate())
'''

# KNeighbors Classifier Model
'''
print("KNeighbors Classifier Model")
knn = Modelling(KNeighborsClassifier(n_neighbors=4), data)
print(knn.evaluate())
'''

# RandomForest Classifier Model
print("RandomForest Classifier Model")
rfc = Modelling(RandomForestClassifier(n_estimators=25, max_depth=4,random_state=1), data)
print(rfc.evaluate())

# AdaBoost Classifier Model
'''
print("AdaBoost Classifier Model")
abc = Modelling(AdaBoostClassifier(n_estimators=25), data)
print(abc.evaluate())
'''

# GradientBoosting Classifier Model
print("GradientBoosting Classifier Model")
gbc = Modelling(GradientBoostingClassifier(learning_rate=0.4, n_estimators=120, max_depth=4), data)
print(gbc.evaluate())

#lr.saveModel('lr.joblib')
#knn.saveModel('knn.joblib')
rfc.saveModel('rfc.joblib')
#abc.saveModel('abc.joblib')
gbc.saveModel('gbc.joblib')

print("***\tSuccess Modelling\t***")

#ngbc = load('gbc.joblib')
#print(ngbc)
###=========================================================================###
'''
if __name__ == "__main__":
    main()
'''