# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

from joblib import dump

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
print("\n***\tData Modelling (SVM)\t***\n")

loaded = np.load('realpacks.npz')

X_train, X_test, y_train, y_test = loaded['a'], loaded['b'], loaded['c'], loaded['d']
#print(X_train.shape)
#print(y_train.shape)

#print(X_test)
#print(y_test)

data = Data(X_train, X_test, y_train, y_test)

# Support Vector Machines Model
svmm = Modelling(svm.SVC(kernel='rbf', gamma=10, C=1000), data)
print(svmm.evaluate())

svmm.saveModel('svmm.joblib')

print("***\tSuccess Modelling\t***")
###=========================================================================###
'''
if __name__ == "__main__":
    main()
'''