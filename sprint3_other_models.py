# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score

from joblib import dump

def randomForestClassifierModel(Xtrain, ytrain):
    rfc = RandomForestClassifier(n_estimators=25, max_depth=4,random_state=1)
    rfc.fit(Xtrain, ytrain)
    return rfc

def adaBoostClassifierModel(Xtrain, ytrain):
    abc = AdaBoostClassifier(n_estimators=25)
    abc.fit(Xtrain, ytrain)
    return abc

def gradientBoostingClassifierModel(Xtrain, ytrain):
    gbc = GradientBoostingClassifier()
    gbc.fit(Xtrain, ytrain)
    return gbc

def accuracy(model, Xtest, Ytest):
    acc = model.score(Xtest, Ytest)
    return acc

def accuracyWithPrediction(model, Xtest, Ytest):
    Ypred = model.predict(Xtest)
    acc = accuracy_score(Ytest, Ypred)
    return acc

def main():
    print("\n***\tData Modelling\t***\n")
    
    loaded = np.load('packs.npz')

    X_train, X_test, y_train, y_test = loaded['a'], loaded['b'], loaded['c'], loaded['d']
    #print(X_train.shape)
    #print(y_train.shape)
    
    #print(X_test)
    #print(y_test)
    
    '''
    # RandomForest Classifier Model
    print("RandomForest Classifier Model")
    rfc1 = randomForestClassifierModel(X_train, y_train)
    acc1 = accuracy(rfc1, X_test, y_test)
    print(acc1)
    '''
    '''
    # AdaBoost Classifier Model
    print("AdaBoost Classifier Model")
    abc1 = adaBoostClassifierModel(X_train, y_train)
    acc1 = accuracy(abc1, X_test, y_test)
    print(acc1)
    '''
    
    # GradientBoosting Classifier Model
    print("GradientBoosting Classifier Model")
    gbc1 = gradientBoostingClassifierModel(X_train, y_train)
    acc1 = accuracy(gbc1, X_test, y_test)
    print(acc1)
    
    
    #dump(..., 'knnmodel3.joblib')
    
    print("***\tSuccess Modelling\t***")

if __name__ == "__main__":
    main()