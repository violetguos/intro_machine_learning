#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:16:31 2017

@author: vikuo
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import random


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target #y is the price
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1] #13 eatures
    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.plot(X[:,i], y, '.')
        plt.ylabel("Price")
        plt.xlabel(features[i])
    plt.tight_layout()
    plt.show()

def split_data_8020(X, Y):
    #select columns from x, y
    trainingSet, testSet = [], []
    chosenSamples = random.sample(range(len(Y)),
                                  len(Y)//5)
    #need whole numbers to divide, use the operator //
    for i in range(len(Y)):
        #implement insert xy sample tuple for now
        if i not in chosenSamples:
            trainingSet.append((X[i:,], Y[i]))#ROW of X
        elif i in chosenSamples:
            testSet.append((X[i:,], Y[i])) 
    return trainingSet, testSet


def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    #raise NotImplementedError()
    w = []
    w_col = np.linalg.solve(np.transpose(X), np.transpose(Y))
    w.append(w_col)
    return w

def main():
    # Load the data
    X, y, features = load_data()
    
    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)
    
    #TODO: Split data into train and test
    
    # Fit regression model
    w = fit_regression(X, y)

    # Compute fitted values, MSE, etc.


if __name__ == "__main__":
    main()

