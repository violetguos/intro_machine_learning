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
    #506 row, 13 columns
    #print X.shape[0]
    y = boston.target #y is the price
    #print y.shape[0]
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
    xrows = X.shape[1]

    chosenSamples = random.sample(range(len(Y)),
                                  len(Y)//5)
    t_len = len(Y) - len(chosenSamples)
    sample_len = len(chosenSamples)
    trainingSetX = np.zeros((t_len, xrows))
    testSetX = np.zeros((sample_len, xrows) )
    trainingSetY = np.zeros(t_len)
    testSetY = np.zeros(sample_len)
    ii, ij = 0,0
    #need whole numbers to divide, use the operator //
    for i in range(len(Y)):
        #implement insert xy sample tuple for now
        if i not in chosenSamples:
            #what = X[i,]
            #print "wnat", what
            trainingSetX[ii,] = X[i,]
            #print "tslit, train X, ", len(trainingSetX)
            trainingSetY[ii]= Y[i]#ROW of X
            #print "thwaraw " ,X[i,]
            ii +=1
            
        elif i in chosenSamples:
            testSetX[ij,]=X[i,]
            testSetY[ij]=Y[i]
            ij +=1
            
    #print trainingSetX #.shape[0], testSetX.shape[0], trainingSetY, testSetY

    return trainingSetX, testSetX,\
             trainingSetY, testSetY

def tabulate_weight(w, x):
    for i in range(len(x)):
        for a, b in zip(w,x[i]):
            #print "{}\t{}".format(repr(a),repr(b))
            c =1 #dummy
    plt.show()

def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    xtx = np.dot(np.transpose(X), X)
    xty = np.dot(np.transpose(X), Y)
    w = np.linalg.solve(xtx, xty)
    #print type(w)
    tabulate_weight(w, X)
    return w #w_1




def main():
    # Load the data
    X, y, features = load_data()
    xrows = X.shape[1]

    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)
    
    #TODO: Split data into train and test
    X = np.concatenate((np.ones((506,1)),X),axis=1) #add constant one feature - no bias needed

    # Fit regression model
    
    training_x, testing_x, training_y, testing_y  = split_data_8020(X, y)
    
    #print "train x ", training_x.shape[1] #shape 0 is 1, shape 1 is 405
    #print "train y ", training_y.shape[0] #shape 0 is 405
    #print "test x ", test_x.shape[1] #shape 0 is 1, shape 1 is 101
    #print "test y ", test_y.shape[0] #shape 0 is 101
    w = fit_regression(training_x, training_y)
    #print w

    # Compute fitted values, MSE, etc.
    
    y_hat = np.dot(testing_x, w)
    #print "y_hat ", y_hat
    #print "y ", y
    
    #MSE
    mse = ((y_hat - testing_y) **2).mean(axis = 0)
    print mse
    
    #another two error measures: 
        #mean norm, mean root
    mnorm = np.absolute(y_hat - testing_y)
    #print mnorm
    
    #feacture selection
    print "-----feature ranking----"
    for i in range(len(X[1])):
        y_hat = np.dot(X[:,i], w[i]) #this is 506 instances
        #print y_hat
        #print len(X[:,i])
        mse = ((y_hat - y[i]) **2).mean()
        print "%s, %f ", features[i], mse
        

if __name__ == "__main__":
    main()

