# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
np.random.seed(0)
from scipy.misc import logsumexp


# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions-y_test)**2).mean()
    print("losses")
    return losses
 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    
    ## TODO
    a_ii_denom_arr = [] #store each j
    #a_ii_denom_sum = np.matrix([]) #store dist matrix N by d
    a_ii_denom = [] #store the log sum over j
    a_ii = [] #the Array that stores each exp over exp sum
    print("test datum", np.shape(test_datum))
    print("x_train", np.shape(x_train))
    
    x_x_dist = l2(np.transpose(test_datum), x_train) #N_train by d matrix
    
    rows = x_x_dist.shape[0] 
    cols = x_x_dist.shape[1] #304
    
    print("row", rows)
    print("cols", cols)
    
    #sum over the column 
    #for i in range (0, rows):
       # a_ii_denom_arr = np.array([]) #re init for each i
    for j in range(0, cols):
        #append all the column values
        a_ii_denom_arr.append(- x_x_dist[0][j]/ (2 * tau**2))
    
    a_ii_denom_arr = np.array(a_ii_denom_arr)
    
    print (type(a_ii_denom_arr)) #1 by 304
    
        #add the column values to the matrix
    print ("aii arr", np.shape(a_ii_denom_arr))
    
        #sum each row in aii_denom_sum
    a_ii_denom_log = logsumexp(a_ii_denom_arr)
    a_ii_denom.append(np.exp(a_ii_denom_log))
    
    a_ii_denom = np.array(a_ii_denom)
    
    for j in range(0, cols):
            a_ii_nom = np.exp(- x_x_dist[0][j]/ (2 * tau**2))
            a_ii.append(a_ii_nom / a_ii_denom)
    
    a_ii = np.array(a_ii)
    #print (a_ii)
    print ("aii dim", np.shape(a_ii))
    Aii = np.diagflat(a_ii) #A must be N by N, 304 by 304
    print ("Aii dim", np.shape(Aii))

    
    
    
    #w∗ = XTAX+λI −1XTAy, (xtax + I)w = Xtay
    lam_i = lam * np.identity(len(x_train[1])) #lambda times I
    print("lami", np.shape(lam_i))

    #x transpose * a* x + lambda I
    #compute x_t times a first
    xta = np.dot(np.transpose(x_train), Aii)
    xtax_i = np.dot(xta, x_train) + lam_i
    #x transpose times A * Y
    xtay = np.dot(xta, y_train )
    
    w = np.linalg.solve(xtax_i, xtay)
    
    y_hat = np.dot(w, test_datum)
    
    return y_hat


def partition_k(x, y, num, i):
    '''
    returns x_train, x_test, y_train, y_test
    '''
    #print("part x, ",np.shape(x))
    #print("part y, ", np.shape(y))
    
    #print ("num = ", num)
    #print ("i=", i)
    x_test = x[(i*num):((i+1)*num):,] #select the test bit
    #print ("x_test", np.shape(x_test))
    A= x[0:(i*num),:]
    #print("len a", len(A))
    B = x[((i+1)*num): ,: ]   
    #print ("B shape", np.shape(B))
    
    if len(A) ==0:
        x_train = B
    else:
        x_train = np.concatenate((A, B), axis =0) #select the rest 304x14
    #print (np.shape(x_train))   
    y_test = y[(i*num):((i+1)*num)] #select elems from array
    y_train  = np.concatenate([y[0:(i)*num], y[(i+1)*num:]]) #select the rest 304
    #print(len(y_test))
    #print("y len = ", len(y_train))
    
    return x_test, y_test, x_train, y_train

def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO
    N = len(y)
    num_per_fold = N//k #floor division
    losses =[]
    for i in range(0, k):
        x_test, y_test,x_train, y_train = partition_k(x,y,num_per_fold, i)
        losses.append(run_on_fold(x_test, y_test, x_train, y_train, taus))
    
    return np.array(losses)
    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)

    plt.plot(losses)
    print("min loss = {}".format(losses.min()))

    #min loss = 63.9109756781