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


######################################################
# This is a program that performs k fold validatinon #
# and Locally reweighted least squares               #
######################################################



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
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()

    print("Running on 1 of k folds")
    return losses
 
 
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

    
    x_x_dist = l2(np.transpose(test_datum), x_train) #N_train by d matrix
    #print ("x x dist. ", x_x_dist.shape())
    rows = x_x_dist.shape[0] #1
    cols = x_x_dist.shape[1] #304
    
    #print("row", rows)
    #print("cols", cols)
    
    #sum over the column 

    for j in range(0, cols):
        #append all the column values
        a_ii_denom_arr.append(- x_x_dist[0][j]/ (2 * tau**2))
    
    a_ii_denom_arr = np.array(a_ii_denom_arr)
    
    a_ii_denom_log = logsumexp(a_ii_denom_arr)
    a_ii_denom.append(np.exp(a_ii_denom_log))
    
    a_ii_denom = np.array(a_ii_denom)
    
    for j in range(0, cols):
        a_ii_nom = np.exp(- x_x_dist[0][j]/ (2 * tau**2))
        a_ii.append(a_ii_nom / a_ii_denom)
    
    a_ii = np.array(a_ii)
    Aii = np.diagflat(a_ii) #A must be N by N
  
    #w∗ = XTAX+λI −1XTAy, (xtax + I)w = Xtay
    lam_i = lam * np.identity(len(x_train[1])) #lambda times I
    #print("lami", np.shape(lam_i))

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

    x_test = x[(i*num):((i+1)*num):,] #select the test bit
    A= x[0:(i*num),:]
    B = x[((i+1)*num): ,: ]   
    
    if len(A) ==0:
        x_train = B
    else:
        x_train = np.concatenate((A, B), axis =0) #select the rest 304x14
    y_test = y[(i*num):((i+1)*num)] #select elems from array
    y_train  = np.concatenate([y[0:(i)*num], y[(i+1)*num:]]) #select the rest 304
    
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
        per_losses = run_on_fold(x_test, y_test, x_train, y_train, taus)
        losses.append(per_losses)
        
    
    return np.array(losses)


#def average_ith(losses):
#    rows = len(losses[1]) #should be 5
#    cols = len(losses[0])
#    sum_list = []
#    for i in range(0, cols):
#        sum1 = 0
#        for j in range(0, rows):
#            sum1 +=losses[i][j]
#        sum1 = sum1/rows
#        sum_list.append(sum1)
#    
#    return np.array(sum_list)
    

def average_loss_per_tau(losses):
    '''
    Average loss of a given tau value
    '''
    avg_list = []
    for i in range(len(losses[0])):
        sum_tau = 0
        for j in range(0, 5):
            sum_tau += losses[j][i]
        avg = sum_tau / float(5)
        avg_list.append(avg)
    return avg_list

if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value.
    # Feel free to play with lambda as well if you wish
    print ("--------Loading and Computing--------------")
    taus = np.logspace(1,3,400)
    losses = run_k_fold(x,y,taus,k=5)

    
    for i in range(0,5):
        plt.plot(taus, losses[i])
    
    plt.ylabel("losses")
    plt.xlabel("taus")
    plt.show()
    
    loss_avg = average_loss_per_tau(losses)
    plt.plot(taus, loss_avg)
    plt.show()

    print("min loss = {}".format(np.array(loss_avg).min()))
