'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import json

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))


def mean_i_digit(i_digits):
    
    '''returns the mean for one digit,
        avg across 700 samples for 64 pixels
        i_digit is ndarray
    '''
    i_mean = np.zeros(64)
    i_sum = np.sum(i_digits, axis = 0)
    for i in range(0,64):
        i_mean[i]=i_sum[i]/700.0
    
    #print i_mean
    return i_mean

        

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    train_data: 7000 by 64
    train_labels: 7000
    '''
    means = np.zeros((10, 64))
    
    for i in range(0, 10):
        i_mean_matrix = np.zeros((8,8))
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        # Compute mean of class i
        #TODO: compute 64 by 64 matrix mean??
        #700 row, 64 columns for each difit

        means[i] = mean_i_digit(i_digits) #imean is 64

    return means


def expected_vector(v1):
    len_v1 = 1.0* len(v1)
    
    prob = 1.0 / (len_v1)
    temp_sum = 0
    for i in range(len(v1)):
         temp_sum += v1[i]
    
    expect_val = temp_sum *prob
    return expect_val
    

def cov_vector(v1, v2):
    '''
    calcs sqrt(v1 - mean)/ 2
    return a component of covar
    '''
    
    e_v1 = np.mean(v1)
    e_v2 = np.mean(v2)
    temp_sum = 0
    for i in range(len(v1)):
        #print "vqeoiajsiof", v1[i]
        temp = (v1[i])*(v2[i]) - e_v1*e_v2
        temp_sum +=temp
    
    temp_sum = temp_sum/(1.0 * len(v1) - 1)
    return (temp_sum)
                
                   
    

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    
    test_cov = np.zeros((10, 64, 64))
    for i in range(0, 10):
  
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        #print "idigit", i_digits[:,i].shape #i digits 700 by 64

        #construct 64 by 64
        for ii in range(0, 64):
            for jj in range(0, 64):
                #print "-------------covar----------"
                i_cov_column = cov_vector(i_digits[:,ii], i_digits[:,jj]) 
                #print i_cov_column  
                covariances[i][ii][jj] = i_cov_column
        #test_cov[i] =np.cov(i_digits.T)
    #test_cov = test_cov.reshape(test_cov.shape[0],\
                                      #test_cov.shape[1]*test_cov.shape[2])
    #np.savetxt('h1.txt',test_cov,fmt='%.5f',delimiter=',')    
    

    return covariances

def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    cov_diag_all = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        #print "-------------covdiag------"
        #print cov_diag
        i_mean_matrix = np.reshape(cov_diag, (8,8))
        means.append(i_mean_matrix)
        cov_diag_all.append(cov_diag)
    all_concat = np.concatenate(cov_diag_all,0)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    
    
    return None

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    return None

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    return None

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass



def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    #covariances = covariances.reshape(covariances.shape[0],\
                                      #covariances.shape[1]*covariances.shape[2])
    #np.savetxt('h.txt',covariances,fmt='%.5f',delimiter=',')

    plot_cov_diagonal(covariances)
    
    
if __name__ == '__main__':
    main()