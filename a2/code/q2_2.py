'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from scipy import stats


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
    

def cov_vector(v1):
    '''
    calcs sqrt(v1 - mean)**2 / 2
    return a component of covar
    '''
    
    e_v1 = expected_vector(v1)
    
    temp_sum = 0
    for i in range(len(v1)):
        temp = (v1[i]- e_v1)**2
        temp_sum +=temp
    
    temp_sum = temp_sum/2.0
    return np.sqrt(temp_sum)
                
                   
    

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    for i in range(0, 1):
        i_cov_column = np.zeros(64)
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        #print "cov dig digit", cov_diag_vector(i_digits[:,i]) 
        #print "idigit", i_digits
        for j in range(0, 64):
            i_cov_column[j] = cov_diag_vector(i_digits[:,i]) 
            
        #construct 64 by 64
        for ii in i_cov_column:
            for jj in i_
        covariances[i] = i_cov_column
    
    
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        # ...

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
    # Evaluation
    
if __name__ == '__main__':
    main()