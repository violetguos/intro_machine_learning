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


def var_vector(v1):
    '''
    v1 is ndarray
    return a number var
    '''
    mean = np.mean(v1)
    sum_mean_diff = 0
    for i in range(len(v1)):
        sum_mean_diff += (v1[i] - mean) **2
    var_v1 = (1.0*sum_mean_diff) / (1.0 *len(v1))
    print "var_v1", var_v1, v1
    print "np var v1", np.var(v1)
    return var_v1

def expected_vector(v1):
    prob = 1.0 / (1.0*(len(v1)))
    
    expect_val = sum(v1) * prob  
    print "v1", v1
    print  "prob", prob
    print "exp val", expect_val
    return expect_val
    

def cov_vectors(v1, v2):
    '''
    v1, v2: 2 arrays
    return a number covariance
    '''
    e_v1 = expected_vector(v1)
    e_v2 = expected_vector(v2)
    #v1_min_e_v1 = [i - e_v1 for i in v1]
    #v2_min_e_v2 = [j - e_v2 for j in v2]
    v1_v2_prod = [(k*1.0*n) for k, n in zip(v1, v2) ]
    print "v12rpdo", v1_v2_prod
    e_v1_v2 = expected_vector(v1_v2_prod) - 1.0*e_v1*e_v2
    return e_v1_v2
    

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    
    
    
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
    #means = compute_mean_mles(train_data, train_labels)
    #covariances = compute_sigma_mles(train_data, train_labels)
    # Evaluation
    v1 = [0,3,1,2]
    v2 = [0, 3, -1,0]
    
    X = np.vstack((v1,v2))
    print "np cov", np.cov(X)
    print "np var 1 cov", np.cov(v1)
    bla = cov_vectors(v1, v2)
    print "cpv vec", bla

if __name__ == '__main__':
    main()