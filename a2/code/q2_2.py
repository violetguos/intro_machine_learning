'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import json


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
                #*this is verified with np cov
                i_cov_column = cov_vector(i_digits[:,ii], i_digits[:,jj])
                
                #print i_cov_column  
                covariances[i][ii][jj] = i_cov_column
            iden_matrix = 0.01*np.identity(64)
            np.add(iden_matrix, covariances[i])
        #test_cov[i] =np.cov(i_digits.T)
    #test_cov = test_cov.reshape(test_cov.shape[0],\
                                      #test_cov.shape[1]*test_cov.shape[2])
    #np.savetxt('h1.txt',test_cov,fmt='%.5f',delimiter=',')    
    

    return covariances

def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    cov_diag_all = []
    for i in range(10):       
        i_mean_matrix = np.zeros((8,8))

        cov_diag = np.diag(covariances[i])
        log_cov_diag = np.log(cov_diag)
        #print "-------------covdiag------"
        #print cov_diag.shape
        i_mean_matrix = np.reshape(log_cov_diag, (8,8))
        cov_diag_all.append(i_mean_matrix)
    all_concat = np.concatenate(cov_diag_all,1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()
    
    

def generative_likelihood_helper(digits, means, covariances):
    n = digits.shape[0]
    p_x = np.zeros((n,10))

    for j in range(0, n):
        for i in range(0, 10):
            x = digits

            pi_term =  pow((2*np.pi), -10/2)
            det_term = np.linalg.det(covariances[i])
            #print "---------eig--------------"
            #print eig_term
            #eig_term = tuple_to_arr(eig_term, 64)
            det_term_root = np.sqrt(det_term) #64 by 64
            #print eig_term_root
            x_diff_miu = np.subtract(x[j], means[i])
            
            #print x[j].shape #64
            #print means[i].shape #64
            x_miu_x_sigmak = np.dot(x_diff_miu.T, np.linalg.inv(covariances[i]) )
            exp_term = (-0.5* np.dot(x_miu_x_sigmak, x_diff_miu)) 
        
            #print "#dot 3 term....."
            #print exp_term
            p_x1 = pi_term /det_term_root
            #print "-----------------------"
            #print "px1 dim", p_x1.shape 
            #print "-----------------------"
            #print "exp term", exp_term
           
            log_p_x1 = np.log(p_x1)    
            p_x[j][i] =  log_p_x1 * exp_term
        #ii += 1

    #p_x = p_x.T
    print p_x
    return p_x
    
    
def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
 
    #print "---------"
    #print p_x
    #print p_x.shape
    n = digits.shape[0]
    log_p_x = generative_likelihood_helper(digits, means, covariances)
    print "-----------p x", log_p_x
    #for i in range(0, n):
    #    for j in range(0, 10):
    #        log_p_x[i][j] = np.log(log_p_x[i][j])
    
    
    return log_p_x

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    n = len(digits)

    p_x_y =generative_likelihood(digits, means, covariances)
    print "-----------p x y", p_x_y

    p_x = 0 # =np.zeros(n) # p(x | sigma, miu)
    
    p_y_x = np.zeros((n, 10))
     
    for i in range(0, n):
        for j in range(0, 10):
            p_x += 0.1 * p_x_y[i][j]
    print "------------------- p x---------------"
    print p_x
    for i in range(0, n):
        for j in range(0, 10):
            p_y_x_ = p_x_y[i][j] + np.log(0.1) - n * np.log(0.1) - np.sum(p_x_y)
            p_y_x[i][j] = (p_y_x_)

    #print p_y_x
    return p_y_x

def avg_conditional_likelihood(digits, labels, means, covariances):
    #(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    
    # Compute as described above and return
    n = len(digits)
    avg_p_y = np.zeros((n, 1))
    
    for i in range(0,n):
        avg_item = np.mean(cond_likelihood[i,:])
        avg_p_y[i] = avg_item
    
    return avg_p_y

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    print "------cond likelihood----- ", cond_likelihood[0]
    n  = digits.shape[0]
    max_class = np.zeros(n)
    # Compute and return the most likely class
    for i in range(digits.shape[0]):
        #go through all n digits, pick the max out of 10
        class_i = cond_likelihood[i,:] #ith row, has 10 digits
        max_class[i] = class_i.argmax() #or is it argmax
    
    #print "-------------class i ", class_i, "  ", class_j
    #print cond_likelihood.shape
    return max_class



def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    #print train_data[0,:].shape
    #test_arr = train_data[101,:]
    print "----------- test_data", test_data.shape[0] #[0] = 4000 #shape = 4000 by 64
    #print "test arr, ", test_arr, " test label ", test_labels[101],
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    
    #print means.shape (64,)
    #print covariances.shape (10, 64, 64)
    
    
    
    #p = conditional_likelihood(train_data[0:2, 0:64], means, covariances)
    #print p
    
    #the final code for classify but need to get everything work now
    accurate_class = 0
    c_predict = classify_data(test_data, means, covariances)

    for i in range(len(test_labels)):
        if c_predict[i] == test_labels[i]:
            accurate_class += 1
    print "-------classify accuracy", (1.0 * accurate_class / len(train_labels))
    
    
    #print np.sqrt(covariances[0][0])
    #eig_term = np.linalg.eig(covariances[0])
    #print "eig value", eig_term
    
    #a = np.arange(16).reshape(4, 4)
    #aa, b = np.linalg.eig(a)
    #print "before", b
    #b = tuple_to_arr(b)
    #print "after", b
    #covariances = covariances.reshape(covariances.shape[0],\
                                      #covariances.shape[1]*covariances.shape[2])
    #np.savetxt('h.txt',covariances,fmt='%.5f',delimiter=',')

    #plot_cov_diagonal(covariances)
    #generative_likelihood((train_data, train_labels), means, covariances)
if __name__ == '__main__':
    main()