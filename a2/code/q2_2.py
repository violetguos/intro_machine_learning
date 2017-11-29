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
    
    
    
    
def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
 
    n = digits.shape[0]
    p_x = np.zeros((n,10))

    for i in range(0, n):
        for j in range(0, 10):
            x = digits

            pi_term = (2* np.pi) #-10/2

            x_diff_miu = np.subtract(x[i], means[j])

            inv_term = np.linalg.inv(covariances[j])
            det_term = np.linalg.det(covariances[j])
      
            x_miu_x_sigmak = np.dot(np.transpose(x_diff_miu), inv_term) #MATMUL
            #print x_miu_x_sigmak
      
            exp_term = np.dot(x_miu_x_sigmak, x_diff_miu)

            p_x[i][j] = -(64 / 2) * np.log(pi_term)\
                        -0.5*np.log(det_term)\
                        -0.5*(exp_term)
                        
                        #np.log(inv_det_root)
                        #+ np.log(0.1)
            
    #p_x = p_x.T
    #print "----------in generative helper"
    #print np.exp(p_x)
    return p_x



def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    n = len(digits)

    log_pxy_gen =generative_likelihood(digits, means, covariances)
    
    for i in range(0, n):
             #print "=============in cond likelihood"
             #print log_pxy_gen[i].shape
             p_x_y_=  np.exp(log_pxy_gen[i])
          		          
             p_x_y_ = p_x_y_ * 0.1 #verfied dim 10
   #print "=============in cond likelihood pxy shape"
  #print p_x_y_.shape
       #print "=============in cond likelihood"
       #print "p_x_y", p_x_y_
  
    p_x_y_sum = np.sum(p_x_y_)

    log_pyx_cond = log_pxy_gen + np.log(0.1)- np.log(p_x_y_sum)
     
   
    return log_pyx_cond

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
    p_y = 0
    avg_p_y = 0
    
    for i in range(0,n):
        #cond_label = avg_item.argmin() #most probable, prediction
        cond_label = labels[i]
        p_y += cond_likelihood[i][int(cond_label)]
    
    avg_p_y = p_y / n
        
    print "-------------in avg cond likelihood--------"
    print avg_p_y
    return avg_p_y

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    cond_exp = np.exp(cond_likelihood)
    #print "------cond likelihood----- ", cond_likelihood[0]
    n  = digits.shape[0]
    max_class = []
    # Compute and return the most likely class
    for class_i in cond_exp:
        #go through all n digits, pick the max out of 10
         #= cond_exp[i,:] #ith row, has 10 digits
        max_class.append(np.argmax(class_i)) #or is it argmax

    return max_class

def classify_accuracy(predict_label, real_label, n):
    accurate_class = 0
    for i in range(0,n):
        if predict_label[i] == real_label[i]:
            accurate_class += 1
    print "-------classify accuracy", (1.0 * accurate_class / n)
    

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    print "============Q2.2 Part1 plot of log of Sigma_k diagonal"
    plot_cov_diagonal(covariances)
    
    print "============Q2.2 part2 average log likelihood========"
    print "===========Train data average log likelihood========="
    avg_train = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    
    print "===========Test data average log likelihood ========"
    avg_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    
    
    
    #the final code for classify but need to get everything work now
    print "=============Q2.2 part3 prediction and accuracy of each predication======"
    print "=============Train data prediction and accuracy========"
    train_predict =  classify_data(train_data, means, covariances)
    n_dim_train = train_labels.shape[0]
    classify_accuracy(train_predict, train_labels, n_dim_train)
    
    print "=============Test data prediction and accuracy========="
    test_predict = classify_data(test_data, means, covariances)
    n_dim_test = test_labels.shape[0]
    classify_accuracy(test_predict, test_labels,n_dim_test )

   
    
    
   
if __name__ == '__main__':
    main()