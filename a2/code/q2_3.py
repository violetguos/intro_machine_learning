'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    #make a hash table list, i is label, nc is total count
    eta = np.zeros((10, 64))
    nc = np.zeros((10, 64))
    #for each class k, count the 1st pixel in 700 vectors that is one
    #add the beta distribution
    for i in range(0, 10):
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        for j in range(0, 700):
            for k in range(0, 64):
                if i_digits[j][k] == 1:
                    nc[i][k] +=1
    #calculate beta(2,2)
    for i in range(0, 10):
        for j in range(0, 64):
            eta[i][j] =  1.0*(nc[i][j] + 2 -1) / (700 +2 +2+ -2)
        
    #print "nc_list", (nc)
    #print eta
    
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    img_matrix = []
    for i in range(10):
        img_i = class_images[i]
        i_matrix = np.zeros((8,8))
        i_matrix = np.reshape(img_i, (8,8))
        img_matrix.append(i_matrix)
    all_concat = np.concatenate(img_matrix,1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()
    
def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))

    for i in range(0, 10):
        for j in range(0, 64):
            if eta[i][j] < 0.5:
                b_j = 1
            else:
                b_j = 0
            generated_data[i][j] = pow(eta[i][j], b_j) *\
                                    pow((1-eta[i][j]),(1 -b_j))
            
    plot_images(generated_data)




def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    log_p_x = np.zeros((64, 10))
    for i in range(0,10):
        i_digit = data.get_digits_by_label(bin_digits[0], bin_digits[1], i)
        for j in range(0, 64):
            p_x = pow(eta[i][j], i_digit[i, j]) *\
                                    pow((1-eta[i][j]),i_digit[i, j])
            log_p_x[j][i] = np.log(p_x)
        
    return log_p_x

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    p_x_y = generative_likelihood(bin_digits, eta)
    
    
    
    return None

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    # Compute as described above and return
    return None

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    # Compute and return the most likely class
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)

    generate_new_data(eta)

if __name__ == '__main__':
    main()
