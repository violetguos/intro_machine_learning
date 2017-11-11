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
            if eta[i][j] > 0.5:
                b_j = 1
            else:
                b_j = 0
            #generated_data[i][j] = pow(eta[i][j], b_j) *\
                                    #pow((1-eta[i][j]),(1 -b_j))
            generated_data[i][j] = b_j
            

    
            
    plot_images(generated_data)




def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    n = bin_digits.shape[0]
    log_p_x = np.zeros((n, 10))
    for i in range(0,n):
        for j in range(0, 10):
            w0c = 0
            for k in range(0,64):
                nkj = (eta[j][k]) **(bin_digits[i][k]) 
                one_min_nkj = (1 -eta[j][k]) **(1 -  bin_digits[i][k]) 
                w0c += (np.log(nkj) + np.log(one_min_nkj))
            log_p_x[i][j] = w0c
            
    #print log_p_x     
    return log_p_x

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    n = bin_digits.shape[0]
    #print "n",n, "  ", bin_digits.shape[1]
    
    p_y_x= generative_likelihood(bin_digits, eta)
    #P(y = c | x , theta) = 0.1 * p(x| y = c)
    '''
    for i in range(0, n):
        for j in range(0, 10):
            w0c = 0
            wcj = 0
            for k in range(0, 64):
                wcj +=  bin_digits[i][k] * np.log((eta[j][k])/(1- eta[j][k]))
                w0c += np.log(1- eta[j][k])
    '''
    bc = np.log(0.1)
    #print "-------in cond likelihood"
    #print "-----before add"
    #print p_y_x
    p_y_x += bc
    #print p_y_x
    return p_y_x

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)

    
    # Compute as described above and return
    n = len(bin_digits)
    p_y = 0
    avg_p_y = 0
    
    for i in range(0,n):
        avg_item = (cond_likelihood[i,:])
        #cond_label = avg_item.argmin() #most probable, prediction
        cond_label = labels[i]
        p_y += cond_likelihood[i][int(cond_label)]
    
    avg_p_y  = p_y / n
        
    print "-------------in avg cond likelihood--------"
    print avg_p_y
    return avg_p_y

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    n = bin_digits.shape[0]
    new_points = np.zeros(n)
    for i in range(0, n):
        #print cond_likelihood[i]
        test = cond_likelihood[i]
        new_points[i] =np.argmax(test)
    
    
    return new_points

def classify_accuracy(predict_label, real_label):
    n = real_label.shape[0]
    accurate_class = 0
    for i in range(0,n):
        if predict_label[i] == real_label[i]:
            accurate_class += 1
    print "-------classify accuracy", (1.0 * accurate_class / n)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    #Q2=------new images------
    # Evaluation
    print "===========Q2.3========="
    print "eta image"
    plot_images(eta)

    print "new sample image"
    generate_new_data(eta)
    
    
    print "---------Q 2.3.2---------"
    train_predict =  classify_data(train_data, eta)
    test_predict = classify_data(test_data, eta)
    
    print "---------avg likelihood----------"
    avg_train = avg_conditional_likelihood(train_data, train_labels, eta)
    avg_test = avg_conditional_likelihood(test_data, test_labels, eta)
    
    print "---------Q 2.3.6 Predication accuracy----"
    train_acc = classify_accuracy(train_predict, train_labels)
    print "train accuracy", train_acc
    test_acc = classify_accuracy(test_predict, test_labels)
    print "test accuracy", test_acc

if __name__ == '__main__':
    main()
