'''
Question 2.0 Skeleton Code

Here you should load the data and plot
the means for each of the digit classes.
'''

import data
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.pyplot.imshow

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



def plot_means(train_data, train_labels):
    means = []
    for i in range(0, 10):
        i_mean_matrix = np.zeros((8,8))
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        # Compute mean of class i
        #TODO: compute 64 by 64 matrix mean??
        #700 row, 64 columns for each difit

        i_mean = mean_i_digit(i_digits) #imean is 64
        #tes_list =i_mean.tolist()
        #print len(tes_list)
        i_mean_matrix = np.reshape(i_mean, (8,8))
        means.append(i_mean_matrix)
        #print means
    # Plot all means on same axis
    all_concat = np.concatenate(means,1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

if __name__ == '__main__':
    train_data, train_labels, _, _ = data.load_all_data_from_zip('a2digits.zip', 'data')
    plot_means(train_data, train_labels)
