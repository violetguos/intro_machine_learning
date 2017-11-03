'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        #train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    
    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        labels = [float(i) for i in range(0,10)]
        distances = np.array(self.l2_distance(test_point))
        #distances = np.asarray(distances[0]) 
        #print "distace", distances
         #indices of the k smallest in distance
        k_idx = np.array((distances.argsort()[:k]))
        #k_idx = np.asarray(k_idx[0]) #conver to a ndarray
        #print "k idx", k_idx
        #build a hash table of label/digit to counts
        #a list [] of tuple(label, count)
        label_count = np.zeros(10)
        #index is the label, number is # of instance in k Neighbours
        for j in k_idx:
            for i in range(len(labels)):
                #print "train label j", self.train_labels[j], j
                if self.train_labels[j] == labels[i]:
                    label_count[i] +=1
        
        #print "label count", label_count
                
        #if label_count.max() > k//2: #randomly pick the frist occuranace
        max_label_idx = label_count.argmax()
        #print "mac label", max_label_idx
        digit = float(max_label_idx)

        return digit

def cross_validation(knn, k_range=np.arange(1,15)):
    all_k = []
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        kf = KFold(n_splits=10)
        k_train_accuracy = []
        for train_index, test_index in kf.split(knn.train_data):
            x_train, x_test = knn.train_data[train_index], knn.train_data[test_index]
            y_train, y_test = knn.train_labels[train_index], knn.train_labels[test_index]
            knn_new = KNearestNeighbor(x_train, y_train)
            k_train_accuracy.append(classification_accuracy(knn_new ,k, x_test, y_test))
        k_accuracy = (1.0 *sum(k_train_accuracy)) / (1.0 *len(k_train_accuracy))   
        all_k.append(k_accuracy)
    print all_k
    all_k = np.array(all_k)
    optimal_k = all_k.argmax()
    return optimal_k
    

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
  
    knn_labels = [] 
    for col in eval_data:
        #col is 64 vector
        knn_labels.append((knn.query_knn(col, k)))
        
    
    #print "knn_labels clasojsaires", type(knn_labels)
    cnt_total = len(eval_labels)
    cnt_accurate = 0
    for j in range(len(eval_labels)):
        if eval_labels[j] == knn_labels[j]:
            cnt_accurate +=1
            
    return float(cnt_accurate) / float(cnt_total)
    
        

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    #print "lenlnelne",len(train_labels)  = 7000, labels are floats
    knn = KNearestNeighbor(train_data, train_labels)


    #===========Q1,2--------#
    #k_1_accuracy = classification_accuracy(knn, 1, test_data, test_labels)
    #k_15_accuracy = classification_accuracy(knn, 15, test_data, test_labels)
    #print "k 1", k_1_accuracy
    
    #print "k 15.", k_15_accuracy
    
    #-----------------Q3---------------#
    #opti_k_index = cross_validation(knn)
    #k1_accuracy is the highest
    k_1_test_accuracy = classification_accuracy(knn, 1, test_data, test_labels)
    k_1_train_accuracy = classification_accuracy(knn, 1, train_data, train_labels)
    print k_1_test_accuracy
    print k_1_train_accuracy
    
if __name__ == '__main__':
    main()