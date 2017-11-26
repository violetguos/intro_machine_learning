import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        
        v_t = (self.beta * params + grad)
        params  += (self.lr * v_t)
        return params

class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        wt = np.transpose(self.w)
        wtx = np.dot(wt, X)
        wtx_plus_c = np.add(wtx, self.c)
        
        n = X.shape[0]
        l_hinge = np.zeros(n)
        for i in range(n):
            l_hinge[i] = max((1 - wtx_pluc_c[i]), 0)
        return l_hinge

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        n, m = X.shape
        w_update = []
        w_sum = []
        for i in range(n):
            w_sum_ = np.sum(X[i])
            w_sum.append(w_sum_)
        for i in range(n):
            dj_dw = self.w[i] -  w_sum[i]
            
            
        
        return dj_dw

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        n, m = X.shape
        
        xt = np.transpose(X)
        xtw = np.dot(xt, self.w)
        
        y = self.c + xtw
        
        res = np.zeros(n)
        for i in range(n): 
            if y > 0:
                res[i] = 1
            else: 
                res[i] = -1
        
        return res

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]
    
    for i in range(steps):
        # Optimize and update the history
        grad = func_grad(w)
        w -= optimizer.update_params(w, grad)
        w_history.append(w)
    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''
    #sample, each penalty
    
    for i in range(iters):
        batch_sample = BatchSampler(train_data, train_targets, batchsize)
        batch_train, batch_targets = batch_sample.get_batch()
        svm = SVM(penalty[i], train_data.shape[0])
        svm.grad(batch_train, batch_targets)
        res = svm.classify(batch_train)
    return res

if __name__ == '__main__':
    """
    gd1 = GDOptimizer(1,0)
    opt_test_1 =  optimize_test_function(gd1)
    gd2 = GDOptimizer(1, 0.9)
    opt_test_2 = optimize_test_function(gd2)
    
    print "=====opt test beta = 0===="
    x = np.linspace(0,4, 201)
    plt.plot(x, opt_test_1)
    plt.show()
    print "======opt test beta = 0.9====="
    plt.plot(x, opt_test_2)
    plt.show()
    """
    train_data, train_targets, test_data, test_targets = load_data()
    penalty = np.array((1,10))
    res = optimize_svm(train_data, train_targets, penalty, GDOptimizer, 10, 2)
    print res
    