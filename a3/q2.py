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

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0


    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        
        v_t = (self.beta * self.vel + grad)
        params  = params - (self.lr * v_t)
        self.vel = v_t
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
        wtx = np.dot( X, self.w)
        wtx_plus_c = np.add(wtx, self.c)
        
        n = X.shape[0]
        l_hinge = np.zeros(n)
        for i in range(n):
            l_hinge[i] = max(y[i]*(1 - wtx_plus_c[i]), 0)
        return l_hinge

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        n, m = X.shape
       
        yt = np.transpose(y)
        
        x_times_y = np.dot(yt, X)
        
        #TODO: is W a single constant, or vector???
        #print "shape x tiems y", x_times_y.shape

        return x_times_y

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        n, m = X.shape
        
        xt = np.transpose(X)
        xtw = np.dot(X, self.w)
        
        y = self.c + xtw
        
        res = np.zeros(n)
        for i in range(n): 
            if y[i] > 0:
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
        w = optimizer.update_params(w, grad)
        w_history.append(w)
    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    
    penalty is penalty = C in the equation
    
    
    
    
    '''
    #sample, each penalty
    
    svm = SVM(penalty, train_data.shape[1])
    w_init = np.sum(svm.w)
    print w_init
    w_history = [w_init]

    for i in range(iters):
        batch_sample = BatchSampler(train_data, train_targets, batchsize)
        batch_train, batch_targets = batch_sample.get_batch() 
        svm_grad = svm.grad(batch_train, batch_targets)
        svm.w -= optimizer.update_params(svm.w, svm_grad)
        w_history.append(np.sum(svm.w))
    return svm

def sign_to_num(res):
    '''
    convert +/-1 into 4 vs 9
    '''
    for i in res:
        if i ==1:
            i = 4
        else:
            i = 9


def accuracy_func(res, targets):
    '''
    simple accuracy calculation 
    '''
    accurate = 0
    for i,j in zip(res, targets):
        if  i==j:
            accurate +=1
    return 1.0 * (accurate)/len(targets)

if __name__ == '__main__':
    
    """
    gd1 = GDOptimizer(1, 0)
    opt_test_1 =  optimize_test_function(gd1)
    gd2 = GDOptimizer(1, 0.9)
    opt_test_2 = optimize_test_function(gd2)
    
    print "=====opt test beta = 0===="
    plt.plot(opt_test_1)
    plt.show()
    print "======opt test beta = 0.9====="
    plt.plot(opt_test_2)
    plt.show()

    """
    gd1 = GDOptimizer(0.05, 0)
    train_data, train_targets, test_data, test_targets = load_data()
    penalty = 1
    res = optimize_svm(train_data, train_targets, penalty, gd1, 100, 500)
    predict = res.classify(test_data)

    #sign_to_num(predict)


    loss = res.hinge_loss(test_data, test_targets)
    print "======= hinge loss ======="
    print loss.sum()
    
    
    print "=======  accuracy ======="
    print accuracy_func(predict, test_targets)

    
    
    