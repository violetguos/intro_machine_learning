import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import math
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

    def __init__(self, lr, beta=0.0, vel = 0.0):
        self.lr = lr
        self.beta = beta
        self.vel = vel


    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        #print "val", self.vel
   
        v_t_plus = self.beta * self.vel
        #print "vt", v_t_plus
        v_t_plus = v_t_plus + grad     
        params  = params - (self.lr * v_t_plus)
        self.vel = v_t_plus
        return params

class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        self.b = 0
        #self.feature_count = feature_count
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        #print self.w.shape (784,)
        #print X.shape (100, 784)
        #print y.shape (100)
        
        wt = np.transpose(self.w)
        #print self.w
        wtx = np.dot(X, self.w)
        wtx_plus_c = wtx #shape (100,)
        #print wtx_plus_c.shape
        n = X.shape[0]
        l_hinge = np.zeros(n)
        for i in range(n):
            l_hinge[i] = max(y[i]*(1- wtx_plus_c[i]), 0)
        return l_hinge

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        n, m = X.shape
        xt= np.transpose(X)        
        sum_vec = np.dot(xt, y)
    
        
        grad = self.w
        
        c_over_n = self.c * 1.0 / n
        c_over_n_arr = [c_over_n * 1.0] * m
        c_over_n_arr[0] = 1 #grad[0]
        c_over_n_arr = np.asarray(c_over_n_arr)
        
        reg_vec = np.multiply(sum_vec, c_over_n_arr)
        
        grad = grad - reg_vec #/ n
        
        return (grad)
    
    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        #w is shape m

        n, m = X.shape #(784, 2757)
        
        xt = np.transpose(X)
        #print xt
        xtw = np.dot(X, self.w)
        #print("######self b ", self.b)
        y = xtw +self.b
        #print "y_ classify ", y
        #print y[0]
        res = np.zeros(n)
        for i in range(n):             
            if y[i] > 0:
                res[i] = 1.0
            else: 
                res[i] = -1.0        
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
    n, m = train_data.shape
    
    svm = SVM(penalty, m)
    w_init = np.sum(svm.w)
    #print w_init
    batch_sampler = BatchSampler(train_data, train_targets, batchsize)


    for i in range(iters):
        grad_estimate = 0

        for j in range(int(n/batchsize)):
            batch_train, batch_targets = batch_sampler.get_batch() 
            svm_grad = svm.grad(batch_train, batch_targets)
            grad_estimate += (optimizer.update_params(svm.w, svm_grad))# + hinge_loss #+ h_loss))
        svm.w = grad_estimate/(n/batchsize)
    svm.b = -penalty * 1.0/m * np.sum(train_targets)#(batch_targets)
        
    return svm

def plot_w(w):
    i_mean_matrix = np.reshape(w, (28,28))
    plt.imshow(i_mean_matrix, cmap='gray')
    plt.show()
    #plt.plot(w)
    #plt.show()
    
def accuracy_func(res, targets):
    '''
    simple accuracy calculation 
    '''
    n = len(res) #targets.shape[0]
    accurate = 0
    for i in range(n):
        #print i, j
        if res[i] == targets[i]:
            #print i, j
            accurate = accurate + 1
    return 1.0*(accurate)/n

def hinge_avg(hinge):
    return np.mean(hinge)

def q2_3_ans1(accu):
    print "test accuracy ", accu

def q2_3_ans2(accu):
    print "train accuracy ", accu
    

if __name__ == '__main__':
    
    
    gd1 = GDOptimizer(1, 0)
    opt_test_1 =  optimize_test_function(gd1)
    gd2 = GDOptimizer(1, 0.9)
    opt_test_2 = optimize_test_function(gd2)
    print "========Q2.1 start =============="
    print "=====opt test beta = 0===="
    plt.plot(opt_test_1, label = 'beta = 0 ')
    #plt.show()
    print "======opt test beta = 0.9====="
    plt.plot(opt_test_2, label = 'beta = 0.9')
    plt.title("SGD test")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    print "=======Q2.1 end ==============="
    print
    
    
    
    print "==========Q2.2 and 2.3 SVM ==========="
    gd1 = GDOptimizer(0.05, 0, 0)
    train_data, train_targets, test_data, test_targets = load_data()
    
    #Add one to bias
    n_train, m_train = train_data.shape
    n_test, m_test = test_data.shape
    #train_ones = np.ones((n_train, 1))
    np.insert(train_data, 0,1, axis=1)
    #test_ones = np.ones((n_test, 1))
    np.insert(test_data, 0,1, axis = 1)
    
    
    penalty = 1
    res = optimize_svm(train_data, train_targets, penalty, gd1, 100, 500)

    pred_train = res.classify(train_data)
    pred_test = res.classify(test_data)
    
    print "=======  SVM , momentum = 0 ======="
    #print "weight, ", res.w
    print "svm hinge loss train ", hinge_avg(res.hinge_loss(train_data, train_targets))
    print "svm hinge loss test ", hinge_avg(res.hinge_loss(test_data, test_targets))
    q2_3_ans1(accuracy_func(pred_train, train_targets))
    q2_3_ans2(accuracy_func(pred_test, test_targets))
    print "plot W, momemtum = 0"
    plot_w(res.w)
    
    print "======= SVM, momentum = 0.1 ======="
    gd2 = GDOptimizer(0.05, 0.1, 0)

    res2 = optimize_svm(train_data, train_targets, penalty, gd2, 100, 500)
    pred_test2 = res2.classify(test_data)
    pred_train2 = res2.classify(train_data)
    #print "weight with momentum ", res2.w 
    print "svm hinge loss train ", hinge_avg(res2.hinge_loss(train_data, train_targets))
    print "svm hinge loss test ", hinge_avg(res2.hinge_loss(test_data, test_targets))
    q2_3_ans1(accuracy_func(pred_train2, train_targets))
    q2_3_ans2(accuracy_func(pred_test2, test_targets))
    print "plot W, momemtum = 0.1"
    plot_w(res2.w)
    