import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt


BATCHES = 50

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


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features) #w is ndarray

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    xTrans = np.transpose(X)
    xtx = np.dot(xTrans, X)
    xtxw =np.dot(xtx, w)
    xty = np.dot(xTrans, y)
    grad = 2*xtxw - 2*xty
    
    return grad

def var_grad (x):
    sum_x = 0
    for i in range(len(x)):
        sum_x += x[i]
    
    avg_x = sum_x /float(len(x))
    
    sum_x2 = 0
    for i in range(len(x)):
        sum_x2+=(x[i] - avg_x)**2
    
    var_x = sum_x2 / float(len(x))
    return var_x

def square_metric(grad, grad_true):
    diff = grad - grad_true
    dist = np.dot(diff, diff)

    return diff


            
        #print "average" , (reduce(lambda x, y: x + y, w_j) / len(w_j))


def grad_500(x, y, w, m, k, sampler):
    grad_sum =0
    batch_sum = 0
    batch_avg =0
    for i in range(0,k):
        X_b, y_b = sampler.get_batch(m)
        #print len(X_b)
        for j in range(m):
            batch_grad = lin_reg_gradient(X_b[j], y_b[j],w)
            batch_sum += batch_grad
        batch_avg = batch_sum/m
    grad_sum +=batch_avg
        #print "batch_grad ", batch_grad
        #print "grad", grad_sum
        
    b_grad = grad_sum / k
    print "b_grad", b_grad  
    return b_grad

def grad_real(x, y, w):

    real_grad = lin_reg_gradient(x, y, w)
    print "real grad ", real_grad
    #grad_sum +=batch_grad
    
    #b_grad = grad_sum / k
    #print batch_grad  
    return real_grad

def plot_log(m, sigma):
    print "----plotint var---"
    plt.plot(m, sigma)
    
    plt.yscale('log')
    plt.show()
    

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # Example usage
    #for K  = 500
    k = 500
    m = 50
    batch_grad = grad_500(X, y, w, m, k, batch_sampler)
    #print "final avg,", batch_grad
    true_grad = grad_real(X, y, w) #BUG!!!!!
    
    #compute diff
    diff_sq = square_metric(batch_grad, true_grad)
    diff_cos = cosine_similarity(batch_grad, true_grad)
    #print "diff_sq", diff_sq
    #print "diff cos", diff_cos    
    #varience
    
    #b_m_grad = []
    sigma = []
    for m1 in range(1,401):
        X_b, y_b = batch_sampler.get_batch(m1)
        b_m = grad_500(X_b, y_b, w, m1, k, batch_sampler)
        sigma_per = var_grad(b_m)
        #print "sig per", sigma_per
        sigma.append(sigma_per)
        
    #print "sigma ", sigma

    
    #reinit m for plotting
    m1 = np.logspace(1,3, 400)
    print type(m1)
    plot_log(m1, sigma)
    
    
    

    
    #print true_grad
    
    '''
    [   905147.49317521   1938519.97649744   2407949.76862212 ...,
   3509262.95811409  64912228.61128841   2572091.7828057 ]
[  7.96380447e+06   1.90440009e+07   2.21960820e+07 ...,   3.46361153e+07
   6.66448338e+08   2.46394200e+07]
    '''
    

if __name__ == '__main__':
    main()