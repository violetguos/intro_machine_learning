import numpy as np
from sklearn.datasets import load_boston

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
    xtxw =2*np.dot(xtx, w)
    xty = 2*np.dot(xTrans, y)
    grad = xtxw - xty
    
    return grad

def grad_var_m(x,y,w,m,k):
    for i in range(0,k):
        w_j = []

        for m in range(1, 400):
            batch_grad = lin_reg_gradient(X_b, y_b, w)
            w_j.append(batch_grad)
            
        print "average" , (reduce(lambda x, y: x + y, w_j) / len(w_j))


def grad_500(x, y, w, k):
    grad_sum =0
    for i in range(0,k):
        batch_grad = lin_reg_gradient(x, y, w)
        grad_sum +=batch_grad
    
    b_grad = grad_sum / k
    print batch_grad  
    return b_grad

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # Example usage
    #for K  = 500
    k = 500
    m = 50
    X_b, y_b = batch_sampler.get_batch(m)
    batch_grad = grad_500(X_b, y_b, w, k)
    print "final avg,", batch_grad
    true_grad = lin_reg_gradient(X, y, w)

    
    #varience
    
    

    
    print true_grad
    
    '''
    [   905147.49317521   1938519.97649744   2407949.76862212 ...,
   3509262.95811409  64912228.61128841   2572091.7828057 ]
[  7.96380447e+06   1.90440009e+07   2.21960820e+07 ...,   3.46361153e+07
   6.66448338e+08   2.46394200e+07]
    '''
    

if __name__ == '__main__':
    main()