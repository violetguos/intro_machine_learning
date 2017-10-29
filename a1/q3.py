import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt


BATCHES = 50



#####################
#This is a program that 
#
#

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
    #grad = np.dot(xtxw,xty)
    grad = xtxw - xty
    
    return (2*grad)

def var_grad (x):
    len_x = np.size(x)
    sum_x = 0
    for i in range(len_x):
        sum_x += x[i]
    
    avg_x = sum_x /float(len_x)
    
    sum_x2 = 0
    for i in range(len(x)):
        sum_x2+=(x[i] - avg_x)**2
    
    var_x = sum_x2 / float(len_x)
    return var_x

def square_metric(grad, grad_true):
    diff = grad - grad_true
    dist = (np.array(diff)**2).mean()

    return np.sqrt(dist)


            
        #print "average" , (reduce(lambda x, y: x + y, w_j) / len(w_j))


def grad_500(x, y, w, m, k, sampler):
    grad_sum =0
    batch_sum = 0
    batch_avg =0
    for i in range(0,k):
        X_b, y_b = sampler.get_batch(m)
        #print len(X_b)
        #for j in range(m):
        batch_grad = lin_reg_gradient(X_b, y_b,w)
        batch_sum += batch_grad
        batch_avg = batch_sum/m
    grad_sum +=batch_avg
        #print "batch_grad ", batch_grad
        #print "grad", grad_sum
        
    b_grad = grad_sum / k
    #print "b_grad", b_grad  
    return b_grad

def grad_var500(x, y, w, m, k, sampler):
    grad_sum =0
    batch_sum = 0
    batch_sum_list = []
    batch_avg =0
    var_list = []
    grad_2d = []
    var_j = 0
    for i in range(0, k): #500 iters
        X_b, y_b = sampler.get_batch(m)
        #print len(X_b)
        #for j in range(m):
        batch_grad = lin_reg_gradient(X_b, y_b,w) #have 13
        num = int(np.shape(batch_grad)[0])
        grad_2d.append(batch_grad)
        #batch_sum = np.sum(batch_grad)
        #batch_sum_list.append(batch_sum)
        #batch_avg = batch_sum/m
    for j in range(0, num):
        for i in range(0, k):
            batch_sum += grad_2d[i][j]
        batch_avg = batch_sum/float(k)
        for i in range(0,k):
            var_j += (grad_2d[i][j] - batch_avg)*(grad_2d[i][j] - batch_avg)
        var_ret = var_j / float(k)
        var_list.append(var_ret)    
        
        
        #print "batch_grad ", batch_grad
        #print "grad", grad_sum
        
    #b_grad = grad_sum / k
    #print "b_grad", b_grad  
    return np.array(var_list)


def grad_real(x, y, w):

    real_grad = lin_reg_gradient(x, y, w)
    #print "real grad ", real_grad
    #grad_sum +=batch_grad
    
    #b_grad = grad_sum / k
    #print batch_grad  
    return real_grad

def plot_log(m, sigma):
    print "----plotint var---"
    for i in range(13): #np.size(sigma):
        print np.shape(sigma)
        plt.plot(np.log(m), np.log(sigma[i]))
    
    #plt.yscale('log')
    plt.show()
    

def main():
    # Load data and randomltq3_y initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # Example usage
    #for K  = 500
    k = 500
    m = 50
    batch_grad = grad_500(X, y, w, m, k, batch_sampler)
    print "batch grad", batch_grad
    #print "final avg,", batch_grad
    true_grad = grad_real(X, y, w)
    
    #compute diff
    diff_sq = square_metric(batch_grad, true_grad)
    diff_cos = cosine_similarity(batch_grad, true_grad)
    print "diff_sq", diff_sq
    print "diff cos", diff_cos    
    #varience
    
    b_m_grad = []
    sigma = []
    '''
    for m1 in range(1,401):
        X_b, y_b = batch_sampler.get_batch(m1)
        b_m = grad_500(X_b, y_b, w, m1, k, batch_sampler)
        sigma_per = var_grad(b_m)
        #print "sig per", sigma_per
        sigma.append(sigma_per)
    '''
    #print "sigma ", sigma

    #########################3
    for m1 in range(1,401):
        #X_b, y_b = batch_sampler.get_batch(m1)
        b_m = grad_var500(X, y, w, m1, k, batch_sampler)
        #print "sig per", sigma_per
        #sigma_per = var_grad(b_m)
        sigma.append(b_m)
    
    print ("sigma")
    sigma = np.array(sigma)
    sigma_reshape = np.transpose(sigma)
    
    sigma_reshape = np.flip(sigma_reshape, 1)
    
    #print ("sigma len ", len(sigma)) 400
    print ("sigma len 1", len(sigma_reshape[0])) #13
 
    m_plot= np.arange(1,401)
    print type(m_plot)
    
    plot_log(m_plot, sigma_reshape)

    
    #reinit m for plotting

    #print "sigma", sigma

    #square Diff = 79165708.6263
    #cosine similarity, diif_cos = 0.999995845102
    
    
    
    

    
    #print true_grad
    
    '''
    [   905147.49317521   1938519.97649744   2407949.76862212 ...,
   3509262.95811409  64912228.61128841   2572091.7828057 ]
[  7.96380447e+06   1.90440009e+07   2.21960820e+07 ...,   3.46361153e+07
   6.66448338e+08   2.46394200e+07]
    '''
    

if __name__ == '__main__':
    main()