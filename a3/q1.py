'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.model_selection import KFold

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
import sklearn.neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.linear_model import Perceptron
from pprint import pprint


def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))
    categroies = newsgroups_train.target_names
    return newsgroups_train, newsgroups_test, categroies

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

single_nn = Perceptron() 

def confusion_mat(true_labels, predict_labels):
    #number of unique labels
    #unique_true_labels = set(true_labels)
    
    conf = np.zeros((20,20))



    #count number of labels for each of the 20 categories
    #for i in range(20):
    for j in range(len(true_labels)):
        curr_true_class = true_labels[j]
        for ii in range(len(predict_labels)):
            curr_pred_class = predict_labels[ii]#0 to 19
            conf[int(curr_true_class)][int(curr_pred_class)] +=1    
                    
        
    return conf

def most_confused_class(conf_mat):
    return np.unravel_index(conf_mat.argmax(), conf_mat.shape)


def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model


def svm_cross_val(X_train, y_train, X_test, y_test):
    rand_states = [0]# [0, 10, 20, 30, 40, 50]
    
    all_accuracy = []
    
    for rand in rand_states:
        # Loop over folds
        # Evaluate SVMs
        # ...
        kf = KFold(n_splits=10)
        fold_test_accuracy = []
        for train_index, test_index in kf.split(X_train):
            x_train_fold, x_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
            svm_res = svm_news(x_train_fold, y_train_fold, x_test_fold, y_test_fold, rand)
            fold_test_accuracy.append(svm_res)
    
        fold_accuracy = (1.0 *sum(fold_test_accuracy)) / (1.0 *len(fold_test_accuracy))   
        all_accuracy.append(fold_accuracy)
    all_accuracy = np.array(all_accuracy)
    optimal_rand = all_accuracy.argmax()
    print "Cross Validate result: rand state = ", optimal_rand
    
    #use the optimal, get the confusion matrix
    
    
    
    
    

def svm_news(X_train, y_train, X_test, y_test, rand_, y_names=None, confusion=False):
    '''
    predicting using SVM
    '''
    print "======================"
    print "SVM algorithm, hyper random state = ", rand_
    
    
    clf = LinearSVC(random_state=rand_)
    #clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors)#, weights=weights)
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    print('svm train accuracy = {}'.format((train_pred == y_train).mean()))
    print('svm train confustion matrix')
    train_conf =  confusion_mat(y_train, train_pred)
    print train_conf
    
    test_pred = clf.predict(X_test)
    test_conf =  confusion_mat(y_test, test_pred)
    test_accuracy = (test_pred == y_test).mean()
    
    print('svm test accuracy = {}'.format((test_pred == y_test).mean()))
    print('svm train confustion matrix')
    #pprint(test_conf.tolist())
    print test_conf.max()
    ci, cj  = most_confused_class(test_conf)
    #print "u est shape", y_names.shape
    print "most confused classes = ", y_names[ci], y_names[cj]
    
    
    #for i in range(20):
    #    for j in range(20):
    #        if j < 19:
    #            print test_conf[i][j], '&',
    #        else:
    #            print test_conf[i][j], '\\\\'
            
    return test_accuracy




def rand_forest_cross_val(X_train, y_train, X_test, y_test):
    num_est_arr = [10, 30, 50, 80, 100, 120, 150]
    best_est = [150]
    all_accuracy = []
    
    for num_est in num_est_arr:
        # Loop over folds
        # Evaluate SVMs
        # ...
        
        kf = KFold(n_splits=10)
        fold_test_accuracy = []
        for train_index, test_index in kf.split(X_train):
            x_train_fold, x_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
            rand_forest_res = rand_forest_news(x_train_fold, y_train_fold, x_test_fold, y_test_fold, num_est)
            fold_test_accuracy.append(rand_forest_res)
    
        fold_accuracy = (1.0 *sum(fold_test_accuracy)) / (1.0 *len(fold_test_accuracy))   
        all_accuracy.append(fold_accuracy)
    all_accuracy = np.array(all_accuracy)
    optimal_rand = all_accuracy.argmax()

    print "Cross Validate result: rand state = ", num_est_arr[optimal_rand]
    

def rand_forest_news(X_train, y_train, X_test, y_test, n_estimate, y_names=None, confusion=False):
    clf = RandomForestClassifier(n_estimators= n_estimate)
    clf.fit(X_train, y_train)
    
    #evaluate accuracy
    print "=============="
    print "Random forest ensamble algorithm"
    print "Fold with num_estimators = ",n_estimate
    train_pred = clf.predict(X_train)
    print('rand forest baseline train accuracy = {}'.format((train_pred == y_train).mean()))
    
    test_pred = clf.predict(X_test)
    print('rand forest baseline test accuracy = {}'.format((test_pred == y_test).mean()))
    test_accuracy = (test_pred == y_test).mean()
    return test_accuracy
  


def nn_news_cross_val(X_train, y_train, X_test, y_test):
    nn_layers = {
            'Single neuron neural network': Perceptron(),
            'hidden layer: (20, 10)':MLPClassifier(hidden_layer_sizes=(20,10 )),
            #'hidden layer: (1, 2, 1)': MLPClassifier(hidden_layer_sizes=(1, 2, 1)),
            'hidden layer: (5, 10, 5)':MLPClassifier(hidden_layer_sizes=(5, 10, 5)),
            'hidden layer: (10, 20, 10)': MLPClassifier(hidden_layer_sizes=(10, 20, 10)),
            'hidden layer: (15, 25, 15)': MLPClassifier(hidden_layer_sizes=(15, 25, 15)),   
            }
    #NOTE: a single neuron NN is a perceptron
    
    all_accuracy = []
    
    for cls_name, cls in nn_layers.items():
        # Loop over folds
        # Evaluate NNs
        print "NN fold with layer ",cls_name
        kf = KFold(n_splits=10)
        fold_test_accuracy = []
        for train_index, test_index in kf.split(X_train):
            x_train_fold, x_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
            nn_res = nn_news(cls, x_train_fold, y_train_fold, x_test_fold, y_test_fold)
            fold_test_accuracy.append(nn_res)
    
        fold_accuracy = (1.0 *sum(fold_test_accuracy)) / (1.0 *len(fold_test_accuracy))   
        all_accuracy.append(fold_accuracy)
    all_accuracy = np.array(all_accuracy)
    optimal_rand = all_accuracy.argmax()
  
    print "Cross Validate result: nn layer = ", nn_layers.items()[optimal_rand][0]
    
    

def nn_news(cls, X_train, y_train, X_test, y_test, y_names=None, confusion=False):
      
    cls.fit(X_train,y_train)
    predictions = cls.predict(X_test)
    train_pred = cls.predict(X_train)
    print "======================="
    
    print('nn train accuracy = {}'.format((train_pred == y_train).mean()))

    test_pred = cls.predict(X_test)
    print('nn test accuracy = {}'.format((test_pred == y_test).mean()))
    test_accuracy = (test_pred == y_test).mean()
    
    return test_accuracy
    
 

 
if __name__ == '__main__':
    train_data, test_data, categories_20 = load_data()
    #train_bow, test_bow, feature_names = bow_features(train_data, test_data)

    #bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    train_tf, test_tf, feature_tf_names = tf_idf_features(train_data, test_data)
    
    #TOP 3 algorithms
    #SVM is the best
    #svm_cross_val(train_tf, train_data.target, test_tf, test_data.target)
    #rand_forest_cross_val(train_tf, train_data.target, test_tf, test_data.target)
    #nn_news_cross_val(train_tf, train_data.target, test_tf, test_data.target)
    
    #print train_data.target_names [string of categories]
    #print set(train_data.target)# 0 to 19
    
    #final result with The picked hyperparameters
    svm_news(train_tf, train_data.target, test_tf, test_data.target, 0 , categories_20, confusion=False)
    
    #trying
    #nn_news(single_nn, train_tf, train_data.target, test_tf, test_data.target)
    #rand_forest_news(train_tf, train_data.target, test_tf, test_data.target, 150)