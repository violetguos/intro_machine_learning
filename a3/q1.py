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

    return newsgroups_train, newsgroups_test

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



def confusion_mat(true_labels, predict_labels):
    #number of unique labels
    #unique_true_labels = set(true_labels)
    
    conf = np.zeros((20,20))

    true_cate = np.zeros((20))
    pred_cate = np.zeros((20))

    #count number of labels in true for each of the 20 categories
    for i in range(20):
        for j in range(len(true_labels)):
            if i == true_labels[j]:
                true_cate[i] +=1
    
    for i in range(20):
        for j in range(len(predict_labels)):
            if i == true_labels[j]:
                pred_cate[i] +=1
    
    
    for i in range(20):
        for j in range(20):
            conf[i][j] = abs(true_cate[j] - pred_cate[i])
    return conf



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
    rand_states = [0, 10, 20, 30, 40, 50]
    
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
    print "isolate test", test_accuracy
    print('svm test accuracy = {}'.format((test_pred == y_test).mean()))
    print('svm train confustion matrix')
    #pprint(test_conf.tolist())
    #print test_conf
    for i in range(20):
        for j in range(20):
            if j < 19:
                print test_conf[i][j], '&',
            else:
                print test_conf[i][j], '\\\\'
            
    return test_accuracy


def knn_news(X_train, y_train, X_test, y_test, k_, y_names=None, confusion=False, feature_sel=False):
    '''
    predicting using KNN
    '''
    n_neighbors = 10
    weights = 'uniform'
    #weights = 'distance'

    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)

    
    if feature_sel:
        clf = clf.fit(X_train, y_train)
        ch2 = SelectKBest(chi2, k=k_)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)

    
    clf = clf.fit(X_train, y_train)
    y_test_predicted = clf.predict(X_test)
    y_train_predicted = clf.predict(X_train)
    if not confusion:
        print('KNN train accuracy = {}'.format((y_train_predicted == y_train).mean()))#, target_names=y_names)

        print('KNN test accuracy = {}'.format((y_test_predicted == y_test).mean()))#, target_names=y_names)
    else:
        print ('Confusion Matrix:', 'magenta')# attrs=['bold'])





def rand_forest_news(X_train, y_train, X_test, y_test, y_names=None, confusion=False):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    #evaluate accuracy
    print "=============="
    print "Random forest ensamble algorithm"
    train_pred = clf.predict(X_train)
    print('rand forest baseline train accuracy = {}'.format((train_pred == y_train).mean()))
    
    test_pred = clf.predict(X_test)
    print('rand forest baseline test accuracy = {}'.format((test_pred == y_test).mean()))
  

def kmeans_news(X_train, y_train, X_test, y_test, y_names=None, confusion=False):
    
    #vectorizer = TfidfVectorizer(stop_words='english')
    #X = vectorizer.fit_transform(X_train)

    clf = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)
    clf.fit(X_train)
    
    predictions = clf.predict(X_test)

    #predictions to the index

    
    closest, _ = sklearn.metrics.pairwise_distances_argmin_min(km.cluster_centers_, X)
    
    print closest
    print y_test
    
    print(classification_report(y_test,predictions))
    

def nn_news(X_train, y_train, X_test, y_test, y_names=None, confusion=False):
    #20: 0,57
    #20, 30, 20: 0.58
    nn_layers = {
            'Single neuron neural network':Perceptron(),
            #'hidden layer': MLPClassifier(hidden_layer_sizes=(10, 20, 10))
            }
    
    
    for cls_name, cls in nn_layers.items():
        cls.fit(X_train,y_train)
        predictions = cls.predict(X_test)
        train_pred = cls.predict(X_train)
        print "======================="
        print cls_name
        print('nn train accuracy = {}'.format((train_pred == y_train).mean()))
        #train_conf =  confusion_mat(y_train, train_pred)
        #print train_conf
        
        
        test_pred = cls.predict(X_test)
        print('nn test accuracy = {}'.format((test_pred == y_test).mean()))
        #print('nn test confustion matrix')
        #test_conf =  confusion_mat(y_test, test_pred)
        #print test_conf
        
        
def decision_tree_news(X_train, y_train, X_test, y_test,k_, feature_sel = True, y_names=None, confusion=False):
    
    
    clf = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=18, min_samples_leaf=5)
    #clf = tree.DecisionTreeRegressor()
    
    if feature_sel:
        
        ch2 = SelectKBest(chi2, k=k_)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
    
    clf = clf.fit(X_train, y_train)
    
    predictions = clf.predict(X_test)
    
    print(sklearn.metrics.accuracy_score(y_test,predictions))



 
if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)

    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    train_tf, test_tf, feature_tf_names = tf_idf_features(train_data, test_data)
    
    #TOP 3 algorithms
    #SVM is the best
    #svm_cross_val(train_tf, train_data.target, test_tf, test_data.target)
    svm_news(train_tf, train_data.target,test_tf, test_data.target, 0, y_names=None, confusion=False)
    # arr is the numpy ndarray

    #rand_forest_news(train_tf, train_data.target, test_tf, test_data.target, feature_tf_names)
    #nn_news(train_tf, train_data.target, test_tf, test_data.target, feature_tf_names)
    
    
    
    #NOTE: here's the algorithms that I tried and got low
    #Accuracy.
    
    
    #test KNN, with different k values
    #k_arr =[180, 320, 350, 500] # [10, 100, 150, 200, 300, 320, 350] #1000]
    #for k in k_arr:
    #    print "======================="
    #    print "k =%d KNN", k
    #    knn_news(train_tf, train_data.target, test_tf, test_data.target,k, feature_tf_names, False, True)
    
    #k_arr = [10, 100, 400, 600]
    #for k in k_arr:
    #    print "======================="
    #    print "k =%d decision tree news ", k
    #    decision_tree_news(train_tf, train_data.target, test_tf, test_data.target, k, True)
    
    #kmeans_news(train_tf, train_data.target, test_tf, test_data.target)
    