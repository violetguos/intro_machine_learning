'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
import sklearn.neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
#TODO: KNN, SVM, 


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

def knn_news(X_train, y_train, X_test, y_test, y_names=None, confusion=False):
    '''
    predicting using KNN
    '''
    n_neighbors = 11
    weights = 'uniform'
    weights = 'distance'
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors)#, weights=weights)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    if not confusion:
        print ('Classification report:', 'magenta') #attrs=['bold'])
        print sklearn.metrics.classification_report(y_test, y_predicted)#, target_names=y_names)
    else:
        print ('Confusion Matrix:', 'magenta')# attrs=['bold'])
        print sklearn.metrics.confusion_matrix(y_test, y_predicted)


def rand_forest_news(X_train, y_train, X_test, y_test, y_names=None, confusion=False):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    #evaluate accuracy
    train_pred = clf.predict(X_train)
    print('rand forest baseline train accuracy = {}'.format((train_pred == y_train).mean()))
    
    test_pred = clf.predict(X_test)
    print('rand forest baseline test accuracy = {}'.format((test_pred == y_test).mean()))
    
def nn_news(X_train, y_train, X_test, y_test, y_names=None, confusion=False):
    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
    mlp.fit(X_train,y_train)
    predictions = mlp.predict(X_test)
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    
    
if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)

    #bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    train_tf, test_tf, feature_tf_names = tf_idf_features(train_data, test_data)
    #knn_news(train_tf, train_data.target, test_tf, test_data.target, feature_tf_names)
    #rand_forest_news(train_tf, train_data.target, test_tf, test_data.target, feature_tf_names)
    nn_news(train_tf, train_data.target, test_tf,test_data.target)