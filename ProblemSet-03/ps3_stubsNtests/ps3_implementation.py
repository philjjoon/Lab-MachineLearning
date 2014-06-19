""" ps3_implementation.py

PUT YOUR NAME HERE:
BUDI YANTO


Write the functions
- cv
- zero_one_loss
- krr
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
import itertools as it
import time
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform

def zero_one_loss(y_true, y_pred):
    ''' your header here!
    '''
    pos_idx = np.nonzero(y_pred >= 0)[0]
    neg_idx = np.nonzero(y_pred < 0)[0]
    y_pred[pos_idx] = 1
    y_pred[neg_idx] = -1
    mult = y_true * y_pred
    loss = (len(np.where(mult == -1)[0])) / float(len(mult))

    return loss



    

def cv(X, y, method, params, loss_function=zero_one_loss, nfolds=10, nrepetitions=5):
    ''' your header here!
    '''
    d, n = X.shape
    #np.random.choice(n, nfolds, replace=False)
    #n_onefold = n / nfolds
    #sizes = compute_size_each_fold(n, nfolds)
    #data_idx = np.arange(n)
    #rest_idx = np.arange(n)
    kernel = params[1]
    kernelparam = params[3]
    regularization = params[5]
    combs = it.product(*[kernel, kernelparam, regularization])
    all_loss = []
    #props = []
    #print 'y: ', y.shape
    params = []
    for param in combs:
        params.append(param)
        loss = 0
        for i in range(nrepetitions):
            splitted_data, splitted_Y = split_data(X, y, nfolds)
            for j in range(nfolds):
                test_data = splitted_data[j]
                training_data, training_Y = join_data(splitted_data, splitted_Y, j)
                #print 'test_data: ', test_data.shape
                #print 'training_data: ', training_data.shape
                method.fit(training_data, training_Y, param[0], param[1], param[2])
                method.predict(test_data)
                loss += loss_function(splitted_Y[j], method.ypred)

        loss = loss / float((nfolds * nrepetitions))
        all_loss.append(loss)

    min_loss = np.argmin(all_loss)
    method.fit(X, y, params[min_loss][0], params[min_loss][1], params[min_loss][2])
    method.predict(X)
    method.cvloss = min(all_loss)


    '''
    for i in range(nfolds):
        print 'Rest-Idx Length: ', len(rest_idx)
        print 'Rest-Idx [' + str(i) + ']: ', rest_idx
        test_idx = np.random.choice(rest_idx, sizes[i], replace=False)
        print 'Test-Idx Length: ', len(test_idx)
        print 'Test-Idx [' + str(i) + ']: ', test_idx
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False

        training_idx = data_idx[mask]

        mask = np.ones()
        rest_idx = rest_idx[mask]
        #rest_index = rest
        print 'Training-Idx Length: ', len(training_idx)
        print 'Training-Idx [' + str(i) + ']: ', training_idx
	'''
    return method

def join_data(splitted_data, splitted_Y, test_idx):
    d, n = splitted_data[0].shape
    join_data = np.empty([d, n])
    join_Y = np.empty(n)
    #join = np.zeros(d, n)
    first_data = True
    for j in range(len(splitted_data)):
        if j != test_idx:
            if first_data:
                join_data = splitted_data[j]
                join_Y = splitted_Y[j]
                first_data = False
            else:
                join_data = np.concatenate((join_data, splitted_data[j]), axis=1)
                join_Y = np.concatenate((join_Y, splitted_Y[j]), axis=1)

    return join_data, join_Y


def split_data(X, Y, nfolds):
    n = X.shape[1]
    #print 'n: ', n
    size = n / nfolds
    rest = n % nfolds
    data_idx = np.arange(n)
    
    splitted_data = []
    splitted_Y = []
    counter = 0
    for i in range(nfolds):
        #print 'i: ', str(i) 
        idx = np.random.choice(data_idx, size + 1 if counter < rest else size, replace=False)
        #print 'idx: ' , idx, 'Length: ', idx.shape
        data = X[:, idx]
        splitted_data.append(data)
        splitted_Y.append(Y[:, idx])

        mask = np.ones(len(data_idx), dtype=bool)
        #mask_idx = np.nonzero(data_idx == idx)[0]
        mask_idx = np.searchsorted(data_idx, idx)
        #print 'mask_idx: ', mask_idx
        mask[mask_idx] = False
        #print 'data_idx: ', data_idx.shape
        data_idx = data_idx[mask]
        counter += 1

    return splitted_data, splitted_Y

def compute_size_each_fold(n, nfolds):
    '''your header here!
    '''
    size = n / nfolds
    rest = n % nfolds
    result = np.zeros(nfolds, dtype=int) + size
    
    if rest != 0:
        result[:rest] = result[:rest] + 1
    
    return result
        
'''
def compute_kernel(X, kernel, param):
        
        if kernel == 'linear':
            return np.dot(X.T, X)

        elif kernel == 'polynomial':
            return np.power((np.dot(X.T, X) + 1), param)
        
        elif kernel == 'gaussian':
            return np.exp(-(squareform(pdist(X.T, 'sqeuclidean'))) / (2 * (param ** 2)))

        else:
            return "ERROR: Kernel is unknown"
'''
class krr():
    ''' KRR class implementation
    '''
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization    
    
    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
        ''' your header here!
        '''
        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization

        d, n = X.shape
        #print 'd: ', d
        #print 'n: ', n
        #print 'y.shape: ', y.shape
        # Add an extra dimension to X which always set to 1 
        #X_new = np.ones([d+1, n])
        #X_new[1:, :] = X

        #I = np.eye(n)
        #I_new = np.zeros([n+1, n+1])
        #I_new[1:, 1:] = I
        self.X = X
        K = self.compute_kernel(X, X, kernel, kernelparameter)
        
        #print 'K.shape: ', K.shape
        #print 'y.shape: ', y.shape
        # alpha = (K + CI)inverse . Y
        self.alpha = np.dot(la.inv(K + (regularization * np.eye(n))), y.T)
        #self.weight = np.dot(X, self.alpha)
        #self.weight = np.dot(np.dot(la.inv(X), K), self.alpha)
        #self.weight = np.array(la.lstsq(X.T, np.dot(K, self.alpha)))
        #print 'self.weight: ', self.weight.shape
        # weight

        return self
                    
    def predict(self, X):
        ''' your header here!
        '''
        #self.ypred -> see task description
        #X_new = np.ones([d+1, n])
        #X_new[1:, :] = X
        #K = compute_kernel(X, self.kernel, self.kernelparameter)
        #print 'K: ', K.shape

        #self.ypred = np.dot(K, self.alpha).T
        #print 'X.shape: ', X.shape
        #print 'Alpha: ', self.alpha.shape
        #weight = np.dot(X, self.alpha)
        K = self.compute_kernel(X, self.X, self.kernel, self.kernelparameter)
        #print 'K: ', K.shape
        #print 'alpha: ', self.alpha.shape
        self.ypred = np.dot(K, self.alpha).T

        return self

    def compute_kernel(self, X, Z, kernel, kernelparameter):
        ''' your header here '''
        if kernel == 'linear':
            return np.dot(X.T, Z)

        elif kernel == 'polynomial':
            return np.power((np.dot(X.T, Z) + 1), kernelparameter)
        
        elif kernel == 'gaussian':
            return np.exp(-(cdist(X.T, Z.T, 'sqeuclidean')) / (2 * (kernelparameter ** 2)))
            #return np.exp(-(squareform(pdist(X.T, 'sqeuclidean'))) / (2 * (kernelparameter ** 2)))

        else:
            return "ERROR: Kernel is unknown"
    


