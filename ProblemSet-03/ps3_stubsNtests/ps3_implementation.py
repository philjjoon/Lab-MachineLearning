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
    for param in combs:
        loss = 0
        for i in range(nrepetitions):
            splitted_data = split_data(X, nfolds)
                for j in range(nfolds):
                    test_data = splitted_data[j]
                    training_data = join_data(splitted_data, j)
                    method.fit(training_data, y, param[0], param[1], param[2])
                    method.predict(test_data)
                    loss += loss_function(y, method.y_pred)

        loss = loss / float((nfolds * nrepetitions))
        all_loss.append(loss)

    min_loss = np.argmin(all_loss)
    method.fit(X, y, combs[min_loss][0], combs[min_loss][1], combs[min_loss][2])
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

def join_data(splitted_data, test_idx):
	d, n = splitted_data[0].shape
	join = np.empty(d, n)
	first_data = True
	for j in range(len(splitted_data)):
		if j != test_idx:
			if first_data:
				join = splitted_data[j]
				first_data = False
			else:
				join = np.concatenate((join, splitted_data[j]), axis=1)

	return join


def split_data(X, nfolds):
    n = X.shape[1]
    size = n / nfolds
    rest = n % nfolds
    data_idx = np.arange(n)
    
    result = []
    counter = 0
    for i in range(nfolds):
        #print 'i: ', str(i) 
        idx = np.random.choice(data_idx, size + 1 if counter < rest else size, replace=False)
        #print 'idx: ' , idx, 'Length: ', idx.shape
        data = X[:, idx]
        result.append(data)

        mask = np.ones(len(data_idx), dtype=bool)
        #mask_idx = np.nonzero(data_idx == idx)[0]
        mask_idx = np.searchsorted(data_idx, idx)
        #print 'mask_idx: ', mask_idx
        mask[mask_idx] = False
        #print 'data_idx: ', data_idx.shape
        data_idx = data_idx[mask]
        counter += 1

    return result

def compute_size_each_fold(n, nfolds):
    '''your header here!
    '''
    size = n / nfolds
    rest = n % nfolds
    result = np.zeros(nfolds, dtype=int) + size
    
    if rest != 0:
        result[:rest] = result[:rest] + 1
    
    return result
        

def compute_kernel(X, kernel, param):
        ''' your header here '''
        if kernel == 'linear':
            return np.dot(X.T, X)

        elif kernel == 'polynomial':
            return np.power((np.dot(X.T, X) + 1), param)
        
        elif kernel == 'gaussian':
            return np.exp(-(squareform(pdist(X.T, 'sqeuclidean'))) / (2 * (param ** 2)))

        else:
            return "ERROR: Kernel is unknown"

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

        # Add an extra dimension to X which always set to 1 
        X_new = np.ones([d+1, n])
        X_new[1:, :] = X

        I = np.eye(n)
        I_new = np.zeros([n+1, n+1])
        I_new[1:, 1:] = I

        K = compute_kernel(X_new, kernel, kernelparameter)
        # alpha = (K + CI)inverse . Y
        alpha = np.dot(la.inverse(K + (regularization * I_new)), y)
        self.weight = np.dot(X_new, alpha)

        # weight
        return self
                    
    def predict(self, X):
        ''' your header here!
        '''
        #self.ypred -> see task description
        X_new = np.ones([d+1, n])
        X_new[1:, :] = X
        self.y_pred = np.dot(X_new.T, self.weight)

        return self

    def compute_kernel(X, kernel, kernelparameter):
        ''' your header here '''
        if kernel == 'linear':
            return np.dot(X.T, X)

        elif kernel == 'polynomial':
            return np.power((np.dot(X.T, X) + 1), kernelparameter)
        
        elif kernel == 'gaussian':
            return np.exp(-(squareform(pdist(X.T, 'sqeuclidean'))) / (2 * (kernelparameter ** 2)))

        else:
            return "ERROR: Kernel is unknown"
    


