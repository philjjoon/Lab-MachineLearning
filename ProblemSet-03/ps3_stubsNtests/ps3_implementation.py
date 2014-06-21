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
    pos_idx = np.nonzero(y_pred >= 0)
    neg_idx = np.nonzero(y_pred < 0)
    y_pred[pos_idx] = 1
    y_pred[neg_idx] = -1
    mult = y_true * y_pred
    loss = (len(np.where(mult == -1)[0])) / float(mult.shape[1])
    
    return loss

def squared_error_loss(y_true, y_pred):
    ''' returns the squared error loss
    '''
    loss = np.mean( (y_true - y_pred)**2 )
    return loss

    

def cv(X, y, method, params, loss_function=zero_one_loss, nfolds=10, nrepetitions=5):
    ''' your header here!
    '''
    start = time.time()
    d, n = X.shape
    kernel = params[1]
    kernelparam = params[3]
    regularization = params[5]
    combs = it.product(*[kernel, kernelparam, regularization])
    all_loss = []
    params = []
    for param in combs:
        print '---------------------------------------------------------'
        print '-  Running CV with params: ', param
        print '---------------------------------------------------------'
        params.append(param)
        loss = 0
        for i in range(nrepetitions):
            print 'Repetition: ', i
            splitted_data, splitted_Y = split_data(X, y, nfolds)
            for j in range(nfolds):
                print 'Fold: ', j
                test_data = splitted_data[j]
                #print 'test_data: ', test_data
                training_data, training_Y = join_data(splitted_data, splitted_Y, j)
                method.fit(training_data, training_Y, param[0], param[1], param[2])
                method.predict(test_data)
                loss += loss_function(splitted_Y[j], method.ypred)

        loss = loss / float((nfolds * nrepetitions))
        all_loss.append(loss)

    min_loss = np.argmin(all_loss)
    method.fit(X, y, params[min_loss][0], params[min_loss][1], params[min_loss][2])
    method.predict(X)
    method.cvloss = min(all_loss)

    end = time.time()
    method.time = end - start
    print 'Finish CV in ' + str(method.time) + ' seconds'
    
    return method

def join_data(splitted_data, splitted_Y, test_idx):
    d, n = splitted_data[0].shape
    join_data = np.empty([d, n])
    join_Y = np.empty(n)
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
    size = n / nfolds
    rest = n % nfolds
    data_idx = np.arange(n)
    
    splitted_data = []
    splitted_Y = []
    counter = 0
    for i in range(nfolds):
        idx = np.random.choice(data_idx, size + 1 if counter < rest else size, replace=False)
        data = X[:, idx]
        splitted_data.append(data)
        splitted_Y.append(Y[:, idx])

        mask = np.ones(len(data_idx), dtype=bool)
        mask_idx = np.searchsorted(data_idx, idx)
        mask[mask_idx] = False
        data_idx = data_idx[mask]
        counter += 1

    return splitted_data, splitted_Y

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
        self.X = X
        K = self.compute_kernel(X, X, kernel, kernelparameter)
        if regularization == 0:
            self.regularization = self.efficient_LOOCV(K, y)

        #self.alpha = np.dot(la.inv(K + (self.regularization * np.eye(n))), y.T)
        self.alpha = la.solve((K + (self.regularization * np.eye(n))), y.T)
        
        return self
                    
    def predict(self, X):
        ''' your header here!
        '''

        K = self.compute_kernel(X, self.X, self.kernel, self.kernelparameter)
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

    def efficient_LOOCV(self, K, y):
        offset = 0.8
        n_grid = 10
        U, L, UT = la.svd(K)
        mu = np.mean(L)
        L = np.diag(L)
        n = L.shape[0]
        min_value = np.log10(mu - offset)
        max_value = np.log10(mu + offset)
        C = np.logspace(min_value, max_value, n_grid)
        UL = np.dot(U, L)
        #UTY = np.dot(UT, y)
        err_list = []
        for c in C:
            D = L + (c * np.eye(n))
            D_inv = np.diag(1. / np.diagonal(D))
            S = np.dot(np.dot(UL, D_inv), UT)
            y_pred = np.dot(S, y.T)
            err = np.mean(((y - y_pred) / (np.ones(n) - np.diagonal(S))) ** 2 )
            err_list.append(err)

        min_idx = np.argmin(err_list)
        
        return C[min_idx]


