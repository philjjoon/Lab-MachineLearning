""" ps3_implementation.py

PUT YOUR NAME HERE:
Benjamin Pietrowicz


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
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D

def zero_one_loss(y_true, y_pred):
    ''' Computes the procentual error between the true and predicted labels,
    where the predicted labels may be real numbers.
    y_true: 1 x n array with labels -1 or 1
    y_pred: 1 x n array with labels with real values
    '''
    _, n = y_true.shape
    diff = y_true - np.sign(y_pred)
    return len(np.nonzero(diff)[1])/float(n)
    

def cv(X, y, method, params, loss_function=zero_one_loss, nfolds=10, nrepetitions=5):
    ''' Returns class object with has been trained with the optimal parameter values.
    Input:
    X: d x n array of data
    y: 1 x n array of labels/regression targets
    method: class object with methods .fit(X,y,params[1::2]) and .predict(X)
    params: list of parameters, where index%2 == 0 elements are parameters names and
        index%2 == 1 are the parameters
    loss_function: (optional) function handle to loss function to be used
    nfolds: (optional) number of folds CV creates to seperate the data
    nrepetitions: (optional) number of repetitions CV seperates the data into folds
    '''
    #print 'X.shape: ', X.shape
    #print 'y.shape: ', y.shape
    d,n = X.shape
    L = list(it.product(*params[1::2]))
    size = len(L)
    CVloss = np.zeros(size)
    cvloss = 0
    verbose = np.unique(np.round(np.linspace(0,size,11)))[:-1]
    method.rocdata = np.zeros([2, n])
    start = time.time()
    for index,param in enumerate( L):
        print 'param: ', param
        for i in range(nrepetitions):
            indices = np.random.permutation(n)
            split = np.round(np.arange(1,nfolds) * n/nfolds)
            indices = np.split(indices,split) # list of indices for each fold
            for j in range(nfolds):
                #try:
                Xtr = np.delete(X,indices[j],axis=1)
                Ytr = np.delete(y,indices[j],axis=1)
                method.fit(Xtr,Ytr,*param)
                Xte = X[:,indices[j]]
                method.predict(Xte)
                method.rocdata[0, indices[j]] = y[0, indices[j]]
                method.rocdata[1, indices[j]] = np.squeeze(method.ypred)[:]
                cvloss += loss_function(y[:,indices[j]],method.ypred)
                #except:
                #    print 'Exception'
                #    continue
                    
        CVloss[index] = cvloss
        cvloss = 0
        if index in verbose:
            stamp = time.time()
            print '%2d / %d parameter combinations completed. Estimated remaining time: %0.3f'%(index+1,size,(size-index)*(stamp-start)/(index+1))
    CVloss /= float(nrepetitions * nfolds)
    method.fit(X,y,*L[np.argmin(CVloss)])
    method.predict(X)
    end = time.time()
    #method.cvloss = loss_function(y,method.ypred)
    method.cvloss = min(CVloss)
    method.time = end - start
    return method

    
class krr():
    ''' Class with methods fit and predict.
    '''
    def __init__(self, kernel='linear', kernelparameter=1, regularization=0):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.regularization = regularization    
    
    def fit(self, X, y, kernel=False, kernelparameter=False, regularization=False):
        ''' trains method with data points X, labels y and given parameters
        '''
        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if regularization is not False:
            self.regularization = regularization
        self.Xtr = X
        self.ytrue = y
        d,n = X.shape
        if self.kernel == 'linear':
            K = X.T.dot(X)
        elif self.kernel == 'polynomial':
            K = (X.T.dot(X) + 1)**self.kernelparameter
        elif self.kernel == 'gaussian':
            K = np.exp((-cdist(X.T,X.T)**2)/(2*self.kernelparameter**2))
        if self.regularization == 0:
            eigvals,eigvecs = np.linalg.eigh(K)
            cand = np.log10(np.mean(eigvals))
            cand = np.logspace(-3+cand,3+cand,10)
            n = len(eigvals)
            A = eigvecs.dot(np.eye(n)*eigvals)
            C = eigvecs.T.dot(self.ytrue.T)
            err = np.zeros(len(cand))
            for i,elem in enumerate(cand):
                B = 1/(eigvals + elem) * np.eye(n)
                S = A.dot(B).dot(eigvecs.T)
                diag = np.diag(S)
                err[i] = np.sum(((self.ytrue.T - A.dot(B).dot(C))/(1-diag))**2)/n
            self.regularization = cand[np.argmin(err)]
        self.alpha = la.solve(K+ self.regularization * np.eye(n), self.ytrue.T)
        return self
                    
    def predict(self, X):
        ''' uses the models fit and returns the method with the additional field
        ypred of predictions
        '''
        if self.kernel == 'linear':
            K = self.Xtr.T.dot(X)
        elif self.kernel == 'polynomial':
            K = (self.Xtr.T.dot(X) + 1)**self.kernelparameter
        elif self.kernel == 'gaussian':
            K = np.exp((-cdist(self.Xtr.T,X.T)**2)/(2*self.kernelparameter**2))
        self.ypred = self.alpha.T.dot(K)
        return self
    


