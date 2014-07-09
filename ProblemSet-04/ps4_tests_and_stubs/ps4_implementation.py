""" ps4_implementation.py

PUT YOUR NAME HERE:
BUDI YANTO


Complete the classes and functions
- svm_smo
- svm_qp
- plot_svm_2d
Write your implementations in the given functions stubs!


(c) Felix Brockherde, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import sklearn.svm
from cvxopt.solvers import qp
from cvxopt import matrix as cvxmatrix

class svm_smo():
    """ Support Vector Machine """

    def __init__(self, kernel='linear', kernelparameter=1., C=0., max_passes=10, tol=1e-5):
        """ Init Support Vector Machine
        kernel: a value that build_kernel accepts as kernel
        kernelparameter: a value that build_kernel accepts as kernelparameter
        C: the regularization parameter
        """
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.max_passes = max_passes
        self.tol = tol

        self.alpha_sv = None
        self.X_sv = None
        self.Y_sv = None
        self.b = None


    def _compute_box_constraints(self, i, j, Y, alpha, C):
        """ Computes the box constraints L and H

        Input are the indexes i and j, the target values Y, the alphas, and the
        regularization parameter C.
        Output are the constraints L and H.
        """

        # INSERT_CODE
        if Y[i] == Y[j]:
        	L = max(0, alpha[i] + alpha[j] - C) 
        	H = min(C, alpha[i] + alpha[j])
        else:
        	L = max(0, alpha[j] - alpha[i])
        	H = min(C, C + alpha[j] - alpha[i])

        return L, H


    def _compute_updated_b(self, E_i, E_j, i, j, K, Y, alpha_old, alpha_new,
            b_old, C):
        """ Computes the updated b """

        b1 = b_old + E_i + (Y[i] * (alpha_new[i] - alpha_old[i]) * K[i, i]) + (Y[j] * (alpha_new[j] - alpha_old[j]) * K[i, j])

        b2 = b_old + E_j + (Y[i] * (alpha_new[i] - alpha_old[i]) * K[i, j]) + (Y[j] * (alpha_new[j] - alpha_old[j]) * K[j, j]) 

        if (alpha_new[i] > 0) and (alpha_new[i] < C):
        	b = b1
        elif (alpha_new[j] > 0) and (alpha_new[j] < C):
        	b = b2
        else:
        	b = (b1 + b2) / 2
        return b


    def _update_parameters(self, E_i, E_j, i, j, K, Y, alpha, b, C):
        """ This updates the parameters alpha_i, alpha_j, and b

        Input are the errors E_i and E_j, the indexes i and j, the kernel K,
        the target values Y, the alphas, the bias b, and the regularization
        parameter C.
        Output are the new alphas, the new bias b, and True if there were
        changes and False if there were no changes.

        See the pseudo code in the Handbook.
        """
        alpha_new = np.copy(alpha)
        changes = False
        k = (2 * K[i, j]) - K[i, i] - K[j, j]
        
        if k < 0:
        	alpha_j_temp = alpha[j] - ((Y[j] * (E_i - E_j))/ k)
        	
        	L, H = self._compute_box_constraints(i, j, Y, alpha, C)
  
        	if alpha_j_temp > H:
        		alpha_new[j] = H
        	elif alpha_j_temp < L:
        		alpha_new[j] = L
        	else:
        		alpha_new[j] = alpha_j_temp

        	alpha_new[i] = alpha[i] + (Y[i] * Y[j] * (alpha[j] - alpha_new[j]))
   
        	if abs(alpha[j] - alpha_new[j]) >= 1e-5:
        		changes = True

        	b = self._compute_updated_b(E_i, E_j, i, j, K, Y, alpha, alpha_new, b, C)

        return alpha_new, b, changes


    def fit(self, X, Y):
        """
        Fit the Support Vector Machine

        Parameters
        ----------
        X: numpy array or sparse matrix of shape [n_samples,n_features]
           Training data
        y: numpy array of shape [n_samples, n_targets]
           Target values

        Returns
        -------
        self: returns an instance of self.
        """
        d, n = X.shape
        alpha = np.zeros(n)
        b, p = 0, 0
        K = buildKernel(X, kernel=self.kernel, kernelparameter=self.kernelparameter)
        while p < self.max_passes:
        	a = 0
        	F_X = np.sum((alpha[np.newaxis, :] * Y[np.newaxis, :] * K), axis=1) - b
        	E = F_X - Y
        	YE = Y * E
        	for i in range(n):
        		if (((YE[i] < -self.tol) and (alpha[i] < self.C))
        			or ((YE[i] > self.tol) and (alpha[i] > 0))):
        			inds = np.arange(n)
        			inds = np.delete(inds, np.nonzero(inds == i))
        			j = np.random.choice(inds)
        			alpha, b, changes = self._update_parameters(E[i], E[j], i, j, K, Y, alpha, b, self.C)
        			if changes:
        				a += 1

        	if a == 0:
        		p += 1
        	else:
        		p = 0

        #gamma = Y * (np.sum((alpha * Y * K + b), axis=1))
        #inds_sv = 
        #inds_sv = np.nonzero(alpha > 0 and alpha < self.C)
        self.alpha_sv = alpha[(0 < alpha) & (alpha < self.C)]
        self.X_sv = X[:, (0 < alpha) & (alpha < self.C)]
        self.Y_sv = Y[(0 < alpha) & (alpha < self.C)]
        self.b = b

        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y : array, shape = [n_samples]
            Predicted class label per sample.
        """
        K = buildKernel(self.X_sv, Y=X, kernel=self.kernel, kernelparameter=self.kernelparameter)
        print 'K.shape predict: ', K.shape
        
        gamma = np.sum((self.alpha_sv[np.newaxis, :] * self.Y_sv[np.newaxis, :] * K.T), axis=1) - self.b
        self.ypred = np.sign(gamma)

        return self


class svm_qp():
    """ Support Vector Machines via Quadratic Programming """

    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        self.kernel = kernel
        self.kernelparameter = kernelparameter
        self.C = C
        self.alpha_sv = None
        self.b = None
        self.X_sv = None
        self.Y_sv = None
    
    def fit(self, X, Y):

        # INSERT_CODE
        
        # Here you have to set the matrices as in the general QP problem
        #P = 
        #q = 
        #G = 
        #h = 
        #A =   # hint: this has to be a row vector
        #b =   # hint: this has to be a scalar
        
        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        '''alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()'''

        #b = 

    def predict(self, X):

        # INSERT_CODE

        return self


# This is already implemented for your convenience
class svm_sklearn():
    """ SVM via scikit-learn """
    def __init__(self, kernel='linear', kernelparameter=1., C=1.):
        if kernel == 'gaussian':
            kernel = 'rbf'
        self.clf = sklearn.svm.SVC(C=C,
                                   kernel=kernel,
                                   gamma=1./(1./2. * kernelparameter ** 2),
                                   degree=kernelparameter,
                                   coef0=kernelparameter)
    def fit(self, X, Y):
        self.clf.fit(X.T, y)
        self.X_sv = X[:, self.clf.support_]
        self.y_sv = y[self.clf.support_]
    def predict(self, X):
        self.ypred = self.clf.predict(X.T)
        return self


def plot_svm_2d(X, y, model):
    # INSERT CODE
    pass

    
def sqdistmat(X, Y=False):
    if type(Y) == type(False) and Y == False:
        X2 = sum(X**2,0)[np.newaxis,:]
        D2 = X2 + X2.T - 2*np.dot(X.T, X)
    else:
        X2 = sum(X**2,0)[:,np.newaxis]
        Y2 = sum(Y**2,0)[np.newaxis,:]
        D2 = X2 + Y2 - 2*np.dot(X.T, Y)
    return D2


def buildKernel(X, Y=False, kernel='linear', kernelparameter=0):
    d, n = X.shape
    if type(Y) == type(False) and Y == False:
        Y = X
    if kernel == 'linear':
        K = np.dot(X.T,Y)
    elif kernel == 'polynomial':
        K = np.dot(X.T,Y) + 1
        K = K**kernelparameter
    elif kernel == 'gaussian':
        K = sqdistmat(X,Y)
        K = np.exp(   K / (- 2 * d * kernelparameter**2)   )
    else:
        raise Exception('unspecified kernel')
    return K
