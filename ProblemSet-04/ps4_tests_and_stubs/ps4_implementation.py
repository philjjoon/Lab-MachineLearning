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

        if (0 < alpha_new[i]) and (alpha_new[i] < C):
        	b = b1
        elif (0 < alpha_new[j]) and (alpha_new[j] < C):
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
        
        L, H = self._compute_box_constraints(i, j, Y, alpha, C)
        if L == H:
        	return alpha, b, False

        eta = (2 * K[i, j]) - K[i, i] - K[j, j]
        if eta >= 0:
        	return alpha, b, False

        alpha_new = np.copy(alpha)
        alpha_new[j] = alpha[j] - ((Y[j] * (E_i - E_j)) / eta)
        if alpha_new[j] > H:
        	alpha_new[j] = H
        elif alpha_new[j] < L:
        	alpha_new[j] = L
        
        if abs(alpha[j] - alpha_new[j]) < self.tol:
        	alpha_new[j] = alpha[j]
        	return alpha, b, False

        alpha_new[i] = alpha[i] + (Y[i] * Y[j] * (alpha[j] - alpha_new[j]))
        b_new = self._compute_updated_b(E_i, E_j, i, j, K, Y, alpha, alpha_new, b, C)

        return alpha_new, b_new, True


    def fit(self, X, Y, kernel=False, kernelparameter=False, C=False):
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
        if kernel is not False:
            self.kernel = kernel
        if kernelparameter is not False:
            self.kernelparameter = kernelparameter
        if C is not False:
            self.C = C

        Y = np.squeeze(Y)
        
        d, n = X.shape
        alpha = np.zeros(n)
        b, p = 0, 0
        K = buildKernel(X, kernel=self.kernel, kernelparameter=self.kernelparameter)
        
        while p < self.max_passes:
        	a = 0
        	for i in range(n):
        		E_i = (np.sum(K[i, :] * Y * alpha) - b)  - Y[i]
        		if (((Y[i] * E_i) < -self.tol) and (alpha[i] < self.C)) or (((Y[i] * E_i) > self.tol) and (alpha[i] > 0)):
        			j = np.random.choice(n)
        			while j == i:
        				j = np.random.choice(n)

        			E_j = (np.sum(K[j, :] * Y * alpha) - b)  - Y[j]
        			alpha, b, changes = self._update_parameters(E_i, E_j, i, j, K, Y, alpha, b, self.C)

        			if changes:
        				a += 1

        	if a == 0:
        		p += 1
        	else:
        		p = 0
       
       	sv = (alpha >= 1e-5)
        self.alpha_sv = alpha[sv]
        self.X_sv = X[:, sv]
        self.Y_sv = Y[sv]
        self.b = b
        self.w = np.dot((self.alpha_sv * self.Y_sv)[np.newaxis, :], self.X_sv.T)

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
        
        K = buildKernel(X, Y=self.X_sv, kernel=self.kernel, kernelparameter=self.kernelparameter)
        p = np.sum(K * self.Y_sv[np.newaxis, :] * self.alpha_sv[np.newaxis, :], axis=1) - self.b
        self.ypred = np.sign(p)

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
    
    def fit(self, X, Y, kernel=False, kernelparameter=False, C=False):
    	if kernel is not False:
    		self.kernel = kernel
    	if kernelparameter is not False:
    		self.kernelparameter = kernelparameter
    	if C is not False:
    		self.C = C

    	d, n = X.shape
        K = buildKernel(X, kernel=self.kernel, kernelparameter=self.kernelparameter)

        # Here you have to set the matrices as in the general QP problem
        P = np.outer(Y, Y) * K
        q = np.ones(n) * -1
        G = np.append(np.diag(np.ones(n) * -1), np.eye(n), axis=0)
        h = np.append(np.zeros(n), np.ones(n) * self.C)
        A = Y[np.newaxis, :]
        b = 0.0
        
        # this is already implemented so you don't have to
        # read throught the cvxopt manual
        alpha = np.array(qp(cvxmatrix(P, tc='d'),
                            cvxmatrix(q, tc='d'),
                            cvxmatrix(G, tc='d'),
                            cvxmatrix(h, tc='d'),
                            cvxmatrix(A, tc='d'),
                            cvxmatrix(b, tc='d'))['x']).flatten()

        sv = (alpha >= 1e-5)
        inds = np.arange(len(alpha))[sv]
        self.alpha_sv = alpha[sv]
        self.X_sv = X[:, sv]
        self.Y_sv = Y[sv]

        self.b = 0
        for i in range(len(self.alpha_sv)):
        	self.b += self.Y_sv[i]
        	self.b -= np.sum(self.alpha_sv * self.Y_sv * K[inds[i], sv])

        self.b /= len(self.alpha_sv)

        return self

    def predict(self, X):

        K = buildKernel(X, Y=self.X_sv, kernel=self.kernel, kernelparameter=self.kernelparameter)
        p = np.sum(K * self.Y_sv[np.newaxis, :] * self.alpha_sv[np.newaxis, :], axis=1) + self.b
        self.ypred = np.sign(p)

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


def plot_svm_2d(X, y, model, title='SVM 2D'):
    # INSERT CODE
    X_pos = X[:, np.nonzero(y == 1)[0]]
    X_neg = X[:, np.nonzero(y == -1)[0]]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(model.X_sv[0, :], model.X_sv[1, :], c='red', marker='x', s=70, label='support vectors')
    ax.scatter(X_pos[0, :], X_pos[1, :], c='yellow', marker='o', label='positive data')
    ax.scatter(X_neg[0, :], X_neg[1, :], c='green', marker='o', label='negative data')
    
    d, _ = X.shape
    n = 100
    x1plot = np.linspace(np.min(X[0, :]), np.max(X[0, :]), n)
    x2plot = np.linspace(np.min(X[1, :]), np.max(X[1, :]), n)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    x = np.zeros([d, n*n])
    x[0, :] = X1.flatten()
    x[1, :] = X2.flatten()
    model.predict(x)

    ax.contour(X1, X2, model.ypred.reshape([n,n]), 1, label='separating hyperplane')

    ax.legend(loc='upper right')
    x_min, x_max = np.min(X[0, :]), np.max(X[0, :])
    y_min, y_max = np.min(X[1, :]), np.max(X[1, :])
    ax.set_xlim(x_min - 1, x_max + 4)
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_xlabel('$X_0$', fontsize=14)
    ax.set_ylabel('$X_1$', fontsize=14)
    ax.set_title(title, fontsize=18)
    #fig.savefig(title + '.png')

    
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
