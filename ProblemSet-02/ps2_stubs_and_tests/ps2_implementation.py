""" ps2_implementation.py

PUT YOUR NAME HERE:
Budi Yanto


Write the functions
- kmeans
- kmeans_agglo
- agglo_dendro
- norm_pdf
- em_gmm
- plot_gmm_solution

(c) Felix Brockherde, TU Berlin, 2013
    Translated to Python from Paul Buenau's Matlab scripts
"""

from __future__ import division  # always use float division
import numpy as np
from scipy.spatial.distance import cdist  # fast distance matrices
from scipy.cluster.hierarchy import dendrogram  # you can use this
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # for when you create your own dendrogram

def kmeans(X, k, max_iter=100):
    """ Performs k-means clustering

    Input:
    X: (d x n) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations

    Output:
    mu: (d x k) matrix with each cluster center in one column
    r: assignment vector
    """
    mu = randomInitCentroids(X, k)
    prev_mu = mu

    prev_r = np.zeros(X.shape[1], dtype=int)
    converged = False
    i = 1 # First iteration
    while ((not converged) and (i != (max_iter+1))):
        
        # Find closest centroids of each data point
        DM = cdist(mu.T, X.T, 'euclidean') # Compute the distance matrix
        C = (DM == np.min(DM, axis=0)) # Get the closest members for each cluster
        
        r = np.dot(np.arange(k), C)
        
        L = DM[np.nonzero(C)]
        
        loss = np.sum(L)
        
        # Compute new centroids
        members = np.sum(C, axis=1)[np.newaxis, :]
        mu = np.dot(X, C.T) / members
        
        converged = np.all(mu == prev_mu)
        #converged = np.sum(np.abs(mu_new - mu)) <= 1e-8

        total_changed = np.sum(r != prev_r)

        # Print some information after each iteration
        print '\nIteration: ' + str(i) + '/' + str(max_iter)
        print 'The number of cluster memberships which changed: ', total_changed
        print 'Loss: ' + str(loss)

        prev_mu  = mu
        prev_r = r
        i = i + 1 # increase iteration

    return mu, r

def randomInitCentroids(X, k):
    """ header here
    """
    d, n = X.shape
    clusters = np.random.choice(n, k, replace=False)
    mu = X[:, clusters]

    return mu


def kmeans_agglo(X, r):
    """ Performs agglomerative clustering with k-means criterion

    Input:
    X: (d x n) data matrix with each datapoint in one column
    r: assignment vector

    Output:
    R: (k-1) x n matrix that contains cluster memberships before each step
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """

    def kmeans_crit(X, r):
        """ Computes k-means criterion

        Input: 
        X: (d x n) data matrix with each datapoint in one column
        r: assignment vector

        Output:
        value: scalar for sum of euclidean distances to cluster centers
        """

        pass
    
    pass



def agglo_dendro(kmloss, mergeidx):
    """ Plots dendrogram for agglomerative clustering

    Input:
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """

    pass


def norm_pdf(X, mu, C):
    """ Computes probability density function for multivariate gaussian

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: vector for center
    C: covariance matrix

    Output:
    pdf value for each data point
    """

    pass


def em_gmm(X, k, max_iter=100, init_kmeans=False, eps=1e-3):
    """ Implements EM for Gaussian Mixture Models

    Input:
    X: (d x n) data matrix with each datapoint in one column
    k: number of clusters
    max_iter: maximum number of iterations
    init_kmeans: whether kmeans should be used for initialisation
    eps: when log likelihood difference is smaller than eps, terminate loop

    Output:
    pi: 1 x k matrix of priors
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """

    pass

def plot_gmm_solution(X, mu, sigma):
    """ Plots covariance ellipses for GMM

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """

    pass
