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
from scipy.linalg import solve
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

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

    d, n = X.shape
    mu = randomInitCentroids(X, k)
    prev_mu = mu

    prev_r = np.zeros(n, dtype=int)
    converged = False
    counter = 1 # First iteration
    while (not converged):
        
        # Find closest centroids of each data point
        DM = cdist(mu.T, X.T, 'euclidean') # Compute the distance matrix
        C = (DM == np.min(DM, axis=0)) # Get the closest members for each cluster
        
        # Compute the vector of cluster membership
        r = np.dot(np.arange(k), C)

        # Compute the loss value        
        loss = np.sum(DM[np.nonzero(C)]) / n
        
        # Compute new centroids
        members = np.sum(C, axis=1)[np.newaxis, :]
        mu = np.dot(X, C.T) / members
        
        # Compute the number of cluster memberships which changed in the preceding step
        total_changed = np.sum(r != prev_r)

        # Print some information after each iteration
        print '\nIteration: ' + str(counter) + '/' + str(max_iter)
        print 'The number of cluster memberships which changed: ', total_changed
        print 'Loss: ' + str(loss)

        counter = counter + 1 # increase iteration
        converged = (np.all(mu == prev_mu) and np.all(r == prev_r)) or (counter > max_iter)
        #converged = np.sum(np.abs(mu_new - mu)) <= 1e-8

        prev_mu  = mu
        prev_r = r

    return mu, r

def randomInitCentroids(X, k):
    """ Get random data points as cluster centers

    Input:
    X: (d x n) data matrix with each data point in one column
    k: number of clusters

    Output:
    mu: (d x k) matrix with random cluster center in each column
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
        
        # Calculate cluster centers of each cluster
        clusters = np.unique(r) 
        k = len(clusters) 
        r = r[:, np.newaxis]
        C = (r == clusters[np.newaxis, :])
        mu = np.dot(X, C) / np.sum(C, axis=0)[np.newaxis, :]
        
        # Calculate the loss value
        DM = cdist(mu.T, X.T, 'euclidean') # Compute the distance matrix
        loss = np.sum((DM * C.T)) / X.shape[1]

        return loss

    def find_min_merge(X, r):
        """ Computes the indices of the two clusters that were merged at each step

        Input: 
        X: (d x n) data matrix with each datapoint in one column
        r: assignment vector

        Output:
        min_r: new assignment vector after merge two clusters
        min_loss: loss value of the merge
        min_mergeidx: indices of the two clusters that were merged
        """
        r_list = []
        loss_values = []
        mergeidx_list = []
        clusters = np.unique(r)
        k = len(np.unique(r))

        for i in range(k-1):
            for j in range(i+1, k):
                r_agglo = np.copy(r)
                r_agglo[np.nonzero(r_agglo==clusters[j])] = clusters[i]
                loss = kmeans_crit(X, r_agglo)
                idx = np.array([clusters[i], clusters[j]])
                r_list.append(r_agglo)
                loss_values.append(loss)
                mergeidx_list.append(idx)

        idx_min = np.argmin(np.array(loss_values)) # Get the smallest loss value after merging two clusters
        min_r = r_list[idx_min]
        min_loss = loss_values[idx_min]
        min_mergeidx = mergeidx_list[idx_min]

        return min_r, min_loss, min_mergeidx

    n = X.shape[1]
    k = len(np.unique(r))

    R = np.zeros([k-1, n])
    kmloss = np.zeros(k)
    mergeidx = np.zeros([k-1, 2])

    R[0, :] = r
    kmloss[0] = kmeans_crit(X, r)
    new_r = r

    for i in range(k-1):
        
        min_r, min_loss, min_mergeidx = find_min_merge(X, new_r)
        min_r[min_r == min_mergeidx[0]] = k + i
        
        if(i < k-2):
            R[i+1, :] = min_r

        mergeidx[i] = min_mergeidx
        kmloss[i+1] = min_loss
        
        new_r = min_r

    return R, kmloss, mergeidx


def agglo_dendro(kmloss, mergeidx):
    """ Plots dendrogram for agglomerative clustering

    Input:
    kmloss: vector with loss after each step
    mergeidx: (k-1) x 2 matrix that contains merge idx for each step
    """

    X = np.zeros([mergeidx.shape[0], 4])
    X[:, [0, 1]] = mergeidx
    X[:, 2] = kmloss[1:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Cluster index')
    ax.set_ylabel('Loss values of k-means criterion function')
    ax.set_title('Agglomerative Clustering Dendrogram', fontweight='bold', fontsize=14);
    dendrogram(X)


def norm_pdf(X, mu, C):
    """ Computes probability density function for multivariate gaussian

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: vector for center
    C: covariance matrix

    Output:
    pdf value for each data point
    """

    d = X.shape[0]
    X_mu = X - mu[:, np.newaxis]
    C = C + (0.1 * np.eye(d))
    det_C = np.linalg.det(C)

    tmp1 = np.power((2 * np.pi), -d/2.) * np.power(det_C, -1/2.)
    tmp2 = np.exp(-1/2. * (np.diag(np.dot(X_mu.T, solve(C, X_mu, sym_pos=True)))))

    return tmp1 * tmp2


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

    d, n = X.shape
    sigma = []
    if init_kmeans:
        mu, r = kmeans(X, k)
        pi = np.ones(k)
        for idx, cl in enumerate(np.unique(r)):
            pi[idx] = np.sum(r == cl) / n
            sigma.append(np.cov(X[:, np.nonzero(r==cl)[0]]))

    else:
        mu = randomInitCentroids(X, k)
        pi = np.ones(k) / k
        for i in range(k):
            sigma.append(np.eye(d))

    prev_likelihood = 0
    counter = 1
    converged = False
    while not converged:
        ''' The E-Step '''
        gamma = np.zeros([k, n])
        for i in range(k):
            gamma[i, :] = pi[i] * norm_pdf(X, mu[:, i], sigma[i])
            
        likelihood = np.sum(np.log(np.sum(gamma, axis=0)))
        gamma = gamma / np.sum(gamma, axis=0) # Normalize gamma

        ''' The M-Step '''
        N = np.sum(gamma, axis=1)
        pi = N / n
        mu = np.dot(X, gamma.T) / N[np.newaxis, :]

        for i in range(k):
            X_zero_mean = X - mu[:, i][:, np.newaxis]
            C = np.dot((gamma[i, :][np.newaxis, :] * X_zero_mean), X_zero_mean.T) / N[i]
            sigma[i] = C

        print '\nIteration: ' + str(counter) + '/' + str(max_iter)
        print 'Likelihood: ', likelihood

        counter = counter + 1
        converged = (np.abs(prev_likelihood - likelihood) < eps) or (counter > max_iter)
        prev_likelihood = likelihood

    return pi, mu, sigma
    

def plot_gmm_solution(X, mu, sigma):
    """ Plots covariance ellipses for GMM

    Input:
    X: (d x n) data matrix with each datapoint in one column
    mu: (d x k) matrix with each cluster center in one column
    sigma: list of d x d covariance matrices
    """
    colors = ['red', 'green', 'yellow', 'magenta', 'cyan', 'blue',  \
                'dimgray', 'orange', 'lightblue', 'lime']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[0, :], X[1, :])
    ax.scatter(mu[0, :], mu[1, :], marker='x', color='red', s=40)
    ax.set_title('GMM with ' + str(len(sigma)) + ' clusters')
    for k in range(len(sigma)):
        #U, s , Vh = np.linalg.svd(sigma[k])
        eigVal, eigVec = np.linalg.eig(sigma[k])
       
        orient = np.arctan2(eigVec[1,0], eigVec[0,0]) * (180 / np.pi)
        el = Ellipse(xy=mu[:,k], width=2.0*np.sqrt(eigVal[0]), \
            height=2.0*np.sqrt(eigVal[1]), angle=orient, \
            facecolor='none', edgecolor='red')
        ax.add_patch(el)

