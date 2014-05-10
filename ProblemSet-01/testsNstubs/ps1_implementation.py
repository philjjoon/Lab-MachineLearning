""" sheet1_implementation.py

PUT YOUR NAME HERE:
Budi Yanto (308819)


Write the functions
- pca
- gammaidx
- lle
Write your implementations in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
import pylab as pl


def pca(X, m):
    ''' Calculate principle components of given data set X

    Usage:
        Z, U, D = pca(X, m)
    Parameters:
        X: (d x n) array of column vectors
        m: (scalar) number of component to be used
    Returns:
        Z: (m x n) matrix of the projected data points on feature vectors
        U: (d x d) matrix contains the principle directions
        D: (1 x d) 1D array contains the principle values sorted in descending order
    '''

    # Substract X from its mean
    X_zero_mean = X - np.mean(X, axis=1)[:, np.newaxis]

    # Calculate the covariance matrix
    S = np.cov(X_zero_mean)
    
    # Calculate the eigenvalues and eigenvectors
    D, U = la.eigh(S)
   
    # Sort the eigenvalues and eigenvectors
    inds = np.argsort(D)
    D = D[inds[::-1]]
    U = U[:, inds[::-1]]
    
    # Form the feature vectors
    U_reduction = U[:, :m]

    # Project the zero-mean X to feature vectos
    Z = np.dot(U_reduction.T, X_zero_mean)
    
    return Z, U, D


def gammaidx(X, k):
    ''' Calculate gamma index for each data point

    Usage: 
        y = gammaidx(X, k)
    Parameters:
        X: (d x n)  array of column vectors
        k: (scalar) number of neighbours
    Returns:
        y: (1 x n)  1D array of gamma index for each data point
    '''

    # Calculate distance matrix
    D = distmat(X, X)
    
    # Sort the distance matrix
    D = np.sort(D, axis=0)

    # Take only the k-nearest neighbours
    D = D[:k, :] # do not use the distance between the point and itself

    # Calculate the mean of the k-nearest neighbours
    y = np.mean(D, axis=0, keepdims=True)

    return y

def distmat(M, X):
    ''' Calculate a distance matrix between M and X
        
    Usage:
        D = distmat( M, X )
    Parameters:
        M: (d x k) array of column vectors
        X: (d x n) array of column vectors
    Returns:
        D: (k x n) array of distances between the vectors in M and X
    '''

    M2 = np.sum(M**2, axis=0)[:, np.newaxis]
    MX = np.dot(M.T, X)
    X2 = np.sum(X**2, axis=0)[np.newaxis, :]
    D2 = M2 - 2*MX + X2
    D2[D2<=0] = np.nan # not using the distance to itself and other minus distance
    return np.sqrt(D2)


def lle(X, m, n_rule, param, tol=1e-3):
    ''' Calculate locally linear embedding of a given data set X

    Usage:
        Y = lle(X, m, n_rule, param, tol)
    Parameters:
        X: (d x n) array of column vectors as input data set
        m: (scalar) dimension of the resulting embedding
        n_rule: (string) rule used to build the neighbourhood graph
        param: (scalar) the corresponding parameter for n_rule
        tol: (scalar) regularization parameter
    Returns:
        Y: (m x n) array of column vectors as resulting embedding
    '''

    # Get the dimension and total number of data points
    d, n = X.shape

    # Step 1: Finding the nearest neighbours
    print 'Step 1: Finding the nearest neighbours by rule ' + n_rule
    
    # Calculate distance matrix
    DM = distmat(X, X)

    if n_rule == 'knn':
        
        # Sort and get the k-nearest neighbours for each data point
        neighbours = np.argsort(DM, axis=0)[:param, :]

    elif n_rule == 'eps-ball':
        
        # Get the neighbours for each data point based on eps-ball
        neighbours = []
        for i in range(n):
            inds = np.where(DM[:, i] <= param)[0]
            neighbours.append(inds)

    else:
        raise Exception('Rule is unknown. Please choose either knn or eps-ball!')


    # Step 2: local reconstruction weights
    print 'Step 2: local reconstruction weights'
    
    # Only use regularization if param <= d
    if param <= d:
        tol = 0

    # Matrix W to store the weights
    W = np.zeros([n, n], dtype=float)                       
    
    for i in range(n): # Iterate through all data points

        # Calculate neighbours based on n_rule
        if n_rule == 'knn':
            neighbour = neighbours[:, i]
        elif n_rule == 'eps-ball':
            neighbour = neighbours[i]

        # Create matrix Z consisting of all neighbours of Xi and shift them to origin
        Z = X[:, neighbour] - X[:, i][:, np.newaxis]        

        # Compute local covariance 
        C = np.dot(Z.T, Z)

        # Use regularization if param > d                         
        C = C + (np.eye(len(neighbour), len(neighbour)) * tol * np.trace(C))

        # Solve C * Wi = 1     
        W[neighbour, i] = la.solve(C, np.ones(len(neighbour)))

        # Enforce sum(Wi) = 1          
        W[neighbour, i] = W[neighbour, i] / np.sum(W[:, i])                     


    # Step 3: compute embedding
    print 'Step 3: compute embedding'

    # Create a sparse matrix M
    M = np.eye(n, n, dtype=float)
    for i in range(n):
        if n_rule == 'knn':
            j = neighbours[:, i]
        elif n_rule == 'eps-ball':
            j = neighbours[i]

        w = W[j, i]
        M[i, j] -= w
        M[j, i] -= w
        w = w[:, np.newaxis]
        M[np.ix_(j, j)] += np.dot(w, w.T)

    # Calculate eigenvalues and eigenvectors
    E, V = la.eig(M)

    # Sort the eigenvalues and eigenvectors in ascending order
    inds = np.argsort(E)
    E = E[inds]
    V = V[:, inds]

    # Take (m+1) smallest eigenctors, discarding the smallest one
    Y = V[:, 1:(m+1)].T * np.sqrt(n)

    return Y


''' Various helper functions '''

def load_data(dataset='usps'):
    ''' Load various dataset '''

    # Load usps data set
    if dataset == 'usps':
        data = np.load('usps.npz')
        images = data['data_patterns']
        images = images - np.min(images)
        images = vec2im(images, (16, 16))
        labels = data['data_labels']
        return images.astype('float'), labels

    # Load banana data set
    if dataset == 'banana':
        X = np.load('banana_data.npz')
        data = X['data']
        label = X['label']
        return data, label

    # Load flatroll data set
    if dataset == 'flatroll':
        X = np.load('flatroll_data.npz')
        data = X['Xflat']
        true_emb = X['true_embedding']

    # Load fishbowl data set
    if dataset == 'fishbowl':
        X = np.load('fishbowl_data.npz')
        data = X['x_noisefree']
        true_emb = X['z']

    # Load dense fishbowl data set
    if dataset == 'fishbowl_dense':
        X = np.load('fishbowl_dense.npz')
        data = X['X']
        true_emb = None

    # Load swissroll data set
    if dataset == 'swissroll':
        X = np.load('swissroll_data.npz')
        data = X['x_noisefree']
        true_emb = X['z']

    return data, true_emb

    

def im2vec(images):
    ''' Convert images 3D to vector 2D '''

    x, y, n = images.shape
    images = images.reshape(x*y, n)
    fact = (x, y)

    return images, fact

def vec2im(images, fact=False):
    ''' Convert vector 2D to images 3D '''

    xy, n = images.shape
    if fact:
        x, y = fact
    else:
        x = int(np.sqrt(xy))
        while xy % x != 0:
            x -= 1
        y = xy / x

    images = images.reshape(x, y, n)

    return images

def visualize_data(ax, data, subs=(3,4)):
    ''' Visualize a set of images '''

    X, Y, N = data.shape
    x, y = subs
    matrix = np.ones([x*X, y*Y])
    row, col = 0, 0
    for i in range(x*y):
        x_start, x_end = row*X, (row+1)*X
        y_start, y_end = col*Y, (col+1)*Y
        matrix[x_start:x_end,y_start:y_end] = data[:,:,i]
        if row == x-1:
            row = 0
            col += 1
        else:
            row += 1
    
    ax.imshow(matrix,cmap='gray')
    ax.set_xticks([],[])
    ax.set_yticks([],[])

