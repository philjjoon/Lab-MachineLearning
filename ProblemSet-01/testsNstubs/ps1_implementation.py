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
    ''' your header here!
    '''

    # Substract X from its mean
    X_zero_mean = X - np.mean(X, axis=1)[:, np.newaxis]

    # Calculate the covariance matrix
    S = np.cov(X_zero_mean)
    
    # Calculate the eigenvectors and eigenvalues
    D, U = la.eigh(S)
   
    # Sort the eigenvalues and eigenvectors
    inds = np.argsort(D)
    D = D[inds[::-1]]
    U = U[:, inds[::-1]]
    
    # Form the feature vector
    U_reduction = U[:, :m]
    Z = np.dot(U_reduction.T, X_zero_mean)
    
    return Z, U, D


def gammaidx(X, k):
    ''' Calculate gamma index for each data point

    Usage: 
        y = gammaidx(X, k)
    Parameters:
        X: (d x n)  array of column vectors
        k: (k)      a scalar used as the number of neighbours
    Returns:
        y: (1 x n)  1D array of gamma index for each data point
    '''
    # Calculate distance matrix
    D = distmat(X, X)
    
    # Sort the distance matrix
    D = np.sort(D, axis=0)

    # Take only the k-nearest
    D = D[:k, :] # do not use the distance between the point and itself

    y = np.mean(D, axis=0, keepdims=True)

    return y

def distmat(M, X):
    ''' calculates a distance matrix between M and X
        
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
    ''' your header here!
    '''

    d, n = X.shape

    # Step 1: Finding the nearest neighbours
    print 'Step 1: Finding the nearest neighbours by rule ' + n_rule
    
    # Calculate distance matrix
    DM = distmat(X, X)

    if n_rule == 'knn':
        
        # Sort and get the k-nearest neighbours for each data point
        neighbours = np.argsort(DM, axis=0)[:param, :] 

    elif n_rule == 'eps-ball':
        
        print 'eps-ball'


    else:
        print 'Rule is unknown. Please choose either knn or eps-ball'

    # Step 2: local reconstruction weights
    print 'Step 2: local reconstruction weights'
    
    if param <= d:
        tol = 0
        
    
    W = np.zeros([param, n], dtype=float)                       # Matrix W to store the weights
    for i in range(n):
        Z = X[:, neighbours[:, i]] - X[:, i][:, np.newaxis]     # Create matrix Z consisting of all neighbours of Xi  
        C = np.dot(Z.T, Z)                                      # Compute local covariance        
        C = C + (np.eye(param, param) * tol * np.trace(C))      # Regularization if param > d
        W[:, i] = (la.solve(C, np.ones([param, 1]))).T          # Solve C * Wi = 1
        W[:, i] = W[:, i] / np.sum(W[:, i])                     # Enforce sum(Wi) = 1
        

    # Step 3: compute embedding
    print 'Step 3: compute embedding'
    M = np.eye(n, n, dtype=float)
    for i in range(n):
        w = W[:, i]
        j = neighbours[:, i]

        M[i, j] -= w
        M[j, i] -= w
        w = w[:, np.newaxis]
        M[np.ix_(j, j)] += np.dot(w, w.T)


    E, V = la.eig(M)
    inds = np.argsort(E)
    E = E[inds]
    V = V[:, inds]

    Y = V[:, 1:(m+1)].T * np.sqrt(n)

    return Y


# Various helper functions
def load_data(dataset='usps'):
    ''' Load various dataset '''

    if dataset == 'usps':
        data = np.load('usps.npz')
        images = data['data_patterns']
        images = images - np.min(images)
        images = vec2im(images, (16, 16))
        labels = data['data_labels']
        return images.astype('float'), labels
    if dataset == 'banana_data':
        print 'Banana'

    if dataset == 'flatroll':
        X = np.load('flatroll_data.npz')
        data = X['Xflat']
        true_emb = X['true_embedding']

    if dataset == 'fishbowl':
        X = np.load('fishbowl_data.npz')
        data = X['x_noisefree']
        true_emb = X['z']

    if dataset == 'fishbowl_dense':
        X = np.load('fishbowl_dense.npz')
        data = X['X']
        true_emb = None

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

