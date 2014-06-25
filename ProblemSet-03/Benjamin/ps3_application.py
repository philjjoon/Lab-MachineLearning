""" ps3_application.py

PUT YOUR NAME HERE:
Benjamin Pietrowicz


Write the functions
- roc_curve
- krr_app
(- roc_fun)

Write your code in the given functions stubs!


(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
#import pylab as pl
import random
import matplotlib.pyplot as pl
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.lines import Line2D
from scipy.stats import norm
import os
import sys
import pickle


import ps3_implementation as imp
imp = reload(imp)


def roc_curve(n):
    ''' Plots an analytical and an empirical ROC curve for 
    the probability distribution
    p(x|y=-1)*p(y=-1)+p(x|y=+1)*p(y=+1),
    where p(x|y=-1) ~ N(0,1) and p(x|y=+1) ~ N(2,1)
    with sample size n
    '''
    pl.close('all')
    mu_minus,mu_plus = 0,2
    def tpr(x):
        tp = norm.cdf(mu_plus-x)
        fn = norm.cdf(x-mu_plus)
        return tp/(tp+fn)
    def fpr(x):
        fp = norm.cdf(mu_minus-x)
        tn = norm.cdf(x-mu_minus)
        return fp/(fp+tn)
    rand = np.round(np.random.rand(n))
    sizeClassTwo = len(np.nonzero(rand)[0])
    sizeClassOne = n-sizeClassTwo
    samples = np.zeros((2,n))
    samples[0,rand==0] = np.random.randn(sizeClassOne)
    samples[1,rand==0] = -1
    samples[0,rand==1] = np.random.randn(sizeClassTwo) + 2 
    samples[1,rand==1] = 1
    sort = np.argsort(samples[0])
    x = np.cumsum(rand[sort])
    y = np.arange(n) -x + 1
    XY = np.hstack((np.zeros(2).reshape(-1,1),np.vstack((x,y))))
    x,y = XY
    d = np.nonzero(rand[sort]) # rand[sort] == np.diff(x), which is the right choice
    y /= sizeClassOne
    x /= sizeClassTwo
    AUC = np.sum(y[d] * 1/sizeClassTwo) # = np.trapz(y,x)
    pl.figure()
    pl.plot(x,y,label='empirical: %0.2f'%AUC)
    pl.title('ROC curves for sample size n = %d with AUC values'%n)
    pl.xlim(-0.2,1.2)
    pl.ylim(-0.2,1.2)
    ticks = [0,0.2,0.4,0.6,0.8,1]
    pl.xticks(ticks)
    pl.yticks(ticks)
    pl.grid(True)
    var = 3 # with sigma= 1, we know that in 3*sigma = 3 about 99.73% of data lies
    x = np.linspace(mu_minus-var,mu_plus+var,n)
    TPR = np.sort(tpr(x))
    FPR = np.sort(fpr(x))
    AUC = np.trapz(TPR,FPR)
    pl.plot(fpr(x),tpr(x),c='r',label='analytical: %0.2f'%AUC)
    pl.legend(loc='center right')

def roc_fun(y_true, y_pred):
    ''' not necessarily what its supposed to be,
    I did not check if that is correct, because I did not understand where to
    get the y_true from.
    '''
    d,n = y_true.shape
    ytrue = np.sign(y_true)
    ypred = np.sign(y_pred)
    ytrue[ytrue == -1] = 0
    ypred[ypred == -1] = 0
    diff = np.aps(ytrue-ypred)
    x = np.cumsum(diff)
    y = np.arange(n) -x +1
    XY = np.hstack((np.zeros(2).reshape(-1,1),np.vstack((x,y))))
    x,y = XY
    pl.plot(x,y)

def squared_error_loss(y_true, y_pred):
    ''' returns the squared error loss
    '''
    loss = np.mean( (y_true - y_pred)**2 )
    return loss
    
def krr_app(reg=False):
    ''' Applies krr to all data sets and saves the result to a file
    '''
    datasets = ['banana','diabetis','flare-solar','image','ringnorm']
    #dataset = ['image'] # for computing the results via console, the dataset was changed manually
    path = 'ps3_datasets/'
    results = dict()
    for data in datasets:
        Xtr = np.loadtxt(path+'U04_'+data+'-xtrain.dat')
        Ytr = np.loadtxt(path+'U04_'+data+'-ytrain.dat')
        Xte = np.loadtxt(path+'U04_'+data+'-xtest.dat')
        d,n = Xtr.shape
        print data, ' was loaded with %d dimensions'%d
        krr = imp.krr()
        kernels = ['gaussian','polynomial','linear']
        kernel_params = [np.logspace(-2,2,10),np.arange(10),np.arange(10)]
        tmp_results = dict()
        for i in range(len(kernels)):
            params = [ 'kernel',[kernels[i]], 'kernelparam', kernel_params[i],
                  'regularization', [0]]
            cvkrr = imp.cv(Xtr, Ytr.reshape(1,-1), krr, params, loss_function=squared_error_loss,
                    nrepetitions=2)
            cvkrr.predict(Xte)
            result = dict()
            result['cvloss'] = cvkrr.cvloss
            result['kernel'] = kernels[i]
            result['kernelparameter'] = cvkrr.kernelparameter
            result['regularization'] = cvkrr.regularization
            result['ypred'] = cvkrr.ypred
            tmp_results[i] = result
            print 'finished %s kernel on %s'%(kernels[i],data)
        CVloss = np.zeros(len(kernels))
        for i in range(len(kernels)):
            CVloss[i] = tmp_results[i]['cvloss']
        print 'CVloss for dataset %s'%data,CVloss
        results[data] = tmp_results[np.argmin(CVloss)]
    #return results # for computing the results via console
