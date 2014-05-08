""" ps1_tests.py

Contains tests of the implementations:
- pca
- lle
- gammaidx

(c) Daniel Bartz, TU Berlin, 2013
"""
import numpy as np
import scipy.linalg as la
import pylab as pl

import ps1_implementation as imp
imp = reload(imp)


def pca_test():
    X = np.array([[ -2.133268233289599,   0.903819474847349,   2.217823388231679, -0.444779660856219,
                    -0.661480010318842,  -0.163814281248453,  -0.608167714051449,  0.949391996219125],
                  [ -1.273486742804804,  -1.270450725314960,  -2.873297536940942,   1.819616794091556,
                    -2.617784834189455,   1.706200163080549,   0.196983250752276,   0.501491995499840],
                  [ -0.935406638147949,   0.298594472836292,   1.520579082270122,  -1.390457671168661,
                    -1.180253547776717,  -0.194988736923602,  -0.645052874385757,  -1.400566775105519]])
    m = 2;
    correct_Z = np.array([  [   -0.264248351888547, 1.29695602132309, 3.59711235194654, -2.45930603721054,
                                1.33335186376208, -1.82020953874395, -0.85747383354342, -0.82618247564525],
                            [   2.25344911712941, -0.601279409451719, -1.28967825406348, -0.45229125158068,
                                1.82830152899142, -1.04090644990666, 0.213476150303194, -0.911071431421484]])

    correct_U = np.array([  [   0.365560300980795,  -0.796515907521792,  -0.481589114714573],
                            [   -0.855143149302632,  -0.491716059542403,   0.164150878733159],
                            [  0.367553887950606,  -0.351820587590931,   0.860886992351241]] )

    correct_D = np.array(   [ 3.892593483673686,   1.801314737893267,   0.356275626798182 ])
    
    Z, U, D = imp.pca(X,m)
    assert np.all( Z.shape == correct_Z.shape ), 'Matrix Z does not have the correct shape!'
    assert np.all( U.shape == correct_U.shape ), 'Matrix U does not have the correct shape!'
    assert np.all( D.shape == correct_D.shape ), 'Matrix D does not have the correct shape!'
    print 'outputs have correct shapes!'
    
    assert np.all(  np.minimum( np.abs( Z - correct_Z ), np.abs( Z + correct_Z ) )  <= 1e-6 ), 'Matrix Z is not correct!'
    assert np.all(  np.minimum( np.abs( U - correct_U ), np.abs( U + correct_U ) )  <= 1e-6 ), 'Matrix U is not correct!'
    assert np.all(  np.abs( D - correct_D )  <= 1e-6 ), 'D is not correct!'
    print 'outputs are correct!'
    
    print 'Tests for PCA passed!'
    
    print '\n(time the test takes on my notebook: approx. 500 micro seconds)'


def gammaidx_test():
    X = np.array([  [   0.5376671395461, -2.25884686100365, 0.318765239858981, -0.433592022305684, 3.57839693972576,
                        -1.34988694015652, 0.725404224946106, 0.714742903826096, -0.124144348216312, 1.40903448980048,
                        0.67149713360808, 0.717238651328838, 0.488893770311789, 0.726885133383238, 0.293871467096658,
                        0.888395631757642, -1.06887045816803, -2.9442841619949, 0.325190539456198, 1.37029854009523],
                    [   1.83388501459509, 0.862173320368121, -1.30768829630527, 0.34262446653865, 2.76943702988488,
                        3.03492346633185, -0.0630548731896562, -0.204966058299775, 1.48969760778546, 1.41719241342961,
                        -1.20748692268504, 1.63023528916473, 1.03469300991786, -0.303440924786016, -0.787282803758638,
                        -1.14707010696915, -0.809498694424876, 1.4383802928151, -0.754928319169703, -1.7115164188537]])

    k = 3;

    correct_gamma = np.array([  [   0.606051220224367, 1.61505686776722, 0.480161964450438, 1.18975154873627,
                                2.93910520141032, 2.15531724762712, 0.393996268071324, 0.30516080506303,
                                0.787481421847747, 0.895402545799062, 0.385599174039363, 0.544395897115756,
                                0.73397995201338, 0.314642851266896, 0.376994725474732, 0.501091387197748,
                                1.3579045507961, 1.96372676400505, 0.389228251829715, 0.910065898315003]])
                                
    gamma = imp.gammaidx(X, k)

    assert np.all( gamma.shape == correct_gamma.shape ), 'gamma does not have the correct shape!'
    print 'outputs have correct shapes!'
    
    assert np.all(  np.abs( gamma - correct_gamma )  <= 1e-6 ), 'gamma is not correct!'
    print 'outputs are correct!'
    
    print 'Tests for gammaidx passed!'

    print '\n(time the test takes on my notebook: approx. 250  micro seconds)'
    

def randrot(d):
    '''generate random orthogonal matrix'''
    M = 100*(np.random.rand(d,d)-0.5);
    M = 0.5*(M-M.T);
    R = la.expm(M);
    return R

    
def lle_test():
    n = 500
    print n
    Xt = 10*np.random.rand(2,n);
    X = np.append( Xt, 0.5*np.random.randn(8,n), 0 );

    
    # Rotate data randomly.
    X = np.dot(randrot(10),X);
    

    for (n_rule, param) in [  ['knn',30],['eps-ball',5],['eps-ball',0.5] ]:
    #for (n_rule, param) in [  ['knn',30]]:
        try:
            Xp = imp.lle(X, 2, n_rule, param);
        except Exception as inst:
            print 'lle crashed for n_rule = ' + n_rule + ' = ' + str(param) + ':'
            print inst.args

        pl.figure(figsize=(14,8))
        pl.subplot(1,3,1)
        pl.scatter(Xt[0,:], Xt[1,:],30)
        pl.title('True 2D manifold')
        pl.ylabel('x_2')
        pl.xticks([],[]); pl.yticks([],[])
        
        pl.subplot(1,3,2);
        pl.scatter(Xp[0,:], Xp[1,:], 30, Xt[0,:]);
        pl.title(n_rule +': embedding colored via x_1');
        pl.xlabel('x_1')
        pl.xticks([],[]); pl.yticks([],[])

        pl.subplot(1,3,3);
        pl.scatter(Xp[0,:], Xp[1,:], 30, Xt[1,:]);
        pl.title(n_rule +': embedding colored via x_2');
        pl.xticks([],[]); pl.yticks([],[])
        
    print 'Tests for LLE passed? Check the figures!'

    print '\n(time the test takes on my notebook: approx. 2.5 seconds)'

