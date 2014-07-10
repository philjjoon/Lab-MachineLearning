""" ps4_tests.py

(c) Felix Brockherde, TU Berlin, 2013
"""

import numpy as np
import ps4_implementation as imp
imp = reload(imp)
#import ps4_application as app
msg = []

# compute box constraints
print('Test _compute_box_constraints')
compute_box_constraints = True
C = imp.svm_smo(kernel='linear', C=1.) 
res1 = C._compute_box_constraints(0, 1, [-1, -1, 1, 1], np.array([1.0, 0.5, 0.01, 0.2]), C=1.0)
res2 = C._compute_box_constraints(0, 1, [-1, -1, 1, 1], np.array([1.0, 0.5, 0.01, 0.2]), C=10.0)
res3 = C._compute_box_constraints(0, 1, [-1, -1, 1, 1], np.array([1.0, 0.5, 0.01, 0.2]), C=1.0)
res4 = C._compute_box_constraints(1, 2, [-1, -1, 1, 1], np.array([1.0, 0.5, 0.01, 0.2]), C=1.0)
res5 = C._compute_box_constraints(1, 3, [-1, -1, 1, 1], np.array([0.01, 0.5, 0.01, 4.2]), C=1.0)
if None in [res1, res2, res3, res4, res5]:
    msg.append('_compute_box_constraints: Not implemented yet.')
    compute_box_constraints = False
else:
    if not(np.allclose(res1, [0.5, 1.0]) and np.allclose(res2, [0.0, 1.5]) and np.allclose(res3, [0.5, 1.0])):
        msg.append('_compute_box_constraints: Error in y_i == y_j case.')
        compute_box_constraints = False
    if not(np.allclose(res4, [0.0, 0.51], atol=1e-3) and np.allclose(res5, [3.7, 1.0], atol=1e-3)):
        msg.append('_compute_box_constraints: Error in y_i != y_j case.')
        compute_box_constraints = False

# compute_udpated_b
print('Test _compute_updated_b')
compute_updated_b = True
res1 = C._compute_updated_b(1.0, 2.0, 0, 1, np.array([[1.0, 0.3], [0.3, 1.0]]), [-1., +1.], [1., 0.5], [1., -1.5], 0.3, 1.2)
res2 = C._compute_updated_b(1.0, 2.0, 0, 1, np.array([[1.0, 0.3], [0.3, 1.0]]), [-1., +1.], [1., 0.5], [2., 0.5], 0.3, 1.2)
res3 = C._compute_updated_b(1.0, 2.0, 0, 1, np.array([[1.0, 0.3], [0.3, 1.0]]), [-1., +1.], [1., 0.5], [2., -1.5], 0.3, 1.2)
if None in [res1, res2, res3]:
    msg.append('_compute_updated_b: Not implemented yet.')
    compute_updated_b = False
else:
    if not(np.allclose([res1, res2, res3], [0.7, 2.0, -0.15], atol=1e-3)):
        msg.append('_compute_updated_b: Error found.')
        compute_updated_b = False

# update parameters
print('Test _update_parameters')
update_parameters = True
if compute_box_constraints and compute_updated_b:
    res1 = C._update_parameters(1.0, 0.5, 0, 1, np.array([[1.0, 1.6], [1.6, 1.0]]), np.array([-1, +1]), np.array([0.4, -0.5]), 1.2, 0.1)
    res2 = C._update_parameters(1.0, 0.5, 0, 1, np.array([[1.0, 0.3], [0.3, 1.0]]), np.array([-1, +1]), np.array([0.4, -0.5]), 1.2, 0.1)
    res3 = C._update_parameters(1.0, 2.0, 0, 1, np.array([[1.0, 0.3], [0.3, 1.0]]), np.array([-1, +1]), np.array([0.4, -0.5]), 1.2, 0.1)
    print 'res1: ', res1
    print 'res2: ', res2
    print 'res3: ', res3
    if None in [res1, res2, res3]:
        msg.append('_update_parameters: Not implemented yet.')
        update_parameters = False
    else:
        if not(res1[2] == False):
            msg.append('_update_parameters Error in kappa condition.')
            update_parameters = False
        if not(np.allclose(res2[0], [0.1, -0.8]) and np.allclose(res2[1], 2.41, atol=1e-3) and np.allclose(res3[0], [0.9, 0.0]) and np.allclose(res3[1], 2.7, atol=1e-3)):
            msg.append('_update_parameters: Error found.')
            update_parameters = False
else:
    msg.append('_update_parameters: Not tested because dependencies do not work.')
    update_parameters = False

# test all svm_smo

print('Test svm_smo')
svm_smo = True
if update_parameters:
    np.random.seed(1)
    X_tr = np.hstack((np.random.normal(size=[2, 30]), np.random.normal(size=[2, 30]) + np.array([2., 2.])[:, np.newaxis]))
    Y_tr = np.array([1] * 30 + [-1] * 30)
    X_te = np.hstack((np.random.normal(size=[2, 30]), np.random.normal(size=[2, 30]) + np.array([2., 2.])[:, np.newaxis]))
    Y_te = np.array([1] * 30 + [-1] * 30)
    C.fit(X=X_tr, Y=Y_tr)
    #print 'C.alpha_sv: ', C.alpha_sv.shape
    #print 'C.X_sv: ', C.X_sv.shape
    #print 'C.Y_sv: ', C.Y_sv.shape
    #print 'C.b: ', C.b
    C.predict(X_te)
    Y_pred = C.ypred
    print 'Y_pred: ', Y_pred
    loss = float(np.sum(np.sign(Y_te) != np.sign(Y_pred)))/float(len(Y_te))
    imp.plot_svm_2d(X_tr, Y_tr, C)
    
    if not(loss < 0.25):
        msg.append('svm_smo: Error. The loss is %.2f and should be below 0.25' % loss)
        svm_smo = False
else:
    msg.append('svm_smo: Not tested because dependencies do not work.')
    svm_smo = False
'''
# test qp
print('Test svm_qp')
svm_qp = True
C = imp.svm_qp(kernel='linear', C=1.) 
np.random.seed(1)
X_tr = np.hstack((np.random.normal(size=[2, 30]), np.random.normal(size=[2, 30]) + np.array([2., 2.])[:, np.newaxis]))
Y_tr = np.array([1] * 30 + [-1] * 30)
X_te = np.hstack((np.random.normal(size=[2, 30]), np.random.normal(size=[2, 30]) + np.array([2., 2.])[:, np.newaxis]))
Y_te = np.array([1] * 30 + [-1] * 30)
C.fit(X=X_tr, Y=Y_tr)
C.predict(X_te)
Y_pred = C.ypred

loss = float(np.sum(np.sign(Y_te) != np.sign(Y_pred)))/float(len(Y_te))
imp.plot_svm_2d(X_tr, Y_tr, C)
if not(loss < 0.25):
    msg.append('svm_qp: Error. The loss is %.2f and should be below 0.25' % loss)
    svm_qp = False
'''
if len(msg) == 0:
    print('Everything seems to work. Good job :)')
else:
    print('\n'.join(msg))
