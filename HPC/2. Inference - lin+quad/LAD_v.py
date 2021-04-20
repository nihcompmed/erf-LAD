import sys
import numpy as np
import scipy as sp
import scipy.optimize as spo
from scipy.special import erf as sperf
import numpy.linalg as npl
import numpy.random as npr
import pickle
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

k = int(sys.argv[1])

i = int(k/10) #column index
j = np.mod(k,10) #CV group

root2over = 1/np.sqrt(2)
erf_max = sperf(root2over)
weights_limit = sperf(1e-10)*1e10

def infer_LAD_v(x, y, x_test, y_test, tol=1e-8, max_iter=5000):
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape
    w_sol = 0.0*(npr.rand(s_pred,s_target) - 0.5)
    b_sol = npr.rand(1,s_target) - 0.5
    for index in range(s_target):
        error, old_error = np.inf, 0
        weights = np.ones((s_sample, 1))
        cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, ddof=0, aweights=weights.reshape(s_sample))
        cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
        counter = 0
        while np.abs(error-old_error) > tol and counter < max_iter:
            counter += 1
            old_error = np.mean(np.abs(b_sol[0,index] + x_test.dot(w_sol[:,index]) - y_test[:,index]))
            w_sol[:,index] = npl.solve(cov_xx,cov_xy).reshape(s_pred)
            b_sol[0,index] = np.mean(y[:,index]-x.dot(w_sol[:,index]))
            weights = (b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index])
            sigma = np.std(weights)
            error = np.mean(np.abs(b_sol[0,index] + x_test.dot(w_sol[:,index]) - y_test[:,index]))
            weights_eq_0 = np.abs(weights) < 1e-10
            weights[weights_eq_0] = weights_limit
            weights[~weights_eq_0] = sigma*sperf(weights[~weights_eq_0]/sigma)/weights[~weights_eq_0]
            cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, ddof=0, aweights=weights.reshape(s_sample))
            cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
    return w_sol, b_sol

# load .npy files
# modify PATH accordingly
quad = np.load('./numpy_files/quad_tv.npy')
y = np.load('./numpy_files/y_tv.npy')

# 10-fold CV
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
for (en, (cell_tr, cell_v)) in enumerate((kfold.split(range(5470)))):
    if en == j:
        tr = np.hstack([cell_tr+(5470*i) for i in range(5)])
        v = np.hstack([cell_v+(5470*i) for i in range(5)])

quad_tr, quad_v = quad[tr], quad[v]
y_tr, y_v = y[tr], y[v]

w, bias = infer_LAD_v(quad_tr, y_tr[:,i:(i+1)], quad_v, y_v[:,i:(i+1)])

res = [w, bias]

# modify PATH accordingly
with open('./pickles/res/%s.pkl' % (str(k)), 'wb') as f:
    pickle.dump(res, f)