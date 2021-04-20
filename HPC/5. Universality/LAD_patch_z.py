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
from pathlib import Path

t = int(sys.argv[1])
s = int(sys.argv[2])
g = int(sys.argv[3])
i = int(sys.argv[4])

root2over = 1/np.sqrt(2)
erf_max = sperf(root2over)
weights_limit = sperf(1e-10)*1e10


def infer_LAD_v_l2(x, y, x_test, y_test, tol=1e-8, max_iter=5000):
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
            w_sol[:,index] = npl.solve(cov_xx+1*np.eye(s_pred),cov_xy).reshape(s_pred)
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

# modify PATH accordingly
quad = np.load('./numpy_files/quad_tv.npy')
y = np.load('./numpy_files/y_tv.npy')
with open('./pickles/test_cells.pkl', 'rb') as f:
    cells_te = pickle.load(f)
with open('./pickles/xyz.pkl', 'rb') as f:
    xyz = pickle.load(f)

cells_tv = np.sort(np.delete(range(6078), cells_te))

if t == 20:
    ratio = np.array([int(5470*t_) for t_ in [0.2,0.4,0.6,0.8]])
if t == 30:
    ratio = np.array([int(5470*t_) for t_ in [0.3,0.6,0.9]])
if t == 40:
    ratio = np.array([int(5470*t_) for t_ in [0.4,0.8]])
if t == 45:
    ratio = np.array([int(5470*t_) for t_ in [0.45,0.9]])

n = np.round(ratio/0.9)
sorted_z = np.sort(xyz[cells_tv,2])
boundary = [sorted_z[int(n_)-1] for n_ in n]

if s == 0:
    cells = np.array(range(5470))[xyz[cells_tv,2] <= boundary[s]]

if s != 0:
    cells = np.array(range(5470))[(xyz[cells_tv,2] > boundary[s-1]) & (xyz[cells_tv,2] <= boundary[s])]

ind = np.sort(np.hstack([cells+(5470*j) for j in range(5)]))

quad_tv = np.copy(quad[ind])
y_tv = np.copy(y[ind])

kfold = KFold(n_splits=10, shuffle=True, random_state=0)
for (en, (cell_tr, cell_v)) in enumerate((kfold.split(range(quad_tv.shape[0])))):
    if en == g:
        quad_tr, quad_v = quad_tv[cell_tr], quad_tv[cell_v]
        y_tr, y_v = y_tv[cell_tr], y_tv[cell_v]

w, bias = infer_LAD_v_l2(quad_tr, y_tr[:,i:(i+1)], quad_v, y_v[:,i:(i+1)])

res = [w, bias]

# modify PATH accordingly
with open('./pickles/patch_z/patch_%s_%s_%s_%s.pkl' % (str(t),str(s),str(g),str(i)), 'wb') as f:
        pickle.dump(res, f)