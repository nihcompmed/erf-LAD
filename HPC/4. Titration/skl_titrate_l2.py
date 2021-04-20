# select t percent of cells and use all of their 5 timepoints
# 10 different cohorts of cells with seed = range(10)

import sys
import numpy as np
import pickle
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

t = int(sys.argv[1])

# load .npy files
# modify PATH accordingly
X_init = np.load('./numpy_files/X_init.npy')
quad_te = np.load('./numpy_files/quad_te.npy')
ya = np.load('./numpy_files/ya.npy')

quad = np.load('./numpy_files/quad_tv.npy')
y = np.load('./numpy_files/y_tv.npy')
    
def skl(X, y):
    w_list = []
    bias_list = []
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    for (ind_tr, ind_v) in (kfold.split(range(X.shape[0]))):
        quad_tr, y_tr = X[ind_tr], y[ind_tr]
        regr = linear_model.Ridge(alpha=0.0001)
        regr.fit(quad_tr, y_tr)
        
        w = regr.coef_.T
        w_list.append(w)
        
        bias = regr.intercept_
        bias_list.append(bias)
        
    w = np.mean(w_list, axis=0)
    bias = np.mean(bias_list, axis=0)
        
    yp = X_init + bias + quad_te.dot(w)
    yp[yp<0] = 0
    ferror = (np.sum(np.abs(yp - ya)**2, axis=0)/np.sum(np.abs(ya)**2, axis=0))**(1/2)
    error1 = np.abs(yp - ya)

    return ferror, error1

ratio = 5470*t/100
n = np.round(ratio/0.9)

skl_f = []
skl_e = []
for j in range(11):
    np.random.seed(j)
    cells_tv = np.random.choice(5470, size=int(n), replace=False)
    ind = np.hstack([cells_tv+(5470*i) for i in range(5)])

    quad_tv = np.copy(quad[ind])
    y_tv = np.copy(y[ind])

    f, e = skl(quad_tv, y_tv)
    skl_f.append(f)
    skl_e.append(e)

dic = {
    'ferror': skl_f,
    'error': skl_e
}

# modify PATH accordingly
with open('./pickles/skl/titrate_l2_%s.pkl' % (str(t)), 'wb') as f:
    pickle.dump(dic, f)
