import numpy as np
import pickle
import sys, os

t = int(sys.argv[1])
s = int(sys.argv[2])

# modify PATH accordingly
X_init = np.load('./numpy_files/X_init.npy')
quad_te = np.load('./numpy_files/quad_te.npy')
ya = np.load('./numpy_files/ya.npy')

if t == 20:
    ratio = np.array([int(5470*t_) for t_ in [0.2,0.4,0.6,0.8]])
if t == 30:
    ratio = np.array([int(5470*t_) for t_ in [0.3,0.6,0.9]])
if t == 40:
    ratio = np.array([int(5470*t_) for t_ in [0.4,0.8]])
if t == 45:
    ratio = np.array([int(5470*t_) for t_ in [0.45,0.9]])

# modify PATH accordingly
with open('./pickles/test_cells.pkl', 'rb') as f:
    cells_te = pickle.load(f)
with open('./pickles/xyz.pkl', 'rb') as f:
    xyz = pickle.load(f)
cells_tv = np.sort(np.delete(range(6078), cells_te))

n = np.round(ratio/0.9)
sorted_x = np.sort(xyz[cells_tv,0])
boundary = [sorted_x[int(n_)-1] for n_ in n]

for sec in range(s):
    w_list = []
    bias_list = []
    for g in range(10):
        w = np.zeros((4950,99))
        bias = np.zeros(99)
        for i in range(99):
            # modify PATH accordingly
            with open('./pickles/patch/patch_%s_%s_%s_%s.pkl' % (str(t), str(sec), str(g), str(i)), 'rb') as f:           
                res = pickle.load(f)
            w[:,i:(i+1)] += res[0]
            bias[i] += res[1]
        w_list.append(w)
        bias_list.append(bias)

    w = np.mean(w_list, axis=0)
    bias = np.mean(bias_list, axis=0)

    yp = X_init + bias + quad_te.dot(w)
    yp[yp < 0] = 0

    ferror = np.sum(np.abs(yp - ya), axis=0)/np.sum(np.abs(ya), axis=0)
    error1 = np.abs(yp - ya)

    if sec == 0:
        cells = np.array(range(608))[xyz[cells_te,0] <= boundary[sec]]
    if sec != 0:
        cells = np.array(range(608))[(xyz[cells_te,0] > boundary[sec-1]) & (xyz[cells_te,0] <= boundary[sec])]
    ind = np.sort(np.hstack([cells+(608*j) for j in range(5)]))
    
    X_init_within = X_init[ind]
    quad_within = quad_te[ind]
    ya_within = ya[ind]
    
    yp = X_init_within + bias + quad_within.dot(w)
    yp[yp < 0] = 0
    ferror_within = np.sum(np.abs(yp - ya_within), axis=0)/np.sum(np.abs(ya_within), axis=0)
    error_within = np.abs(yp - ya_within)
    
    dic = {
        'ferror': ferror,
        'error': error1,
        'ferror_within': ferror_within,
        'error_within': error_within
    }

    # modify PATH accordingly
    with open('./pickles/LAD/LAD_patch_%s_%s.pkl' % (str(t), str(sec)), 'wb') as f:
        pickle.dump(dic, f)
