import numpy as np
import pickle
import sys, os

# load .npy files
# modify PATH accordingly
quad_te = np.load('./numpy_files/quad_te.npy')
X_init = np.load('./numpy_files/X_init.npy')
ya = np.load('./numpy_files/ya.npy')

ratio = [10,20,30,40,50,60,70,80]

for t in ratio:
    lad_f = []
    lad_e = []
    for seed in range(11):
        w_list = []
        bias_list = []

        for g in range(10):
            w = np.zeros((4950,99))
            bias = np.zeros(99)
            for i in range(99):
                # modify PATH accordingly
                with open('./pickles/l2/0.0001/titrate_l2_%s_%s/%s_%s.pkl' % (t,seed,g,i), 'rb') as f:
                    res = pickle.load(f)
                w[:, i:(i+1)] += res['w']
                bias[i] += res['bias']
            w_list.append(w)
            bias_list.append(bias)

        w = np.mean(w_list, axis=0)
        bias = np.mean(bias_list, axis=0)

        yp = X_init + bias + quad_te.dot(w)
        yp[yp<0] = 0
        ferror = np.sum(np.abs(yp - ya), axis=0)/np.sum(np.abs(ya), axis=0)
        error1 = np.abs(yp - ya)

        lad_f.append(ferror)
        lad_e.append(error1)

    dic = {
        'ferror': lad_f,
        'error': lad_e
    }
    # modify PATH accordingly
    with open('./pickles/LAD/titrate_l2_%s.pkl' % (t), 'wb') as f:
        pickle.dump(dic, f)