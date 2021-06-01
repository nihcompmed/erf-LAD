import numpy as np
import pickle

# load .npy files
# modify PATH accordingly
X_init = np.load('./numpy_files_new/X_init.npy')
quad_te = np.load('./numpy_files_new/quad_te.npy')
ya = np.load('./numpy_files_new/ya.npy')

w_list = []
bias_list = []
ferror_list = []
error_list = []
for g in range(10):
    w = np.zeros((4950,99))
    bias = np.zeros(99)
    for j in range(99):
        i = 10*j + g
        # modify PATH accordingly
        with open('./pickles/res/%s.pkl' % (str(i)), 'rb') as f:
            res = pickle.load(f)
        w[:,j:(j+1)] += res[0]
        bias[j] += res[1]

# to compute and save the ferror and error from each CV, uncomment the below    
#         yp = X_init + bias + quad_te.dot(w)
#         yp[yp < 0] = 0
        
#         ferror = np.sum(np.abs(yp - ya), axis=0)/np.sum(np.abs(ya), axis=0)
#         error1 = np.abs(yp - ya)

#         ferror_list.append(ferror)
#         error_list.append(error1)
        
    w_list.append(w)
    bias_list.append(bias)

w = np.mean(w_list, axis=0) #average W from 10 W's
bias = np.mean(bias_list, axis=0) #average bias
w_list.append(w)
bias_list.append(bias)

yp = X_init + bias + quad_te.dot(w)
yp[yp < 0] = 0
ferror = np.sum(np.abs(yp - ya), axis=0)/np.sum(np.abs(ya), axis=0)
error = np.abs(yp - ya)

ferror_list.append(ferror)
error_list.append(error)

dic = {
    'ferror': ferror,
    'error': error,
    'w': w,
    'bias': bias,
}

# modify PATH accordingly
with open('./pickles/LAD_(99,99)_tvt.pkl', 'wb') as f:
    pickle.dump(dic, f)
