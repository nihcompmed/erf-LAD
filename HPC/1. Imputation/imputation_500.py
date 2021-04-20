import numpy as np
import pickle

n = 500
matrix = np.ones((n,99))*-1
tp_matrix = np.ones((n,99))*-1

for i in range(n):
    with open('./pickles/imputation_tf_new/imputation_n13/%s.pkl' % i, 'rb') as f: #change folder name
        res = pickle.load(f)
    matrix[i,res[0]] = res[1]
    tp_matrix[i,res[0]] = [len(res[2][j]) for j in range(3)]


ans_mean = np.apply_along_axis(lambda v: np.mean(v[v >= 0]), 0, matrix)
ans_std = np.apply_along_axis(lambda v: np.std(v[v >= 0]), 0, matrix)
ans_mean[np.isnan(ans_mean)] = 0
ans_std[np.isnan(ans_std)] = 0

tp_mean = np.apply_along_axis(lambda v: np.mean(v[v >= 0]), 0, tp_matrix)
tp_std = np.apply_along_axis(lambda v: np.std(v[v >= 0]), 0, tp_matrix)
tp_mean[np.isnan(tp_mean)] = 0
tp_std[np.isnan(tp_std)] = 0

res_ = [ans_mean, ans_std, tp_mean, tp_std]

with open('./pickles/tf_n13.pkl', 'wb') as f: #change pickle name
    pickle.dump(res_, f)
