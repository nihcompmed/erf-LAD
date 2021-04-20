import sys
import scipy as sp
import numpy as np
from sklearn.model_selection import KFold
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, initializers
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle

ind = int(sys.argv[1])
node = int(sys.argv[2])

def tf_train(data, comp_ind, gene_sel):
    data_ = np.copy(data)
    incomp_ind = np.array([i for i in range(data.shape[1]) if i not in set(comp_ind)])
    tr_rind = {i:np.array([j for j in range(data.shape[0]) if data[j,i] != 0]) for i in gene_sel}
    te_rind = {i:np.array([j for j in range(data.shape[0]) if data[j,i] == 0]) for i in gene_sel}
    for (e,i) in enumerate(gene_sel):
        tr_in = np.copy(data[:,comp_ind][tr_rind[i]])
        tr_out = np.copy(data[tr_rind[i],i:i+1])
        tr_in, tr_out = shuffle(tr_in, tr_out, random_state=e)
        in_size = tr_in.shape[1]
        model = Sequential()
        model.add(Dense(node, activation='linear', input_dim=in_size, kernel_initializer=initializers.glorot_uniform(seed=0), bias_initializer=initializers.Zeros()))
        model.add(Dense(1, activation='linear', kernel_initializer=initializers.glorot_uniform(seed=0), bias_initializer=initializers.Zeros()))
        model.compile(optimizer='SGD', loss='mean_squared_error')
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=0, restore_best_weights=True)
        # modify PATH accordingly
        mc = ModelCheckpoint(('./savedmodels/imputation_n%s/%s_%s.h5' % (str(node), str(ind), str(i))), monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        hist = model.fit(tr_in, tr_out, batch_size=32, epochs=5000, validation_split=0.1, verbose=0, callbacks=[es,mc])
        # modify PATH accordingly
        saved_model = load_model('./savedmodels/imputation_n%s/%s_%s.h5' % (str(node), str(ind), str(i)))        
        te_in = np.copy(data[:,comp_ind][te_rind[i]])
        data_[te_rind[i],i:i+1] = saved_model.predict(te_in)
    return data_
    
def imputation(data):
    np.random.seed(ind)
    tp_list = []
    saved = []
    # modify PATH accordingly
    complete_all = ([int(x) - 1 for x in open('indices_complete.txt','r').readline().split()])
    comp_ind = list(map(int, list((np.array(complete_all)[::6]-3)/6)))
    gene_sel = np.sort(np.random.choice(comp_ind, 3, replace=False))

    for i in range(len(gene_sel)):
        mt = np.random.randint(1, 6, 1)
        tp = np.random.choice(range(1, 7), mt, replace=False)
        tp_list.append(tp)
        for j in tp:
            saved.append(np.copy(data[(6078*(j-1)):(6078*j), gene_sel[i]]))
            data[(6078*(j-1)):(6078*j), gene_sel[i]] = 0
    comp_ind_ = [i for i in comp_ind if i not in gene_sel]
    
    imputed_data = tf_train(data, comp_ind_, gene_sel)
    imputed = []
    for i in range(len(gene_sel)):
        for j in tp_list[i]:
            imputed.append(np.copy(imputed_data[(6078*(j-1)):(6078*j), gene_sel[i]]))

    corr = []
    begin=0
    for t in range(len(tp_list)):
        end = begin + len(tp_list[t])
        saved_ = np.hstack(saved[begin:end])
        imputed_ = np.hstack(imputed[begin:end])
        imputed_[imputed_ < 0] = 0
        corr.append(sp.stats.linregress(saved_, imputed_)[2])
        begin = end
    return gene_sel, corr, tp_list
   
# modify PATH accordingly
raw_data = np.loadtxt('dmel_data.txt').T 
gene_exp = np.copy(raw_data[:,3:])  # excluding (x,y,z) coordinates)

tr_data = np.vstack([gene_exp[:, i::6] for i in range(6)])

gene, corr, tp_list = imputation(tr_data)
res = [gene, corr, tp_list]

# modify PATH accordingly
with open('./pickles/imputation_n%s/%s.pkl' % (str(node), str(ind)), 'wb') as f:
    pickle.dump(res,f)

