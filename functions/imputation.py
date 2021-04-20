import numpy as np
from sklearn.model_selection import KFold
import numpy.ma as ma
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, initializers
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def tf_train(data, comp_ind):
    data_ = np.copy(data)
    incomp_ind = np.array([i for i in range(data.shape[1]) if i not in set(comp_ind)])
    tr_rind = {i:np.array([j for j in range(data.shape[0])\
                                     if data[j,i] != 0]) for i in incomp_ind}
    te_rind = {i:np.array([j for j in range(data.shape[0]) if data[j,i] == 0]) for i in incomp_ind}
    printProgressBar(0, len(incomp_ind), length=50)
    for (e,i) in enumerate(incomp_ind):
        tr_in = np.copy(data[:,comp_ind][tr_rind[i]])
        tr_out = np.copy(data[tr_rind[i],i:i+1])
        tr_in, tr_out = shuffle(tr_in, tr_out, random_state=e)
        in_size = tr_in.shape[1]
        model = Sequential()
        model.add(Dense(int(in_size/2), activation='linear', input_dim=in_size, kernel_initializer=initializers.glorot_uniform(seed=0), bias_initializer=initializers.Zeros()))
        model.add(Dense(1, activation='linear', kernel_initializer=initializers.glorot_uniform(seed=0), bias_initializer=initializers.Zeros()))
        model.compile(optimizer='SGD', loss='mean_squared_error')
        
#         es = EarlyStopping(monitor='val_loss', baseline=1e-8, mode='min', verbose=0)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, restore_best_weights=True)
        mc = ModelCheckpoint(('./savedmodels/imputation/%s.h5' % (str(i))), monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        model.fit(tr_in, tr_out, batch_size=32, epochs=5000, validation_split=0.1, verbose=0, callbacks=[es,mc])
        
        saved_model = load_model('./savedmodels/imputation/%s.h5' % (str(i)))
        te_in = np.copy(data[:,comp_ind][te_rind[i]])
        data_[te_rind[i],i:i+1] = saved_model.predict(te_in)
        printProgressBar(e+1, len(incomp_ind), length=50)
    data_[data_<0] = 0
    return data_

def imputation(data, comp_ind):
    imputed_data = tf_train(data, comp_ind)
    return imputed_data

def imputation(data):
    complete_all = ([int(x) - 1 for x in open('../indices_complete.txt','r').readline().split()])
    comp_ind = list(map(int, list((np.array(complete_all)[::6]-3)/6)))
    gene_sel = np.sort(np.random.choice(comp_ind, 3, replace=False))
#     print("selected genes: ", gene_sel)
    
    for i in range(len(gene_sel)):
        mt = np.random.randint(1, 6, 1)
        tp = np.random.choice(range(1, 7), mt, replace=False)
#         print("# of missing time points: ", mt, " which time point is missing: ", tp)
        for j in tp:
            data[(6078*(j-1)):(6078*j), gene_sel[i]] = 0
    comp_ind_ = [i for i in comp_ind if i not in gene_sel]
#     print(len(comp_ind), len(comp_ind_), comp_ind_)
    
    imputed_data = lad_train(data, comp_ind_)
    return gene_sel, np.median(imputed_data, axis=0)
