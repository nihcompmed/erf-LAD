import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, initializers, optimizers
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

t = int(sys.argv[1])
seed = int(sys.argv[2])

# load .npy files
# modify PATH accordingly
quad_te = np.load('./numpy_files/quad_te.npy')
X_init = np.load('./numpy_files/X_init.npy')
ya = np.load('./numpy_files/ya.npy')

quad = np.load('./numpy_files/quad_tv.npy')
y = np.load('./numpy_files/y_tv.npy')

ratio = 5470*t/100
n = np.round(ratio/0.9)

np.random.seed(seed)
cells_tv = np.random.choice(5470, size=int(n), replace=False)
ind = np.hstack([cells_tv+(5470*i) for i in range(5)])

quad_tv = np.copy(quad[ind])
y_tv = np.copy(y[ind])

kfold = KFold(n_splits=10, shuffle=True, random_state=0)
pred_list=[]
diff_p_list=[]
ferror_list=[]
error_list=[]
hist_list=[]

for (cell_tr, cell_v) in (kfold.split(range(quad_tv.shape[0]))):
    model = Sequential()
    model.add(Dense(10000, activation='linear', input_dim=4950, kernel_initializer=initializers.glorot_uniform(seed=0), bias_initializer=initializers.Zeros()))
    model.add(Dense(99, activation='linear', kernel_initializer=initializers.glorot_uniform(seed=1), bias_initializer=initializers.Zeros()))

    sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_absolute_error', optimizer=sgd)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=0)
    # modify PATH accordingly
    mc = ModelCheckpoint(('./savedmodels/mae/titrate_%s_%s.h5' % (str(t), str(seed))), monitor='val_loss', mode='min', verbose=0, save_best_only=True)

    hist = model.fit(quad_tv[cell_tr], y_tv[cell_tr], batch_size=32, epochs=5000, validation_data=(quad_tv[cell_v], y_tv[cell_v]), verbose=0, callbacks=[es,mc])
    # modify PATH accordingly
    saved_model = load_model('./savedmodels/mae/titrate_%s_%s.h5' % (str(t), str(seed)))

    diff_p = saved_model.predict(quad_te)
    yp = X_init + diff_p
    yp[yp<0] = 0

#    ferror = np.sum(np.abs(yp - ya), axis=0)/np.sum(np.abs(ya), axis=0)
#    error = np.abs(yp - ya)
    
    pred_list.append(yp)
#    ferror_list.append(ferror)
#    error_list.append(error)

yp = np.mean(pred_list, axis=0)
ferror = np.sum(np.abs(yp - ya), axis=0)/np.sum(np.abs(ya), axis=0)
error = np.abs(yp - ya)
ferror_list.append(ferror)
error_list.append(error)

res = {'pred': pred_list,
       'ferror': ferror_list,
       'error': error_list
        }
# modify PATH accordingly
with open('./pickles/keras/mae/titrate_%s_%s.pkl' % (str(t), str(seed)), 'wb') as f:
    pickle.dump(res,f)
