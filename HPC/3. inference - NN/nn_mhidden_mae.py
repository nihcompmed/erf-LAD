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

n_layer = int(sys.argv[1])
n_node = int(sys.argv[2])

# modify PATH accordingly
quad_tv = np.load('./numpy_files/quad_tv.npy')
y_tv = np.load('./numpy_files/y_tv.npy')

quad_te = np.load('./numpy_files/quad_te.npy')
X_init = np.load('./numpy_files/X_init.npy')
ya = np.load('./numpy_files/ya.npy')

kfold = KFold(n_splits=10, shuffle=True, random_state=0)
pred_list=[]
diff_p_list=[]
ferror_list=[]
error_list=[]
hist_list=[]

for (cell_tr, cell_v) in (kfold.split(range(5470))):
    tr = np.hstack([cell_tr+(5470*i) for i in range(5)])
    v = np.hstack([cell_v+(5470*i) for i in range(5)])
    quad_tr, y_tr = quad_tv[tr], y_tv[tr]
    quad_v, y_v = quad_tv[v], y_tv[v]

    model = Sequential()
    model.add(Dense(n_node, activation='linear', input_dim=4950, kernel_initializer=initializers.glorot_uniform(seed=0), bias_initializer=initializers.Zeros()))
    for i in range(n_layer):
        model.add(Dense(n_node, activation='linear', kernel_initializer=initializers.glorot_uniform(seed=(i+1)), bias_initializer=initializers.Zeros()))
    model.add(Dense(99, activation='linear', kernel_initializer=initializers.glorot_uniform(seed=n_layer+1), bias_initializer=initializers.Zeros()))

    sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_absolute_error', optimizer=sgd)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=0)
    # modify PATH accordingly
    mc = ModelCheckpoint(('./savedmodels/mae/layer%s_%s.h5' % (str(n_layer+1), str(n_node))), monitor='val_loss', mode='min', verbose=0, save_best_only=True)

    hist = model.fit(quad_tr, y_tr, batch_size=32, epochs=5000, validation_data=(quad_v, y_v), verbose=0, callbacks=[es,mc])
    # modify PATH accordingly
    saved_model = load_model('./savedmodels/mae/layer%s_%s.h5' % (str(n_layer+1), str(n_node)))

    diff_p = saved_model.predict(quad_te)
    yp = X_init + diff_p
    yp[yp<0] = 0

    ferror = np.sum(np.abs(yp - ya), axis=0)/np.sum(np.abs(ya), axis=0)
    error = np.abs(yp - ya)

    pred_list.append(yp)
    ferror_list.append(ferror)
    error_list.append(error)

yp = np.mean(pred_list, axis=0)
ferror = np.sum(np.abs(yp - ya), axis=0)/np.sum(np.abs(ya), axis=0)
error = np.abs(yp - ya)

pred_list.append(yp)
ferror_list.append(ferror)
error_list.append(error)

res = {'pred': yp,
        'ferror': ferror,
        'error': error,
        }

#res = {'pred': pred_list,
#        'ferror': ferror_list,
#        'error': error_list,
#        }

# modify PATH accordingly
with open('./pickles/keras/mae/layer%s_%s.pkl' % (str(n_layer+1), str(n_node)), 'wb') as f:
    pickle.dump(res,f)