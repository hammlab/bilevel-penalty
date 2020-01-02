import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import time

from bilevel_poisoning import bilevel_poisoning
from keras.datasets import mnist
import keras

def get_balanced_set(X, Y, points):
    classes = len(np.unique(Y))
    num_per_class = int(points / classes)
    for i in range(classes):
        clss = np.argwhere(Y == i).flatten()
        np.random.shuffle(clss)
        clss = clss[:num_per_class]
        if i == 0:
            X_ = np.array(X[clss])
            Y_ = np.array(Y[clss])
        else:
            X_ = np.concatenate([X_, X[clss]])
            Y_ = np.concatenate([Y_, Y[clss]])
            
    idx = np.arange(len(X_))
    np.random.shuffle(idx)
    X_ = X_[idx]
    Y_ = Y_[idx]
    return X_, Y_

train_points = 1000
val_points = 4000
test_points = 5000
poisoned_points = 60

lr_u = 1E-1
lr_v = 1E-4

nepochs = 5001
niter = 20
full_epochs = 10001

rho_0 = 1E1
lamb_0 = 1E0
eps_0 = 1E0
nu_0 = 0.

c_rho = 1.1
c_lamb = 0.9
c_eps = 0.9

height = 28
width = 28
nch = 1

tf.set_random_seed(1234)
sess = tf.Session()

## Read data
(X_train_all, Y_train_all), (X_test_all, Y_test_all) = mnist.load_data()

X_train_all = X_train_all.reshape(len(X_train_all), 784)
X_test_all = X_test_all.reshape(len(X_test_all), 784)

X_train_all = X_train_all.astype('float32')
X_test_all = X_test_all.astype('float32')
X_train_all /= 255.
X_test_all /= 255.

nclass = 10

X_train, Y_train = get_balanced_set(X_train_all[:2*train_points], Y_train_all[:2*train_points], train_points)

X_val, Y_val = get_balanced_set(X_train_all[2*train_points:], Y_train_all[2*train_points:], val_points)
idx_8_val = np.argwhere(Y_val == 8).flatten()
X_val = np.array(X_val[idx_8_val])
Y_val = np.array(Y_val[idx_8_val])
Y_val[np.argwhere(Y_val == 8).flatten()] = 3

X_test_full, Y_test_full = get_balanced_set(X_test_all, Y_test_all, test_points)
idx_8_test = np.argwhere(Y_test_full == 8).flatten()
X_test = np.array(X_test_full[idx_8_test])
Y_test = np.array(Y_test_full[idx_8_test])
Y_test[np.argwhere(Y_test == 8).flatten()] = 3

idx_3 = np.argwhere(Y_train==3).flatten()
idx_8 = np.argwhere(Y_train==8).flatten()
X_poisoned = np.array(np.concatenate([X_train[idx_3[:int(poisoned_points/2)]], X_train[idx_8[:int(poisoned_points/2)]]]))
Y_poisoned = np.array(np.concatenate([Y_train[idx_3[:int(poisoned_points/2)]], Y_train[idx_8[:int(poisoned_points/2)]]]))

idx_poison = np.arange(len(X_poisoned))
np.random.shuffle(idx_poison)
X_poisoned = np.array(X_poisoned[idx_poison])
Y_poisoned = np.array(Y_poisoned[idx_poison])

for i in range(len(Y_poisoned)):
    curr_class = Y_poisoned[i]
    assert curr_class == 3 or curr_class == 8 
    
    if curr_class == 3:
        j = 8
    elif curr_class == 8:
        j = 3

    Y_poisoned[i] = j
    
Y_train = keras.utils.to_categorical(Y_train, nclass)
Y_val = keras.utils.to_categorical(Y_val, nclass)
Y_test = keras.utils.to_categorical(Y_test, nclass)
Y_poisoned = keras.utils.to_categorical(Y_poisoned, nclass)

x_train_tf = tf.placeholder(tf.float32, shape=(None,height*width*nch))
y_train_tf = tf.placeholder(tf.float32, shape=(None,nclass))

x_val_tf = tf.placeholder(tf.float32, shape=(None,height*width*nch))
y_val_tf = tf.placeholder(tf.float32, shape=(None,nclass))

x_test_tf = tf.placeholder(tf.float32, shape=(None,height*width*nch))
y_test_tf = tf.placeholder(tf.float32, shape=(None,nclass))

x_poisoned_tf = tf.placeholder(tf.float32, shape=(None,height*width*nch))
y_poisoned_tf = tf.placeholder(tf.float32, shape=(None,nclass))

x_original_tf = tf.placeholder(tf.float32, shape=(None,height*width*nch))

var_cls = tf.get_variable('W', shape=(height * width * nch, nclass))    

bl_poisoning = bilevel_poisoning(sess, x_train_tf, x_val_tf, x_test_tf, x_poisoned_tf, x_original_tf, y_train_tf, y_val_tf, y_test_tf, y_poisoned_tf,
                                 poisoned_points, height, width, nch, 
                                 var_cls, lr_u, lr_v, rho_0, lamb_0, eps_0, nu_0, c_rho, c_lamb, c_eps)

sess.run(tf.global_variables_initializer())

X_poisoned_orig = np.array(X_poisoned)
bl_poisoning.train_simple(X_train, Y_train, X_test, Y_test, full_epochs)
for epoch in range(0, nepochs):
    f, gvnorm, gv_nu, lamb_g, new_X_poisoned = bl_poisoning.train_together(X_train, Y_train, X_val, Y_val, X_poisoned, Y_poisoned, X_poisoned_orig, niter)
    X_poisoned = np.array(new_X_poisoned)
    
new_X_train = np.concatenate([X_train, X_poisoned])
new_Y_train = np.concatenate([Y_train, Y_poisoned])

sess.run(tf.global_variables_initializer())
bl_poisoning.train_simple(new_X_train, new_Y_train, X_test, Y_test, full_epochs) 
