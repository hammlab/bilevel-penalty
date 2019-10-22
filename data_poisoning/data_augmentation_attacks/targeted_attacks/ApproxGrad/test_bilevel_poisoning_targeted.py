import numpy as np
import tensorflow as tf
from bilevel_poisoning_approxgrad_targeted import bilevel_poisoning
from keras.datasets import mnist
import keras
import time

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
poisoned_points = 50

lr_u = 1E-3
lr_v = 1E-4
lr_p = 1E-3
sig = 1E-3

nepochs = 5001
niter1 = 20
niter2 = niter1
full_epochs = 10001

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

print(X_train.shape, X_val.shape, X_test.shape, X_poisoned.shape)

for i in range(len(Y_poisoned)):
    curr_class = Y_poisoned[i]
    assert curr_class == 3 or curr_class == 8 
    
    if curr_class == 3:
        j = 8
    elif curr_class == 8:
        j = 3
    else:
        print("\n\n\n ERROR \n\n\n")
    Y_poisoned[i] = j
    
Y_train = keras.utils.to_categorical(Y_train, nclass)
Y_val = keras.utils.to_categorical(Y_val, nclass)
Y_test = keras.utils.to_categorical(Y_test, nclass)
Y_test_full = keras.utils.to_categorical(Y_test_full, nclass)
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

var_cls =  tf.get_variable('W', shape=(height * width * nch, nclass))      


bl_poisoning = bilevel_poisoning(sess, x_train_tf, x_val_tf, x_test_tf, x_poisoned_tf, x_original_tf, y_train_tf, y_val_tf, y_test_tf, y_poisoned_tf,
                                 poisoned_points, height, width, nch,
                                 var_cls, sig)

sess.run(tf.global_variables_initializer())

X_poisoned_orig = np.array(X_poisoned)
bl_poisoning.train_simple(X_train, Y_train, X_test, Y_test, full_epochs)

for epoch in range(1, nepochs):
    fval, gval, hval, new_X_poisoned = bl_poisoning.train(X_train, Y_train, X_val, Y_val, X_poisoned, Y_poisoned, X_poisoned_orig, lr_u, lr_v, lr_p, niter1, niter2)
    X_poisoned = np.array(new_X_poisoned)
    
    if epoch % 1000 == 0 and epoch != 0:
        lr_u *= 0.5
        lr_v *= 0.5

      
new_X_train = np.concatenate([X_train, X_poisoned])
new_Y_train = np.concatenate([Y_train, Y_poisoned])

sess.run(tf.global_variables_initializer())
bl_poisoning.train_simple(new_X_train, new_Y_train, X_test, Y_test, full_epochs) 