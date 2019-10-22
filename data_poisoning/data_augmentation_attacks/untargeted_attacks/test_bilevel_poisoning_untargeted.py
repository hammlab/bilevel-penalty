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

train_points = 1000
val_points = 1000
test_points = 8000
poisoned_points = 0

lr_u = 1E-1
lr_v = 1E-3

nepochs = 5000
niter = 20
full_epochs = 10001

rho_0 = 1E1
lamb_0 = 1E2
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
(X_train_all, Y_train_all), (X_test, Y_test) = mnist.load_data()

X_train_all = X_train_all.reshape(len(X_train_all), 784)
X_test = X_test.reshape(len(X_test), 784)

X_train_all = X_train_all.astype('float32')
X_test = X_test.astype('float32')
X_train_all /= 255.
X_test /= 255.

classes = [0,1,2,3,4,5,6,7,8,9]
nclass = len(classes)

idx_train = np.arange(len(X_train_all))
np.random.shuffle(idx_train)
idx_test = np.arange(len(X_test))
np.random.shuffle(idx_test)

X_train_all = X_train_all[idx_train]
Y_train_all = Y_train_all[idx_train]
X_test = X_test[idx_test]
Y_test = Y_test[idx_test]

Y_train_all = keras.utils.to_categorical(Y_train_all, len(classes))
Y_test = keras.utils.to_categorical(Y_test, len(classes))

## Noisy data
X_train = np.array(X_train_all[:train_points])
Y_train = np.array(Y_train_all[:train_points])

X_val = np.array(X_train_all[train_points:train_points+val_points])
Y_val = np.array(Y_train_all[train_points:train_points+val_points])

X_test = np.array(X_train_all[train_points+val_points:train_points+val_points+test_points])
Y_test = np.array(Y_train_all[train_points+val_points:train_points+val_points+test_points])


X_poisoned = np.array(X_train[:poisoned_points])
Y_poisoned = np.array(Y_train[:poisoned_points])

for i in range(len(Y_poisoned)):
    curr_class = np.argmax(Y_poisoned[i])
    Y_poisoned[i] = np.zeros(len(classes))
    
    #random label
    j = curr_class
    while j == curr_class:
        j = np.random.randint(0, len(classes))

    Y_poisoned[i][j] = 1

x_train_tf = tf.placeholder(tf.float32, shape=(None,height*width*nch))
y_train_tf = tf.placeholder(tf.float32, shape=(None,nclass))

x_val_tf = tf.placeholder(tf.float32, shape=(val_points,height*width*nch))
y_val_tf = tf.placeholder(tf.float32, shape=(val_points,nclass))

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

start_time = time.time()

for epoch in range(nepochs):
    f, gvnorm, gv_nu, lamb_g, new_X_poisoned = bl_poisoning.train_together(X_train, Y_train, X_val, Y_val, X_poisoned, Y_poisoned, X_poisoned_orig, niter)
    X_poisoned = np.array(new_X_poisoned)
    
X_train = np.concatenate([X_train, X_poisoned])
Y_train = np.concatenate([Y_train, Y_poisoned])

print("\nTest Set", poisoned_points, niter, lamb_0, rho_0)
sess.run(tf.global_variables_initializer())
bl_poisoning.train_simple(X_train, Y_train, X_test, Y_test, full_epochs)  

