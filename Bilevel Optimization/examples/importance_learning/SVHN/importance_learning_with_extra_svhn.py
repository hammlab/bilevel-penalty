# -*- coding: utf-8 -*-
"""
Created on Tue May 08 08:57:53 2018

@author: Akshay
"""

## test_cw_mnist.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import sys
sys.path.append('/media/data/akshay/Bilevel Optimization/methods')

import h5py
import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags

import logging
import os
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, tf_model_load, batch_eval, model_argmax
from cleverhans_tutorials.tutorial_models import make_basic_picklable_cnn
from bilevel_penalty_mt import bilevel_mt
import time
from cleverhans.utils_tf import model_loss

# Open the file as readonly
h5f = h5py.File('SVHN_single_grey_with_extra.h5', 'r')

# Load the training, test and validation set
X_train = h5f['X_train'][:]
Y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
Y_test = h5f['y_test'][:]
X_val = h5f['X_val'][:]
Y_val = h5f['y_val'][:]

# Close this file
h5f.close()

# We know that SVHN images have 32 pixels in each dimension
img_rows = X_train.shape[1]
img_cols = X_train.shape[2]
# Greyscale images only have 1 color channel
channels = X_train.shape[-1]

# Number of classes, one class for each of 10 digits
num_classes = Y_train.shape[1]

# Calculate the mean on the training data
train_mean = np.mean(X_train, axis=0)

# Calculate the std on the training data
train_std = np.std(X_train, axis=0)

# Subtract it equally from all splits
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean)  / train_std
X_val = (train_mean - X_val) / train_std
        
print('Training set', X_train.shape, Y_train.shape)
print('Validation set', X_val.shape, Y_val.shape)
print('Test set', X_test.shape, Y_test.shape)


corrupt = int(0.25 * X_train.shape[0])
correct_points = {-1}
for i in range(corrupt):
    a = np.argmax(Y_train[i])
    Y_train[i] = np.zeros(10)
    #j = np.random.randint(0, 10)
    j = (a + 1)%10
    Y_train[i][j] = 1
    if a == j:
        correct_points.add(i)

for i in range(corrupt, X_train.shape[0]):
    correct_points.add(i)
    
correct_points.remove(-1)
print(len(correct_points))

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_val.npy", X_val)
np.save("Y_val.npy", Y_val)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)

# Object used to keep track of (and return) key accuracies
report = AccuracyReport()
# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

# Create TF session
sess = tf.Session()
print("Created TensorFlow session.")

set_log_level(logging.DEBUG)
# Define input TF placeholder
x_tf = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
y_tf = tf.placeholder(tf.float32, shape=(None, 10))

# Define TF model graph
scope_model = 'svhn_classifier'
with tf.variable_scope(scope_model):    
    model = make_basic_picklable_cnn(input_shape = (None, 32, 32, 1))
preds = model(x_tf)
    
var_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope_model)        
saver_model = tf.train.Saver(var_model,max_to_keep=None)
print("Defined TensorFlow model graph.")

train_params = {
    'nb_epochs': 100,
    'batch_size': 4096,
    'learning_rate': 1E-3,
}

rng = np.random.RandomState([2017, 8, 30])

######## Bilevel ########## 0.8627
if True:
    '''
    X = X_val
    Y = Y_val
    model_train(sess, x_tf, y_tf, preds, X, Y, args=train_params, rng=rng)
    
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': 1024}
    accuracy = model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args=eval_params)
    #assert X_test.shape[0] == 10000 - 0, X_test.shape
    print('Test accuracy of the ORACLE: {0}'.format(accuracy))
    accuracy = model_eval(sess, x_tf, y_tf, preds, X_train, Y_train, args=eval_params)
    #assert X_test.shape[0] == 10000 - 0, X_test.shape
    print('Train accuracy of the ORACLE: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy
    
    batch_size = 1024
    Y_train_init = []
    nb_batches = int(np.floor(float(X_train.shape[0]) / batch_size))
    left = X_train.shape[0] - nb_batches * batch_size
    for batch in range(nb_batches):
        ind = range(batch_size*batch, min(batch_size*(1+batch), X_train.shape[0]))
        if batch == 0:
            Y_train_init = model_argmax(sess, x_tf, preds, X_train[ind, :])
        else:
            Y_train_init = np.concatenate((Y_train_init, model_argmax(sess, x_tf, preds, X_train[ind, :])))
    Y_train_init = np.concatenate((Y_train_init, model_argmax(sess, x_tf, preds, X_train[-left:])))
    
    importance_atan = np.ones((X_train.shape[0]))
    
    c = 0
    d = 0
    for i in range(Y_train.shape[0]):
        a = np.argwhere(Y_train[i] == 1)
        b = Y_train_init[i]
        e = np.max(Y_train_init[i])
        #print(b)
        if a == b:
            c += 1
            importance_atan[i] = np.arctanh(2.*0.8-1)#0.8
        else:
            d += 1
            importance_atan[i] = np.arctanh(2.*0.65-1)#0.65
    
    print(c)
    print(d)
    # Normalize
    importance = 0.5*(np.tanh(importance_atan)+1.)
    importance = 0.5*importance/np.mean(importance)
    importance = np.maximum(.00001,np.minimum(.99999,importance))
    importance_atan = np.arctanh(2.*importance-1.)
    np.save("importance_1.npy", importance)
    
    print(np.max(importance))
    print(np.min(importance))
    mimp_points = np.argwhere(importance >= np.max(importance)).flatten()
    print(mimp_points)
    recovered = 0
    extra_points = 0
    for imp_pts in range(len(mimp_points)):
        if mimp_points[imp_pts] in correct_points:
            recovered += 1
        else:
            extra_points += 1
    
    print(recovered)
    print(extra_points)
    print(len(correct_points))
    '''
    
    lr_outer = 3#1
    
    lr_inner = 1E-3#1E-2

    rho = 0 # 1E1 for non L1 radius
    sig = lr_inner
    batch_size = min(1024, len(X_val))
    nepochs = 10#100#40#25
    height = 32
    width = 32
    nch = 1
    nb_classes = 10
    rho_t = 1E-2
    lamb_t = 1E0
    
    tick = time.time()
    blmt = bilevel_mt(sess, model, var_model, batch_size, lr_outer, lr_inner, height, width, nch, nb_classes, rho, sig)
    print("--- %s seconds ---" % (time.time() - tick))
    #'''
    importance_atan = np.ones((X_train.shape[0])) * np.arctanh(2.*0.8-1)
    # Normalize
    importance = 0.5*(np.tanh(importance_atan)+1.)
    importance = 0.5*importance/np.mean(importance)
    importance = np.maximum(.00001,np.minimum(.99999,importance))
    importance_atan = np.arctanh(2.*importance-1.)
    np.save("importance_1.npy", importance)
    sess.run(tf.global_variables_initializer())
    #'''
    
    for epoch in range(nepochs):
        tick = time.time()        
        print("epoch")
        print(epoch)
        #nb_batches = int(np.ceil(float(Ntrain) / FLAGS.batch_size))
        nb_batches = int(np.floor(float(X_train.shape[0]) / batch_size))
        index_shuf = np.arange(X_train.shape[0])
        np.random.shuffle(index_shuf)
        print(nb_batches)
        for batch in range(nb_batches):
            if batch%200 == 0:
                print(batch)
            ind = range(batch_size*batch, min(batch_size*(1+batch), X_train.shape[0]))
            #if len(ind)<FLAGS.batch_size:
            #    ind.extend([np.random.choice(Ntrain,FLAGS.batch_size-len(ind))])
            ind_val = np.random.choice(X_val.shape[0], size=(batch_size), replace=False)
            #lin,lout1,lout2,lout3,timp_atan = blmt.train(X_train[index_shuf[ind],:], Y_train[index_shuf[ind],:], X_val[ind_val,:], Y_val[ind_val,:], importance_atan[index_shuf[ind]])
            l1, l2,l3, timp_atan = blmt.train(X_train[index_shuf[ind],:], Y_train[index_shuf[ind],:], X_val[ind_val,:], Y_val[ind_val,:], importance_atan[index_shuf[ind]],rho_t,lamb_t)
            
            importance_atan[index_shuf[ind]] = timp_atan
        print("--- %s seconds ---" % (time.time() - tick))
        
        rho_t *= 1.05
        lamb_t *= 0.99
        
        ## Should I renormalize importance_atan?
        if True:
            importance = 0.5 * (np.tanh(importance_atan)+1.) # scale to beteen [0 1] from [-1 1]
            importance = 0.5 * X_train.shape[0] * importance / sum(importance)
            importance = np.maximum(.00001, np.minimum(.99999, importance))
            importance_atan = np.arctanh(0.99999 * (2. * importance - 1.))
            
        
        if epoch %1 == 0:
            #print('epoch %d: loss_inner=%f, loss_inner2=%f, loss_outer1=%f, loss_outer3=%f'%(epoch,lin,lout1,lout2,lout3))
            print('epoch %d: rho=%f, lamb=%f, f=%f, gvnorm=%f, lamb_g=%f, total=%f'%(epoch, rho_t, lamb_t, l1, l2, l3, l1+l2+l3))
            print('mean ai=%f, mean I(ai>0.1)=%f'%(np.mean(importance),len(np.where(importance>0.1)[0])/np.float(X_train.shape[0])))
            mimp_points = np.argwhere(importance >= 0.5).flatten()
            print(mimp_points)
            recovered = 0
            extra_points = 0
            for imp_pts in range(len(mimp_points)):
                if mimp_points[imp_pts] in correct_points:
                    recovered += 1
                else:
                    extra_points += 1
            
            print("correct:", recovered)
            print("incorrect:", extra_points)
            print(len(correct_points))
            #print(len(np.argwhere(importance[corrupt:] >= 0.75)))
            #print(len(np.argwhere(importance[corrupt:] < 0.75)))
            #print(len(np.argwhere(importance[:corrupt] >= 0.75)))
            #print(len(np.argwhere(importance[:corrupt] < 0.75)))
            
        if epoch % 1 == 0:
            print('acc = %f'%(model_eval(sess, x_tf, y_tf, preds, X_val, Y_val, args={'batch_size':batch_size})))
            #print('tr acc = %f'%(model_eval(sess, x_tf, y_tf, preds, X_train, Y_train, args={'batch_size':batch_size})))
            #print('epoch %d: loss_inner=%f, loss_outer1=%f, loss_outer2=%f'%(epoch,lin,lout1,lout2))   
                
    #importance = 0.5*(np.tanh(importance_atan)+1.)
    np.save('./importance.npy',importance)
