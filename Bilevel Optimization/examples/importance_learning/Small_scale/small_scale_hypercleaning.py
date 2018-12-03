# -*- coding: utf-8 -*-
"""
Created on Fri May 04 08:49:55 2018

@author: Akshay
"""

## test_cw_mnist.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import sys
sys.path.append('/media/data/akshay/Bilevel Optimization/methods')

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
from bilevel_penalty_mt_keras import bilevel_mt
import time
from cleverhans.utils_tf import model_loss
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D

# MNIST-specific dimensions
img_rows = 28
img_cols = 28
channels = 1

# Get MNIST data
X_train_all, Y_train_all, X_test_all, Y_test_all = data_mnist(train_start=0, train_end=60000, test_start=0, test_end=10000)
only_y = []
for i in range(len(Y_train_all)):
    a = np.argmax(Y_train_all[i])
    only_y.append(a)

def balanced_subsample(y, s):
    """Return a balanced subsample of the population"""
    sample = []
    # For every label in the dataset
    for label in np.unique(y):
        print(label)
        # Get the index of all images with a specific label
        images = np.where(y==label)[0]
        # Draw a random sample from the images
        random_sample = np.random.choice(images, size=s, replace=False)
        # Add the random sample to our subsample list
        sample += random_sample.tolist()
    return sample

# Pick 20000 samples per class from the training samples
train_samples = balanced_subsample(only_y, 2000)
train_samples = np.array(train_samples)
shuff_ind = np.arange(20000)
np.random.shuffle(shuff_ind)
print(shuff_ind)
train_samples = train_samples[shuff_ind[:]]

train = 5000
val = 5000
test = 10000

X_train = X_train_all[train_samples[:train]]
Y_train = Y_train_all[train_samples[:train]]
X_val = X_train_all[train_samples[train:train + val]]
Y_val = Y_train_all[train_samples[train:train + val]]
X_test = X_train_all[train_samples[train + val:]]
Y_test = Y_train_all[train_samples[train + val:]]

print("Training", X_train.shape, Y_train.shape)
print("Val", X_val.shape, Y_val.shape)
print("Test", X_test.shape, Y_test.shape)

input_dim = 784
X_train = X_train.reshape(train, input_dim)
X_val = X_val.reshape(val, input_dim)
X_test = X_test.reshape(test, input_dim)

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

corrupt = 2500
correct_points = {-1}
for i in range(corrupt):
    a = np.argmax(Y_train[i])
    Y_train[i] = np.zeros(10)
    j = np.random.randint(0, 10)
    Y_train[i][j] = 1
    if a == j:
        correct_points.add(i)

for i in range(corrupt, train):
    correct_points.add(i)

correct_points.remove(-1)    
print(len(correct_points))

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_val.npy", X_val)
np.save("Y_val.npy", Y_val)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)

Keras_model = Sequential()
Keras_model.add(Dense(10, input_dim = input_dim))
Keras_model.add(Activation('softmax'))
Keras_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Object used to keep track of (and return) key accuracies
report = AccuracyReport()
# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

# Create TF session
sess = tf.Session()
print("Created TensorFlow session.")

set_log_level(logging.DEBUG)
# Define input TF placeholder
x_tf = tf.placeholder(tf.float32, shape=(None, 784))
y_tf = tf.placeholder(tf.float32, shape=(None, 10))

# Define TF model graph
scope_model = 'mnist_classifier'
with tf.variable_scope(scope_model):  
   model = Keras_model
preds = model(x_tf)
    
var_model = model.trainable_weights      
saver_model = tf.train.Saver(var_model, max_to_keep = None)
print("Defined TensorFlow model graph.")


###########################################################################
# Training the model using TensorFlow
###########################################################################

# Train an MNIST model
train_params = {
    'nb_epochs': 100,
    'batch_size': 128,
    'learning_rate': 1E-3
}
rng = np.random.RandomState([2017, 8, 30])

if True:
    #'''
    X = X_val
    Y = Y_val
    model_train(sess, x_tf, y_tf, preds, X, Y, args=train_params, rng=rng)
    
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args=eval_params)
    #assert X_test.shape[0] == 10000 - 0, X_test.shape
    print('Test accuracy of the ORACLE: {0}'.format(accuracy))
    accuracy = model_eval(sess, x_tf, y_tf, preds, X_train, Y_train, args=eval_params)
    #assert X_test.shape[0] == 10000 - 0, X_test.shape
    print('Train accuracy of the ORACLE: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy
    
    #Y_train_init = model_argmax(sess, x_tf, preds, X_train)
    feed_dict = {x_tf: X_train}
    Y_train_init = sess.run(preds, feed_dict)
    
    importance_atan = np.ones((X_train.shape[0]))
    
    c = 0
    d = 0
    for i in range(Y_train.shape[0]):
        a = np.argwhere(Y_train[i] == 1)
        b = np.argmax(Y_train_init[i])
        e = np.max(Y_train_init[i])
        #print(b)
        if a == b:
            c += 1
            importance_atan[i] = np.arctanh(2.*0.8-1)
        else:
            d += 1
            importance_atan[i] = np.arctanh(2.*0.65-1)
    
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
    #'''
    
    lr_outer = 0.5
    
    lr_inner = 1E-4

    rho = 0 # 1E1 for non L1 radius
    sig = lr_inner
    batch_size = min(128, len(X_val))
    nepochs = 50#100#40#25
    height = 28
    width = 28
    nch = 1
    nb_classes = 10
    rho_t = 1E-2
    lamb_t = 1E0
    
    tick = time.time()
    blmt = bilevel_mt(sess, model, var_model, batch_size, lr_outer, lr_inner, height, width, nch, nb_classes, rho, sig)
    print("--- %s seconds ---" % (time.time() - tick))
    '''
    importance_atan = np.ones((X_train.shape[0]))    
    sess.run(tf.global_variables_initializer())
    #'''
    
    for epoch in range(nepochs):
        tick = time.time()        
        #nb_batches = int(np.ceil(float(Ntrain) / FLAGS.batch_size))
        nb_batches = int(np.floor(float(X_train.shape[0]) / batch_size))
        index_shuf = np.arange(X_train.shape[0])
        np.random.shuffle(index_shuf)
        
        for batch in range(nb_batches):
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
            
            print(recovered)
            print(extra_points)
            print(len(correct_points))
            #print(len(np.argwhere(importance[corrupt:] >= 0.75)))
            #print(len(np.argwhere(importance[corrupt:] < 0.75)))
            #print(len(np.argwhere(importance[:corrupt] >= 0.75)))
            #print(len(np.argwhere(importance[:corrupt] < 0.75)))
            
        if epoch % 1 == 0:
            print('acc = %f'%(model_eval(sess, x_tf, y_tf, preds, X_val, Y_val, args={'batch_size':batch_size})))
            #print('tr acc = %f'%(model_eval(sess, x_tf, y_tf, preds, X_train, Y_train, args={'batch_size':batch_size})))
            #print('epoch %d: loss_inner=%f, loss_outer1=%f, loss_outer2=%f'%(epoch,lin,lout1,lout2))   
                
    saver_model.save(sess,'./model_bilevel_mt_mnist.ckpt')
    #importance = 0.5*(np.tanh(importance_atan)+1.)
    np.save('./importance.npy',importance)
    