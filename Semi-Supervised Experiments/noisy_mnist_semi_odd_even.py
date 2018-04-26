# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:26:21 2018

@author: Akshay
"""

## test_cw_mnist.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import sys
#sys.path.append('/home/hammj/Dropbox/Research/AdversarialLearning/codes/lib/cleverhans-master')
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags

import logging
import os
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, tf_model_load, model_argmax
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from bilevel_semi import bilevel_semi
import time

# MNIST-specific dimensions
img_rows = 28
img_cols = 28
channels = 1
batch_size = 32
num_classes = 10
epochs = 25

# Get MNIST data
X_train_all, Y_train_all, X_test_all, Y_test_all = data_mnist(train_start=0, train_end=60000, test_start=0, test_end=10000)
'''
Y_train_all_new = np.zeros([Y_train_all.shape[0], num_classes])
for i in range(Y_train_all.shape[0]):
    a = np.where(Y_train_all[i] == 1)[0]
    if a%2 == 0:
        Y_train_all_new[i][0] = 1
    else:
        Y_train_all_new[i][1] = 1

Y_test_all_new = np.zeros([Y_test_all.shape[0], num_classes])
for i in range(Y_test_all.shape[0]):
    a = np.where(Y_test_all[i] == 1)[0]
    if a%2 == 0:
        Y_test_all_new[i][0] = 1
    else:
        Y_test_all_new[i][1] = 1
Y_all = Y_train_all_new[:60000]
Y_test = Y_test_all_new
'''
X_all = X_train_all[:60000]
Y_all = Y_train_all[:60000]

tr = 59900
X_train = X_all[:tr]
Y_train = Y_all[:tr]
X_val = X_all[tr:60000]
Y_val = Y_all[tr:60000]
X_test = X_test_all
Y_test = Y_test_all

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
y_tf = tf.placeholder(tf.float32, shape=(None, num_classes))

# Define TF model graph
scope_model = 'mnist_classifier'
with tf.variable_scope(scope_model):    
    model = make_basic_cnn(nb_classes = num_classes)
preds = model(x_tf)
    
var_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope_model)         
saver_model = tf.train.Saver(var_model,max_to_keep=None)
print("Defined TensorFlow model graph.")

###########################################################################
# Training the model using TensorFlow
###########################################################################

# Train an MNIST model
train_params = {
    'nb_epochs': 100,
    'batch_size': 128,
    'learning_rate': 1E-3,
    'train_dir': os.path.join(*os.path.split(os.path.join("models", "mnist"))[:-1]),
    'filename': os.path.split( os.path.join("models", "mnist"))[-1]
}

rng = np.random.RandomState([2017, 8, 30])

######## Validation only ########## 0.9785
if True:
    #'''
    X = X_val
    Y = Y_val
    print(Y.shape)
    
    model_train(sess, x_tf, y_tf, preds, X, Y, args=train_params, save=os.path.exists("models"), rng=rng)
    
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args=eval_params)
    #assert X_test.shape[0] == 10000 - 0, X_test.shape
    print('Test accuracy of the ORACLE: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy
    
    Y_train_labels = model_argmax(sess, x_tf, preds, X_train)
    c_train = 0
    for i in range(len(X_train)):
        a = np.argmax(Y_train[i])
        if a == Y_train_labels[i]:
            c_train += 1
            
    print(c_train)
    
    Y_test_labels = model_argmax(sess, x_tf, preds, X_test)
    c_test = 0
    for i in range(len(X_test)):
        a = np.argmax(Y_test[i])
        if a == Y_test_labels[i]:
            c_test += 1
            
    print(c_test)
    
    feed_dict = {x_tf: X_train}
    Y_train_init = sess.run(preds, feed_dict)
    
    c_train = 0
    for i in range(len(X_train)):
        a = np.argmax(Y_train[i])
        b = np.argmax(Y_train_init[i])
        if a == b:
            c_train += 1
            
    print(c_train)
    
    minlogit = -1E2
    importance_logit = np.maximum(np.log(Y_train_init), minlogit)
    #importance_logit = Y_train_init#np.maximum(np.log(Y_train_init),minlogit)
    #'''
    lr_outer = 1#0.5
    lr_inner = 1E-3
    #rho = 0
    sig = lr_inner
    batch_size = 64
    nepochs = 100
        
    blsemi = bilevel_semi(sess, model, var_model, batch_size, lr_outer, lr_inner, img_rows, img_cols, channels, num_classes, sig)
    #sess.run(tf.global_variables_initializer())
    
    for epoch in range(nepochs):
        tick = time.time()        
        #nb_batches = int(np.ceil(float(Ntrain) / FLAGS.batch_size))
        nb_batches = int(np.floor(float(X_train.shape[0]) / batch_size))
        index_shuf = np.arange(X_train.shape[0])
        np.random.shuffle(index_shuf)
        for batch in range(nb_batches):
            ind = range(batch_size*batch,min(batch_size*(1+batch), X_train.shape[0]))
            #if len(ind)<FLAGS.batch_size:
            #    ind.extend([np.random.choice(Ntrain,FLAGS.batch_size-len(ind))])
            ind_val = np.random.choice(X_val.shape[0], size=(batch_size), replace=False)
            lin,lout1,lout2,timp_logit = blsemi.train(X_train[index_shuf[ind],:],X_val[ind_val,:],Y_val[ind_val,:],importance_logit[index_shuf[ind],:])
            importance_logit[index_shuf[ind],:] = timp_logit

        ## Should I renormalize importance? 
        if True:
            importance_logit = np.maximum(importance_logit, minlogit)            
            #importance = np.exp(importance_logit - np.tile(np.max(importance_logit,1,keepdims=True),[1,nb_classes]))
            importance = np.exp(importance_logit)                
            importance /= np.tile(np.sum(importance,1,keepdims=True),[1, num_classes])
        
        if epoch%1==0:
            print('epoch %d: loss_inner=%f, loss_outer1=%f, loss_outer3=%f'%(epoch,lin,lout1,lout2))
            #print('min logit=%f, max logit=%f'%(np.min(importance_logit),np.max(importance_logit)))
            
        #if epoch%1==0:
        #    print('acc = %f'%(model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args={'batch_size':batch_size})))
            
    tY = np.zeros((X_train.shape[0], num_classes))
    tY[range(X_train.shape[0]), np.argmax(importance, 1)] = 1
    np.save("new_Y.npy", tY)
    
    Y_train_labels = importance
    c_train = 0
    for i in range(len(X_train)):
        a = np.argmax(Y_train[i])
        b = np.argmax(Y_train_labels[i])
        if a == b:
            c_train += 1
            
    print(c_train)