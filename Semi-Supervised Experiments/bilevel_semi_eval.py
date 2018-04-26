# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 19:45:21 2018

@author: Akshay
"""
## test_cw_mnist.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
import tensorflow as tf

import logging
import os
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import  AccuracyReport
from cleverhans.utils_tf import model_train, model_eval, tf_model_load, model_argmax
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

tr = 59000
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
tY = np.load("new_Y.npy")


for pct in [0, 1, 2, 3, 4, 5]: # percentile from top
        #noise = 1E-6*np.random.normal(size=importance.shape)
        #th = np.percentile(importance + noise,100.-pct) # percentile from bottom
        #ind = pct * X_train.shape[0] #np.where(importance + noise >=th)[0]
        ## Make sure the training starts with random initialization
        if pct != 0:
            ind = pct * 10000 
            #print(np.maximum(importance), np.minimum(importance))
            X = np.concatenate((X_train[:ind, :], X_val), axis = 0)
            print(len(X))
            Y = np.concatenate((tY[:ind], Y_val), axis = 0)
            print(len(Y)) 
        else:
            ind = 0
            #print(np.maximum(importance), np.minimum(importance))
            X = X_val
            print(len(X))
            Y = Y_val
            print(len(Y)) 
        sess.run(tf.global_variables_initializer())
        model_train(sess, x_tf, y_tf, preds, X, Y,args=train_params)
        
        print('Bilevel acc (pct=%f,N=%d) = %f'%(pct, ind, model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args={'batch_size':batch_size})))
