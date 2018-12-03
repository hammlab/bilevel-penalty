# -*- coding: utf-8 -*-
"""
Created on Fri May 04 09:10:16 2018

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
from cleverhans.utils import AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_train, model_eval
from keras.models import Sequential
from keras.layers import Dense, Activation


# MNIST-specific dimensions
img_rows = 28
img_cols = 28
channels = 1

X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")
X_val = np.load("X_val.npy")
Y_val = np.load("Y_val.npy")
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")


# Object used to keep track of (and return) key accuracies
report = AccuracyReport()
# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

Keras_model = Sequential()
Keras_model.add(Dense(10, input_dim = 784))
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

importance = np.load("importance.npy")
batch_size = min(128, len(X_val))

for pct in [0, 1, 2, 3]:#, 1, 2, 3]: # percentile from top
    #noise = 1E-6*np.random.normal(size=importance.shape)
    #th = np.percentile(importance + noise,100.-pct) # percentile from bottom
    #ind = pct * X_train.shape[0] #np.where(importance + noise >=th)[0]
    ## Make sure the training starts with random initialization
    if pct == 0: #ORacle
        ind = []
        #print(np.maximum(importance), np.minimum(importance))
        X = np.concatenate((X_train[2500:], X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train[2500:], Y_val), axis = 0)
        print(len(Y))
    
    elif pct == 1: #Our
        ind = np.argwhere(importance > 0.5).flatten()
        print(ind)
        #print(np.maximum(importance), np.minimum(importance))
        X = np.concatenate((X_train[ind], X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train[ind], Y_val), axis = 0)
        print(len(Y)) 
   
    elif pct == 2: #Baseline
        ind = []
        #print(np.maximum(importance), np.minimum(importance))
        X = np.concatenate((X_train, X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train, Y_val), axis = 0)
        print(len(Y))
    
    elif pct == 3:
        ind = []
        #print(np.maximum(importance), np.minimum(importance))
        X = X_val
        print(len(X))
        Y = Y_val
        print(len(Y))
        
    else:
        importance = np.load("importance_1.npy")
        ind = np.argwhere(importance > 0.99).flatten()
        print(ind)
        #print(np.maximum(importance), np.minimum(importance))
        X = np.concatenate((X_train[ind], X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train[ind], Y_val), axis = 0)
        print(len(Y)) 
    
    #Keras_model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
    sess.run(tf.global_variables_initializer())
    model_train(sess, x_tf, y_tf, preds, X, Y, args=train_params, rng=rng)
    # Score trained model.
    #scores = Keras_model.evaluate(x_test, y_test, verbose=1)
    #print('Test loss:', scores[0])
    #print('Test accuracy:', scores[1])
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args=eval_params)
    #assert X_test.shape[0] == 10000 - 0, X_test.shape
    print('Test accuracy of the ORACLE: {0}'.format(accuracy))

