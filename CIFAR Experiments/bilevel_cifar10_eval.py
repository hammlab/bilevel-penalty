# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 02:55:49 2018

@author: Akshay
"""

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
import tensorflow as tf
from cleverhans.utils import AccuracyReport
import logging
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_train, model_eval, batch_eval
import time
from bilevel_mt_cifar10 import bilevel_mt

batch_size = 32
num_classes = 10
epochs = 25
data_augmentation = True
num_predictions = 20

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

X_all = x_train
Y_all = y_train

tr = 49024
X_train = X_all[:tr]
Y_train = Y_all[:tr]
X_val = X_all[tr:]
Y_val = Y_all[tr:]
X_test = x_test
Y_test = y_test

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


corrupt = 36750#12250#24500#36750
print(corrupt)


for i in range(corrupt):
    Y_train[i] = np.zeros(10)
    j = np.random.randint(0, 10)
    Y_train[i][j] = 1
'''
for i in range(corrupt):
    j = np.where(Y_train[i] == 1)[0]
    #print(j)
    j = (j+1)%10
    Y_train[i] = np.zeros(10)
    Y_train[i][j] = 1

print(Y_train[0])
'''

Keras_model = Sequential()
Keras_model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
Keras_model.add(Activation('relu'))
Keras_model.add(Conv2D(32, (3, 3)))
Keras_model.add(Activation('relu'))
Keras_model.add(MaxPooling2D(pool_size=(2, 2)))
Keras_model.add(Dropout(0.25))

Keras_model.add(Conv2D(64, (3, 3), padding='same'))
Keras_model.add(Activation('relu'))
Keras_model.add(Conv2D(64, (3, 3)))
Keras_model.add(Activation('relu'))
Keras_model.add(MaxPooling2D(pool_size=(2, 2)))
Keras_model.add(Dropout(0.25))

Keras_model.add(Flatten())
Keras_model.add(Dense(512))
Keras_model.add(Activation('relu'))
Keras_model.add(Dropout(0.5))
Keras_model.add(Dense(num_classes))
Keras_model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
Keras_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
#Keras_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Object used to keep track of (and return) key accuracies
report = AccuracyReport()
# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

# Create TF session
sess = tf.Session()
print("Created TensorFlow session.")

set_log_level(logging.DEBUG)
# Define input TF placeholder
x_tf = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
y_tf = tf.placeholder(tf.float32, shape=(None, 10))

# Define TF model graph
scope_model = 'cifar_classifier'
with tf.variable_scope(scope_model):  
   model = Keras_model
preds = model(x_tf)
    
var_model = model.trainable_weights      
saver_model = tf.train.Saver(var_model, max_to_keep = None)
print("Defined TensorFlow model graph.")


###########################################################################
# Training the model using TensorFlow
###########################################################################

# Train an CIFAR model
train_params = {
    'nb_epochs': 100,
    'batch_size': 128,
    'learning_rate': 1E-3,
    'train_dir': os.path.join(*os.path.split(os.path.join("models", "cifar"))[:-1]),
    'filename': os.path.split( os.path.join("models", "cifar"))[-1]
}

importance = np.load("importance.npy")
#importance = np.arange(X_train.shape[0])
rng = np.random.RandomState([2017, 8, 30])

for pct in [0, 1, 2, 3, 4, 5]: # percentile from top
    #noise = 1E-6*np.random.normal(size=importance.shape)
    #th = np.percentile(importance + noise,100.-pct) # percentile from bottom
    #ind = pct * X_train.shape[0] #np.where(importance + noise >=th)[0]
    ## Make sure the training starts with random initialization
    if pct != 0:
        '''
        if pct == 1:
            ind = np.argsort(importance)[-pct * 500 :]
        else:
            ind = np.argsort(importance)[-pct * 500 : -(pct - 1) * 500]
        #print(np.maximum(importance), np.minimum(importance))
        X = X_train[ind,:]#np.concatenate((X_train[ind,:], X_val), axis = 0)
        print(len(X))
        print(len(np.where(ind > corrupt)[0]))
        Y = Y_train[ind,:]#np.concatenate((Y_train[ind,:], Y_val), axis = 0)
        print(len(Y))
        '''
        ind = np.argsort(importance)[-pct * 500 :]
        #print(np.maximum(importance), np.minimum(importance))
        X = np.concatenate((X_train[ind,:], X_val), axis = 0)
        print(len(X))
        print(len(np.where(ind > corrupt)[0]))
        Y = np.concatenate((Y_train[ind,:], Y_val), axis = 0)
        print(len(Y)) 
    else:
        ind = []
        #print(np.maximum(importance), np.minimum(importance))
        X = X_val
        print(len(X))
        Y = Y_val
        print(len(Y))
        
    Keras_model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

    # Score trained model.
    scores = Keras_model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])