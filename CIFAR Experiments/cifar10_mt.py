# -*- coding: utf-8 -*-
"""
Created on Sun Apr 01 10:49:43 2018

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
from cleverhans.utils_tf import model_train, model_eval, batch_eval, model_argmax
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

# Train an MNIST model
train_params = {
    'nb_epochs': 100,
    'batch_size': 128,
    'learning_rate': 1E-3,
    'train_dir': os.path.join(*os.path.split(os.path.join("models", "cifar"))[:-1]),
    'filename': os.path.split( os.path.join("models", "cifar"))[-1]
}


rng = np.random.RandomState([2017, 8, 30])

######## Bilevel ########## 0.8627
if True:
    #'''
    X = X_val
    Y = Y_val
    model_train(sess, x_tf, y_tf, preds, X, Y, args=train_params, save=os.path.exists("models"), rng=rng)
    
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args=eval_params)
    #assert X_test.shape[0] == 10000 - 0, X_test.shape
    print('Test accuracy of the ORACLE: {0}'.format(accuracy))
    accuracy = model_eval(sess, x_tf, y_tf, preds, X_train, Y_train, args=eval_params)
    #assert X_test.shape[0] == 10000 - 0, X_test.shape
    print('Train accuracy of the ORACLE: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy
    
    Y_train_init = []
    nb_batches = int(np.floor(float(X_train.shape[0]) / batch_size))
    for batch in range(nb_batches):
        ind = range(batch_size*batch, min(batch_size*(1+batch), X_train.shape[0]))
        if batch == 0:
            Y_train_init = model_argmax(sess, x_tf, preds, X_train[ind, :])
        else:
            Y_train_init = np.concatenate((Y_train_init, model_argmax(sess, x_tf, preds, X_train[ind, :])))
    importance_atan = np.ones((X_train.shape[0]))
    c = 0
    d = 0
    for i in range(Y_train.shape[0]):
        a = np.argwhere(Y_train[i] == 1)
        b = Y_train_init[i]
        #print(b)
        if a == b :
            c += 1
            importance_atan[i] = np.arctanh(0.9)
        else:
            d += 1
            importance_atan[i] = np.arctanh(0.5)
                          
    print(c)
    print(d)
    print(len(np.where(importance_atan > 0.5)[0]))
    print(len(np.where(importance_atan < 0.5)[0]))
    #'''
    lr_outer = 0.5
    lr_inner = 1E-4
    rho = 0
    sig = lr_inner
    batch_size = 128
    nepochs = 50
    height = 32
    width = 32
    nch = 3
    nb_classes = 10
    blmt = bilevel_mt(sess, model, var_model, batch_size, lr_outer, lr_inner, height, width, nch, nb_classes, rho, sig)
    #importance_atan = np.ones((X_train.shape[0]))    
    #sess.run(tf.global_variables_initializer())
    
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
            lin,lout1, timp_atan = blmt.train(X_train[index_shuf[ind],:], Y_train[index_shuf[ind],:], X_val[ind_val,:], Y_val[ind_val,:], importance_atan[index_shuf[ind]])
            importance_atan[index_shuf[ind]] = timp_atan

        ## Should I renormalize importance_atan?
        if True:
            importance = 0.5 * (np.tanh(importance_atan)+1.) # scale to beteen [0 1] from [-1 1]
            importance = 0.5 * X_train.shape[0] * importance / sum(importance)
            importance = np.maximum(.00001, np.minimum(.99999, importance))
            importance_atan = np.arctanh(0.99999 * (2. * importance - 1.))
            
        
        if epoch %1 == 0:
            print('epoch %d: loss_inner=%f, loss_outer1=%f'%(epoch,lin,lout1))
            print('mean ai=%f, mean I(ai>0.1)=%f'%(np.mean(importance),len(np.where(importance>0.1)[0])/np.float(X_train.shape[0])))
            print(np.mean(importance[corrupt:]))
            print(np.mean(importance[:corrupt]))
            
        if epoch % 1 == 0:
            print('acc = %f'%(model_eval(sess, x_tf, y_tf, preds, X_val, Y_val, args={'batch_size':batch_size})))
            #print('epoch %d: loss_inner=%f, loss_outer1=%f, loss_outer2=%f'%(epoch,lin,lout1,lout2))   
                
    saver_model.save(sess,'./model_bilevel_mt_cifar.ckpt')
    #importance = 0.5*(np.tanh(importance_atan)+1.)
    np.save('./importance.npy',importance)
    
######## Random ########## 0.8627
if False:
    ## Now, retrain temp model with the reduced set and evaluate accuracy
    importance = np.arange(0, X_train.shape[0])
    np.random.shuffle(importance)
    np.save("importance.npy", importance)