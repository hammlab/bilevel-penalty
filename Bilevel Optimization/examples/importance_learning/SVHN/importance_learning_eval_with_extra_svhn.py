
## test_cw_mnist.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import sys
#sys.path.append('/home/hammj/Dropbox/Research/AdversarialLearning/codes/lib/cleverhans-master')

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
import time
from cleverhans.utils_tf import model_loss

# Load the training, test and validation set
X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")
X_val = np.load("X_val.npy")
Y_val = np.load("Y_val.npy")
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")


# We know that SVHN images have 32 pixels in each dimension
img_rows = X_train.shape[1]
img_cols = X_train.shape[2]
# Greyscale images only have 1 color channel
channels = X_train.shape[-1]

# Number of classes, one class for each of 10 digits
num_classes = Y_train.shape[1]
        
print('Training set', X_train.shape, Y_train.shape)
print('Validation set', X_val.shape, Y_val.shape)
print('Test set', X_test.shape, Y_test.shape)

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

###########################################################################
# Training the model using TensorFlow
###########################################################################

# Train an MNIST model
train_params = {
    'nb_epochs': 100,
    'batch_size': 4096,
    'learning_rate': 1E-3,
}

rng = np.random.RandomState([2017, 8, 30])
importance = np.load("importance.npy")
batch_size = min(1024, len(X_val))

corrupt = int(0.5 * X_train.shape[0])

for pct in [3, 4, 0, 1, 2]: # percentile from top
    #noise = 1E-6*np.random.normal(size=importance.shape)
    #th = np.percentile(importance + noise,100.-pct) # percentile from bottom
    #ind = pct * X_train.shape[0] #np.where(importance + noise >=th)[0]
    ## Make sure the training starts with random initialization
    if pct == 0:
        #ind = np.argsort(importance)[-pct * 1000 :]
        #print(np.maximum(importance), np.minimum(importance))
        X = np.concatenate((X_train[corrupt:], X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train[corrupt:], Y_val), axis = 0)
        print(len(Y)) 
    
    elif pct == 1:
        X = X_val
        print(len(X))
        Y = Y_val
        print(len(Y))
        
    elif pct == 2:
        #ind = np.argsort(importance)[-pct * 1000 :]
        #print(np.maximum(importance), np.minimum(importance))
        X = np.concatenate((X_train, X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train, Y_val), axis = 0)
        print(len(Y)) 
    
    elif pct == 3:
        #ind = np.argsort(importance)[-pct * 1000 :]
        ind = np.argwhere(importance > 0.5).flatten()
        print(ind)
        #print(np.maximum(importance), np.minimum(importance))
        X = np.concatenate((X_train[ind], X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train[ind], Y_val), axis = 0)
        print(len(Y)) 
   
    elif pct == 4:
        
        importance = np.load("importance_1.npy")
        ind = np.argwhere(importance >= np.max(importance)).flatten()
        print(ind)
        #print(np.maximum(importance), np.minimum(importance))
        X = np.concatenate((X_train[ind], X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train[ind], Y_val), axis = 0)
        print(len(Y)) 
    
        
    #Keras_model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
    sess.run(tf.global_variables_initializer())
    model_train(sess, x_tf, y_tf, preds, X, Y, args=train_params, save=os.path.exists("models"), rng=rng)
    # Score trained model.
    #scores = Keras_model.evaluate(x_test, y_test, verbose=1)
    #print('Test loss:', scores[0])
    #print('Test accuracy:', scores[1])
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': 1024}
    accuracy = model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args=eval_params)
    #assert X_test.shape[0] == 10000 - 0, X_test.shape
    print('Test accuracy of the ORACLE: {0}'.format(accuracy))