
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
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from bilevel_mt_test import bilevel_mt
import time
from cleverhans.utils_tf import model_loss

# MNIST-specific dimensions
img_rows = 28
img_cols = 28
channels = 1

# Get MNIST data
X_train_all, Y_train_all, X_test_all, Y_test_all = data_mnist(train_start=0, train_end=60000, test_start=0, test_end=10000)


X_all = X_train_all[:60000]
Y_all = Y_train_all[:60000]

tr = 59000
X_train = X_all[:tr]
Y_train = Y_all[:tr]
X_val = X_all[tr:60000]
Y_val = Y_all[tr:60000]
X_test = X_test_all
Y_test = Y_test_all

print(Y_train[0])
corrupt = 44250#14750#29750#44250
for i in range(corrupt):
    Y_train[i] = np.zeros(10)
    j = np.random.randint(0,10)
    Y_train[i][j] = 1

'''
ind_shuff = np.arange(tr)
np.random.shuffle(ind_shuff)
X_train = X_train[ind_shuff]
Y_train = Y_train[ind_shuff]
'''
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
scope_model = 'mnist_classifier'
with tf.variable_scope(scope_model):    
    model = make_basic_cnn()
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
importance = np.load("importance.npy")
batch_size = min(1024, len(X_val))

for pct in [0, 1, 2, 3, 4, 5]: # percentile from top
        #noise = 1E-6*np.random.normal(size=importance.shape)
        #th = np.percentile(importance + noise,100.-pct) # percentile from bottom
        #ind = pct * X_train.shape[0] #np.where(importance + noise >=th)[0]
        ## Make sure the training starts with random initialization
        if pct != 0:
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
        sess.run(tf.global_variables_initializer())
        model_train(sess, x_tf, y_tf, preds, X, Y,args=train_params)
        
        print('Bilevel acc (pct=%f,N=%d) = %f'%(pct,len(ind),model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args={'batch_size':batch_size})))
