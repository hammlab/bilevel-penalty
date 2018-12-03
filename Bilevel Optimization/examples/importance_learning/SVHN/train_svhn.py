import os
import h5py
import numpy as np
import tensorflow as tf
import logging
from cleverhans.utils import AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_train, model_eval
from cleverhans_tutorials.tutorial_models import make_basic_cnn

# Open the file as readonly
h5f = h5py.File('SVHN_single_grey.h5', 'r')

# Load the training, test and validation set
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]
X_val = h5f['X_val'][:]
y_val = h5f['y_val'][:]

# Close this file
h5f.close()

# We know that SVHN images have 32 pixels in each dimension
img_rows = X_train.shape[1]
img_cols = X_train.shape[2]
# Greyscale images only have 1 color channel
channels = X_train.shape[-1]

# Number of classes, one class for each of 10 digits
num_classes = y_train.shape[1]

# Calculate the mean on the training data
train_mean = np.mean(X_train, axis=0)

# Calculate the std on the training data
train_std = np.std(X_train, axis=0)

# Subtract it equally from all splits
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean)  / train_std
X_val = (train_mean - X_val) / train_std
        
print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)
print('Test set', X_test.shape, y_test.shape)

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
    model = make_basic_cnn(input_shape = (None, 32, 32, 1))
preds = model(x_tf)
    
var_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope_model)        
saver_model = tf.train.Saver(var_model,max_to_keep=None)
print("Defined TensorFlow model graph.")

train_params = {
    'nb_epochs': 20,
    'batch_size': 4096,
    'learning_rate': 1E-3,
}

rng = np.random.RandomState([2017, 8, 30])

X = X_train
Y = y_train
model_train(sess, x_tf, y_tf, preds, X, Y, args=train_params, rng=rng)

# Evaluate the accuracy of the MNIST model on legitimate test examples
eval_params = {'batch_size': 1024}
accuracy = model_eval(sess, x_tf, y_tf, preds, X_test, y_test, args=eval_params)
#assert X_test.shape[0] == 10000 - 0, X_test.shape
print('Test accuracy of the ORACLE: {0}'.format(accuracy))
report.clean_train_clean_eval = accuracy
