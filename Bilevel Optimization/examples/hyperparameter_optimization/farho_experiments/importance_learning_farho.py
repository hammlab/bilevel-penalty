# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:51:35 2018

@author: Akshay
"""

## test_cw_mnist.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals



import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from cleverhans.utils_mnist import data_mnist
from cleverhans_tutorials.tutorial_models import make_basic_cnn
import far_ho as far

# MNIST-specific dimensions
img_rows = 28
img_cols = 28
channels = 1

# Get MNIST data
X_train_all, Y_train_all, X_test_all, Y_test_all = data_mnist(train_start=0, train_end=60000, test_start=0, test_end=10000)

X_train_all = X_train_all[:10000]
Y_train_all = Y_train_all[:10000]

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
train_samples = balanced_subsample(only_y, 100)
train_samples = np.array(train_samples)
shuff_ind = np.arange(1000)
np.random.shuffle(shuff_ind)
#print(shuff_ind)
train_samples = train_samples[shuff_ind[:]]


X_val = X_train_all[train_samples]
Y_val = Y_train_all[train_samples]

X_train = np.delete(X_train_all, train_samples, axis=0)
Y_train =  np.delete(Y_train_all, train_samples, axis=0)

X_test = X_test_all
Y_test = Y_test_all

print(X_train_all.shape)
print("Training", X_train.shape, Y_train.shape)
print("Val", X_val.shape, Y_val.shape)
print("Test", X_test.shape, Y_test.shape)


corrupt = int(0.5 * X_train.shape[0])
correct_points = {-1}
for i in range(corrupt):
    a = np.argmax(Y_train[i])
    Y_train[i] = np.zeros(10)
    j = np.random.randint(0, 10)
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

num_features = X_train.shape[1]
num_labels = Y_train.shape[1]
lr_inner = 1E-6
lr_outer = 1E-5
alpha_i = np.ones((X_train.shape[0]))
batch_size = 512

tf.reset_default_graph()
ss = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='y')

with tf.variable_scope('model'):    
    W_conv1 = tf.Variable(np.zeros((5, 5, 1, 32), np.float32))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    W_conv2 = tf.Variable(np.zeros((5, 5, 32, 64), np.float32))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    W_fc1 = tf.Variable(np.zeros((7 * 7 * 64, 1024), np.float32))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = tf.Variable(np.zeros((1024, 10), np.float32))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

with tf.variable_scope('hy_model'):
    W_conv1_hy = tf.Variable(np.zeros((5, 5, 1, 32), np.float32), trainable = False, collections = far.HYPERPARAMETERS_COLLECTIONS)
    b_conv1_hy = tf.Variable(tf.constant(0.1, shape=[32]), trainable = False, collections = far.HYPERPARAMETERS_COLLECTIONS)
    h_conv1_hy = tf.nn.relu(tf.nn.conv2d(x, W_conv1_hy, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_hy)
    
    h_pool1_hy = tf.nn.max_pool(h_conv1_hy, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    W_conv2_hy = tf.Variable(np.zeros((5, 5, 32, 64), np.float32), trainable = False, collections = far.HYPERPARAMETERS_COLLECTIONS)
    b_conv2_hy = tf.Variable(tf.constant(0.1, shape=[64]), trainable = False, collections = far.HYPERPARAMETERS_COLLECTIONS)
    h_conv2_hy = tf.nn.relu(tf.nn.conv2d(h_pool1_hy, W_conv2_hy, strides=[1, 1, 1, 1], padding='SAME') + b_conv2_hy)
    
    h_pool2_hy = tf.nn.max_pool(h_conv2_hy, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    W_fc1_hy = tf.Variable(np.zeros((7 * 7 * 64, 1024), np.float32), trainable = False, collections = far.HYPERPARAMETERS_COLLECTIONS)
    b_fc1_hy = tf.Variable(tf.constant(0.1, shape=[1024]), trainable = False, collections = far.HYPERPARAMETERS_COLLECTIONS)
    
    h_pool2_flat_hy = tf.reshape(h_pool2_hy, [-1, 7*7*64])
    h_fc1_hy = tf.nn.relu(tf.matmul(h_pool2_flat_hy, W_fc1_hy) + b_fc1_hy)
    
    keep_prob_hy = tf.placeholder(tf.float32)
    h_fc1_drop_hy = tf.nn.dropout(h_fc1_hy, keep_prob)
    
    W_fc2_hy = tf.Variable(np.zeros((1024, 10), np.float32), trainable = False, collections = far.HYPERPARAMETERS_COLLECTIONS)
    b_fc2_hy = tf.Variable(tf.constant(0.1, shape=[10]), trainable = False, collections = far.HYPERPARAMETERS_COLLECTIONS)
    
    y_conv_hy = tf.matmul(h_fc1_drop_hy, W_fc2_hy) + b_fc2_hy

alpha = far.get_hyperparameter('alpha', tf.ones((batch_size)))

logits_inner = y_conv
g = tf.reduce_mean(tf.sigmoid(alpha) * tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits_inner))
#g = tf.reduce_sum(tf.multiply(tf.sigmoid(alpha), tf.nn.softmax_cross_entropy_with_logits(logits=logits_inner, labels=y)))/tf.reduce_sum(tf.sigmoid(alpha))
#g = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits_inner))


logits_outer = y_conv
f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits_outer))

train_prediction = tf.nn.softmax(logits_inner)

accuracy_g = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits_inner, 1)), tf.float32))
accuracy_f = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits_outer, 1)), tf.float32))

io_optim = far.GradientDescentOptimizer(lr_inner)
oo_optim = tf.train.AdamOptimizer(lr_outer) 

print('hyperparameters to optimize')
for h in far.utils.hyperparameters():
    print(h)

print('parameters to optimize')    
for h in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'):
    print(h)    

print("here")
farho = far.HyperOptimizer()
run = farho.minimize(f, oo_optim, g, io_optim, init_dynamics_dict={v: h for v, h in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model'), far.utils.hyperparameters()[:-1])})

print('Variables (or tensors) that will store the values of the hypergradients')
print(far.hypergradients())

tf.global_variables_initializer().run()

for epoch in range(10000):

    nb_batches = int(np.floor(float(X_train.shape[0]) / batch_size))
    index_shuf = np.arange(X_train.shape[0])
    np.random.shuffle(index_shuf)
    
    for batch in range(nb_batches):
        #print("batch ", batch, " of ", nb_batches)
        ind = range(batch_size*batch, min(batch_size*(1+batch), X_train.shape[0]))
        ind_val = np.random.choice(X_val.shape[0], size=(batch_size), replace=False)

        #print(X_train[index_shuf[ind]].shape)
        inner = {x: X_train[index_shuf[ind]], y:Y_train[index_shuf[ind]], alpha:alpha_i[index_shuf[ind]], keep_prob: 1.}
        outer = {x: X_val[ind_val], y:Y_val[ind_val], keep_prob: 1.}
        
        run(1, inner_objective_feed_dicts = inner, outer_objective_feed_dicts = outer)

        alpha_i[index_shuf[ind]] = alpha.eval()
        #break
    #break
    if True:
        alpha_i = 0.5 * X_train.shape[0] * alpha_i / sum(alpha_i)
        alpha_i = np.maximum(.00001, np.minimum(.99999, alpha_i))
     
    print(epoch)
    val = {x: X_val, y:Y_val, keep_prob: 0.5}
    #print('training accuracy', accuracy_g.eval(inner))
    print('validation accuracy', accuracy_f.eval(val))
    #print('g:', g.eval(inner))
    print('val loss:', f.eval(val))
    print("alpha_norm:", tf.norm(alpha).eval())
    #print("alpha:", alpha.eval())
        
    mimp_points = np.argsort(alpha_i).flatten()[corrupt:]
    #print(mimp_points)
    recovered = 0
    extra_points = 0
    for imp_pts in range(len(mimp_points)):
        if mimp_points[imp_pts] in correct_points:
            recovered += 1
        else:
            extra_points += 1
    
    print("Correct:", recovered)
    print("Incorrect:", extra_points)
    print(len(correct_points))
    
    print("\n")

np.save("importance_farho.npy", alpha_i)
#'''