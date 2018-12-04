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

from cleverhans.utils_mnist import data_mnist
import far_ho as far

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
#print(shuff_ind)
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

num_features = X_train.shape[1]
num_labels = Y_train.shape[1]
lr_inner = 1E-3
lr_outer = 1E-2
alpha_i = np.ones((X_train.shape[0]))

tf.reset_default_graph()
ss = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=(None, 28**2), name='x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='y')

scope_model = 'model'
with tf.variable_scope(scope_model):
    weights = tf.Variable(np.zeros((num_features, num_labels), np.float32))

weights_hy = tf.Variable(np.zeros((num_features, num_labels), np.float32), trainable = False, 
                         collections = far.HYPERPARAMETERS_COLLECTIONS)

alpha = far.get_hyperparameter('alpha', tf.ones(X_train.shape[0]))

logits_inner = tf.matmul(x, weights)
g = tf.reduce_mean(tf.sigmoid(alpha) * tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits_inner))

logits_outer = tf.matmul(x, weights)
f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits_outer))

train_prediction = tf.nn.softmax(logits_inner)

accuracy_g = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits_inner, 1)), tf.float32))
accuracy_f = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits_outer, 1)), tf.float32))

io_optim = far.GradientDescentOptimizer(lr_inner)
oo_optim = tf.train.AdamOptimizer(lr_outer) 

print('hyperparameters to optimize')
for h in far.hyperparameters():
    print(h)

print('parameters to optimize')    
for h in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_model):
    print(h)    

print("here")
farho = far.HyperOptimizer()
run = farho.minimize(f, oo_optim, g, io_optim, 
                     init_dynamics_dict={v: h for v, h in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_model), 
                                                              far.utils.hyperparameters()[:1])})

print('Variables (or tensors) that will store the values of the hypergradients')
print(far.hypergradients())

tf.global_variables_initializer().run()

for it in range(1000):
    
    inner = {x: X_train, y:Y_train, alpha: alpha_i}
    outer = {x: X_val, y:Y_val}
    test =  {x: X_test, y:Y_test}
    
    run(20, inner_objective_feed_dicts = inner, outer_objective_feed_dicts = outer)    
    
    alpha_i = alpha.eval()
    
    if (it % 100 == 0):
        print(it)
        print('training accuracy', accuracy_g.eval(inner))
        print('validation accuracy', accuracy_f.eval(outer))
        print('g:', g.eval(inner))
        print('f:', f.eval(outer))
        print("alpha_norm:", tf.norm(alpha).eval())
        print("alpha:", alpha.eval())
        
        mimp_points = np.argsort(alpha_i).flatten()[corrupt:]
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
        
        print("\n")
        
np.save("importance_farho.npy", alpha_i)