# -*- coding: utf-8 -*-
"""
Created on Wed May 16 18:35:56 2018

@author: Akshay
"""

from __future__ import absolute_import, print_function, division

import far_ho as far
import tensorflow as tf
import far_ho.examples as far_ex

import os

tf.reset_default_graph()
ss = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=(None, 28**2), name='x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='y')

# load a small portion of mnist data
datasets = far_ex.mnist(folder=os.path.join(os.getcwd(), 'MNIST_DATA'), partitions=(.1, .1,))
datasets = far_ex.Datasets.from_list(datasets)

# build up a feddforward NN calssifier
import tensorflow.contrib.layers as tcl
with tf.variable_scope('model'):
    h1 = tcl.fully_connected(x, 300)
    out = tcl.fully_connected(h1, datasets.train.dim_target)
    print('Ground model weights (parameters)')
    [print(e) for e in tf.model_variables()];
with tf.variable_scope('inital_weight_model'):
    h1_hyp = tcl.fully_connected(x, 300,
                                 variables_collections=far.HYPERPARAMETERS_COLLECTIONS, 
                                trainable=False)
    out_hyp = tcl.fully_connected(h1_hyp, datasets.train.dim_target,
                                 variables_collections=far.HYPERPARAMETERS_COLLECTIONS,
                                 trainable=False)
    print('Initial model weights (hyperparameters)')
    [print(e) for e in far.utils.hyperparameters()];
     
weights = far.get_hyperparameter('ex_weights', tf.zeros(datasets.train.num_examples))

with tf.name_scope('errors'):
    tr_loss = tf.reduce_mean(tf.sigmoid(weights)*tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
    # outer objective (validation error) (not weighted)
    val_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(out, 1)), tf.float32))

# optimizers
# get an hyperparameter for the learning rate
lr = far.get_hyperparameter('lr', 0.01)
io_optim = far.GradientDescentOptimizer(lr)  # for training error minimization an optimizer from far_ho is needed
oo_optim = tf.train.AdamOptimizer()  # for outer objective optimizer all optimizers from tf are valid

print('hyperparameters to optimize')
[print(h) for h in far.hyperparameters()];

# build hyperparameter optimizer
farho = far.HyperOptimizer()
run = farho.minimize(val_loss, oo_optim, tr_loss, io_optim, 
                     init_dynamics_dict={v: h for v, h in zip(tf.model_variables(), far.utils.hyperparameters()[:4])})

print('Variables (or tensors) that will store the values of the hypergradients')
print(*far.hypergradients(), sep='\n')

# run hyperparameter optimization 
T = 200 # performs 200 iteraitons of gradient descent on the training error (rise this values for better performances)
# get data suppliers (could also be stochastic for SGD)
tr_supplier = datasets.train.create_supplier(x, y)
val_supplier = datasets.validation.create_supplier(x, y)
tf.global_variables_initializer().run()

print(tr_supplier())
a = tr_supplier()
print(a[x].shape)

print('training accuracy', accuracy.eval(tr_supplier()))
print('validation accuracy', accuracy.eval(val_supplier()))
print('-'*50)

tr_accs, val_accs = [], []
for _ in range(20):
    
    
    run(T, inner_objective_feed_dicts=tr_supplier, outer_objective_feed_dicts=val_supplier)
    tr_accs.append(accuracy.eval(tr_supplier())), val_accs.append(accuracy.eval(val_supplier()))
    print('training accuracy', tr_accs[-1])
    print('validation accuracy', val_accs[-1])
    print('learning rate', lr.eval())
    print('norm of examples weight', tf.norm(weights).eval())
    print('-'*50)

                                 