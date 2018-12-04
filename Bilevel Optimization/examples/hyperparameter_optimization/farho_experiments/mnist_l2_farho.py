# -*- coding: utf-8 -*-
"""
Created on Thu May 17 01:05:14 2018

@author: Akshay
"""
# load some data
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import far_ho as far
from sklearn.utils.extmath import safe_sparse_dot
from  scipy.misc import logsumexp
from sklearn.preprocessing import LabelBinarizer
import time

# utility function to calculate accuracy
def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu

def _multinomial_loss(w, X, Y, alpha, sample_weight):
    """Computes multinomial loss and class probabilities.

    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    loss : float
        Multinomial loss.

    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities.

    w : ndarray, shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.

    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
    alpha = alpha.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * (alpha * w * w).sum()
    p = np.exp(p, p)
    return loss, p, w

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
      
def resize_xs(xs):
    N,dim = xs.shape
    if dim != 28*28:
        return
    xs_new = np.zeros((N, 14*14),dtype='float32')
    for i in range(N):
        xi = xs[i].reshape(28,28)
        xs_new[i] = xi[::2,::2].reshape(14*14)
    return xs_new

X_train_all = mnist.train.images
X_train_all = resize_xs(X_train_all)
y_train_all = mnist.train.labels

X_test = mnist.test.images
X_test = resize_xs(X_test)

val_size = 10000
train_size = X_train_all.shape[0] - val_size


X_test = X_test
y_test = mnist.test.labels

sample_weight_test = np.ones(X_test.shape[0])
num_epochs = 500
times = 5
num_features = X_train_all.shape[1]
num_labels = 10
lr_outer = 1E-2#1E-2#3*1E-3
lr_inner = 1E-4#1E-4#1E-5

highest = 0
best = np.ones((num_features, num_labels))

tf.reset_default_graph()
ss = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=(None, num_features), name='x')
y = tf.placeholder(tf.float32, shape=(None, num_labels), name='y')
tf_entire_test_dataset = tf.constant(X_test)
tf_entire_test_labels = tf.constant(y_test)

scope_model = 'model'
with tf.variable_scope(scope_model):
    weights = tf.Variable(np.zeros((num_features, num_labels), np.float32))

weights_hy = tf.Variable(np.zeros((num_features, num_labels), np.float32), trainable = False, 
                         collections = far.HYPERPARAMETERS_COLLECTIONS)

alpha = far.get_hyperparameter('alpha', tf.ones((num_features, num_labels)))

logits_inner = tf.matmul(x, weights)
g1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits_inner))
#g1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_inner))
#g2 = tf.nn.l2_loss(tf.multiply(tf.exp(alpha), weights))
g2 = 0.5 * tf.reduce_sum(tf.multiply(tf.exp(alpha), tf.square(weights)))
g = g1 + g2

logits_outer = tf.matmul(x, weights)
f = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits_outer))
#f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_test_labels, logits=logits_outer))

# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(logits_inner)
test_prediction = tf.nn.softmax(tf.matmul(tf_entire_test_dataset, weights))

logits_test = tf.matmul(tf_entire_test_dataset, weights)
test_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf_entire_test_labels, logits=logits_test))

logits_val = tf.matmul(x, weights)
val_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_val))

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


total_loss = np.zeros(num_epochs)
total_time = np.zeros(num_epochs)
for time_t in range(times):
    print(time_t)
    shuff_idx = np.arange(0, len(X_train_all))
    np.random.shuffle(shuff_idx)
    
    X_train = X_train_all[shuff_idx]
    y_train = y_train_all[shuff_idx]
    
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]

    alpha_i = np.ones((num_features, num_labels), np.float32)
    tf.global_variables_initializer().run()
    cum_time = 0
    for epoch in range(num_epochs):
        inner = {x: X_train, y:y_train, alpha: alpha_i}
        outer = {x: X_val, y:y_val}
        test =  {x: X_test, y:y_test}
    
        tick = time.time()
        run(1, inner_objective_feed_dicts=inner, outer_objective_feed_dicts=outer)
        st_t = time.time() - tick
        cum_time += st_t
        
        
        alpha_i = alpha.eval()
        acc = accuracy(test_prediction.eval(), y_test)
        if acc > highest:
            highest = acc
        
        '''
        print(epoch)
        #print('training accuracy', accuracy_g.eval(inner))
        #print('validation accuracy', accuracy_f.eval(outer))
        print('g1:', g1.eval(inner))
        print('g2:', g2.eval(inner))
        print('Test Loss:', test_loss.eval(test))
        print('Val Loss:', val_loss.eval(outer))
        #print("Minibatch accuracy: ", accuracy(train_prediction.eval(inner), batch_labels_train))
        print("Test accuracy: ", accuracy(test_prediction.eval(test), y_test))
        print("Highest = ", highest)
        print(tf.norm(alpha).eval())
        #print(np.allclose(alpha_i, np.zeros((784, 10))))
        print("\n")
        '''
        w_fin = weights.eval()
        w_fin = w_fin.T
        w_fin = w_fin.flatten()
        loss = _multinomial_loss(w_fin, X_test, y_test, np.zeros(w_fin.size), sample_weight_test)[0]
        print(epoch, test_loss.eval(), loss, val_loss.eval(outer))
        #total_loss[epoch] += test_loss.eval()
        total_loss[epoch] += val_loss.eval(outer)
        total_time[epoch] += cum_time
                  
print(total_loss/times)
print("\n\n")
print(total_time/times)

st_loss = ""
for idx in range(len(total_loss)):
    st_loss += str(float(total_loss[idx])/times) + ", "
print(st_loss, "\n\n")
st_time = ""
for idx in range(len(total_time)):
    st_time += str(float(total_time[idx])/times) + ", "
print(st_time)
print("FARHO")        