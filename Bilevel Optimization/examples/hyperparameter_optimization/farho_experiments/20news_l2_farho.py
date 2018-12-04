# -*- coding: utf-8 -*-
"""
Created on Wed May 16 18:44:58 2018

@author: Akshay
"""

# load some data
from sklearn import datasets, linear_model
import tensorflow as tf
import numpy as np
import keras
import far_ho as far
import far_ho.examples as far_ex
import tensorflow.contrib.layers as tcl
import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
categories = None

def _intercept_dot(w, X, y):
    """Computes y * np.dot(X, w).
    It takes into consideration if the intercept should be fit or not.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    """
    c = 0.
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]

    z = safe_sparse_dot(X, w) + c
    return w, c, y * z

def _logistic_loss(w, X, y, alpha, sample_weight=None):
    """Computes the logistic loss.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    out : float
        Logistic loss.
    """
    w, c, yz = _intercept_dot(w, X, y)
    print(c)

    if sample_weight is None:
        sample_weight = np.ones(y.shape[0])

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(sample_weight * log_logistic(yz)) + .5 * alpha * np.dot(w, w)
    return out

data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=False, random_state=42)#, remove=remove)
data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=False, random_state=42)#, remove=remove)

target_names = data_train.target_names
y_train, y_test = data_train.target, data_test.target
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
#####
#X_train = X_train.todense()
#print("n_samples: %d, n_features: %d" % X_train_f.shape)
X_test = vectorizer.transform(data_test.data)
#####
#X_test = X_test.todense()
#print("n_samples: %d, n_features: %d" % X_test_f.shape)

#####
X_all = vstack([X_train, X_test])
#X_all = np.concatenate(([X_train, X_test]), axis = 0)

# binarize labels
y_train[data_train.target < 10] = 0
y_train[data_train.target >= 10] = 1
y_test[data_test.target < 10] = 0
y_test[data_test.target >= 10] = 1
  
Y_all = np.concatenate([y_train, y_test], axis = 0)
#####


num_features = X_train.shape[1]
num_labels = 1
lr_inner = 1E-2#1E-1 
lr_outer = 1E-1#1E-1


print("loaded")

tf.reset_default_graph()
ss = tf.InteractiveSession()

x = tf.sparse_placeholder(tf.float32)
y = tf.placeholder(tf.float32, shape=(None, num_labels), name='y')

scope_model = 'model'
with tf.variable_scope(scope_model):
    weights = tf.Variable(np.zeros((num_features, num_labels), np.float32))
    
weights_hy = tf.Variable(np.zeros((num_features, num_labels), np.float32), trainable = False, collections = far.HYPERPARAMETERS_COLLECTIONS)

alpha = far.get_hyperparameter('alpha', 0.0)

logits_inner = tf.sparse_tensor_dense_matmul(x, weights)
g1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits_inner))
g2 = 0.5 * tf.exp(alpha) * tf.reduce_mean(tf.square(weights))
g = g1 + g2

logits_outer = tf.sparse_tensor_dense_matmul(x, weights)
f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits_outer))

accuracy_g = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits_inner, 1)), tf.float32))
accuracy_f = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits_outer, 1)), tf.float32))

io_optim = far.GradientDescentOptimizer(lr_inner)
oo_optim = tf.train.AdamOptimizer(lr_outer) 

print('hyperparameters to optimize')
for h in far.hyperparameters():
    print(h)

print('parameters to optimize')    
for h in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope_model):
    print(h)    

print("here")
farho = far.HyperOptimizer()
run = farho.minimize(f, oo_optim, g, io_optim, 
                     init_dynamics_dict={v: h for v, h in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope_model), far.utils.hyperparameters()[:1])})

print('Variables (or tensors) that will store the values of the hypergradients')
print(far.hypergradients())

num_steps = 101
loss_total = np.zeros(num_steps)
time_total = np.zeros(num_steps)
times = 100
for it_idx in range(times):
    shuff_idx = np.arange(0, X_all.shape[0])
    np.random.shuffle(shuff_idx)
    
    X_all_new = X_all[shuff_idx]
    Y_all_new = Y_all[shuff_idx]
    
    val = int(X_all_new.shape[0]/3)     
     
    X_train = X_all_new[:val]
    X_train_a, X_train_b = X_train.nonzero()
    X_train_indices = []
    for tr_i in range(len(X_train_a)):
        X_train_indices.append([int(X_train_a[tr_i]), int(X_train_b[tr_i])])
    X_train_data = np.array(X_train.data , dtype=np.float32) 
    X_train_indices = np.array(X_train_indices, dtype=np.int64)
    X_train_shape = np.array(X_train.shape, dtype=np.int64)
    y_train = Y_all_new[:val]
    print("X_train_indices", X_train_indices.shape)
          
    X_val = X_all_new[val: 2*val]
    X_val_a, X_val_b = X_val.nonzero()
    X_val_indices = []
    for tr_i in range(len(X_val_a)):
        X_val_indices.append([int(X_val_a[tr_i]), int(X_val_b[tr_i])])
    X_val_data = np.array(X_val.data , dtype=np.float32) 
    X_val_indices = np.array(X_val_indices, dtype=np.int64)
    X_val_shape = np.array(X_val.shape, dtype=np.int64)
    y_val = Y_all_new[val: 2*val]
    
    X_test = X_all_new[2*val: ]
    #X_test_a, X_test_b = X_test.nonzero()
    #X_test_indices = []
    #for tr_i in range(len(X_test_a)):
    #    X_test_indices.append([int(X_test_a[tr_i]), int(X_test_b[tr_i])])
    #X_test_data = np.array(X_test.data , dtype=np.float32) 
    #X_test_indices = np.array(X_test_indices, dtype=np.int64)
    #X_test_shape = np.array(X_test.shape, dtype=np.int64)
    y_test = Y_all_new[2*val: ]
    zero_val = np.argwhere(y_test == 0).flatten()
    y_test[zero_val] = -1
    
    y_train = y_train.reshape([y_train.shape[0], 1])
    y_val = y_val.reshape([y_val.shape[0], 1])
    #y_train = keras.utils.to_categorical(y_train, num_labels)
    #y_val = keras.utils.to_categorical(y_val, num_labels)      
          
    
    tf.global_variables_initializer().run()
    alpha_i = 0.0
    cum_time = 0
    for it in range(num_steps):
        
        inner = {x: tf.SparseTensorValue(X_train_indices, X_train_data, X_train_shape), y:y_train, alpha: alpha_i}
        outer = {x: tf.SparseTensorValue(X_val_indices, X_val_data, X_val_shape), y:y_val}
        
        tick = time.time()
        run(1, inner_objective_feed_dicts=inner, outer_objective_feed_dicts=outer)
        st_time = time.time() - tick
                           
        alpha_i = alpha.eval()
        if it % 1 == 0:
            
            '''                           
            #print('training accuracy', accuracy_g.eval(inner))
            #print('validation accuracy', accuracy_f.eval(outer))
            print("it:", it)
            print('g:', g.eval(inner))
            print('f:', f.eval(outer))
            print('alpha:', alpha.eval())
            '''
            
            #clf = linear_model.LogisticRegression(solver='lbfgs', C=np.exp(-alpha.eval()), fit_intercept=False, tol=1e-22, max_iter=500)
            #clf.fit(X_train_f, y_train_f)
            #cost_1 = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_test_f, y_test_f, 0.)
            #print("Test Cost ", cost)
            
            #cost_2 = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_val_f, y_val_f, 0.)
            #print("Val Cost ", cost)
            #print('-'*50)
            
            w = weights.eval()
            w = w.flatten()
            cost_3 = _logistic_loss(w, X_test, y_test, 0.)
            
            #print(st_time, ",", cost_1, ",", cost_2, ",", alpha.eval())
            print(st_time, ",", g.eval(inner), ",", f.eval(outer), ",", alpha.eval(), ",", cost_3)
            cum_time += st_time
            loss_total[it] += cost_3
            time_total[it] += cum_time
                      
st_loss = ""
for idx in range(num_steps):
    st_loss += str(float(loss_total[idx])/times) + ", "
print(st_loss, "\n\n")
st_time = ""
for idx in range(num_steps):
    st_time += str(float(time_total[idx])/times) + ", "
print(st_time)

print("FARHO")