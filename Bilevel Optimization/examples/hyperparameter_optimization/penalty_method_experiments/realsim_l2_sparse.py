# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:24:23 2018

@author: Akshay
"""
# load some data
from sklearn import datasets, linear_model
import tensorflow as tf
import numpy as np
import keras
import time
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from sklearn.utils.extmath import log_logistic, safe_sparse_dot
from scipy.sparse import csr_matrix
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

    if sample_weight is None:
        sample_weight = np.ones(y.shape[0])

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(sample_weight * log_logistic(yz)) + .5 * alpha * np.dot(w, w)
    return out


f = open("real-sim//real-sim", "r")
idx = 0
Y_all = []
row = []
col = []
data = []
max_col = 0
for line in f:
    entries = line.split(" ")
    Y_all.append(int(entries[0]))
    for ent in range(1, len(entries) - 1):
        entry = entries[ent]
        row.append(idx)
        item = entry.split(":")
        max_col = max(max_col, int(item[0]))
        col.append(item[0])
        data.append(item[1])
    idx+=1
    #print row
    #print col
    #print data
    #break

print "here"
row = np.array(row).astype(np.int)
col = np.array(col).astype(np.int)
data = np.array(data).astype(np.float)
X_all = csr_matrix((data,(row,col)), shape=(idx, max_col+1))
Y_all = np.array(Y_all)
minus_one = np.argwhere(Y_all == -1).flatten()
Y_all[minus_one] = 0

print X_all.shape[1]
# number of features
num_features = X_all.shape[1]
# number of target labels
num_labels = 1
# learning rate (alpha)
batch_size = int(Y_all.shape[0]/3)#64#512
# number of epochs
times = 5
lr_outer = 1E-1#1
lr_inner = 5*1E-1#1E-1
num_steps = 101
highest = 0
best = 0

#y_train = keras.utils.to_categorical(y_train, num_labels)
#y_val = keras.utils.to_categorical(y_val, num_labels)
#y_test = keras.utils.to_categorical(y_test, num_labels)      


def norm_sq_sum2(xs):
    return tf.reduce_sum([tf.reduce_sum(tf.square(t)) for t in xs])

# utility function to calculate accuracy
def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu

# initialize a tensorflow graph
graph = tf.Graph()
 
with graph.as_default():
    """
    defining all the nodes
    """
    
    tf_train_dataset = tf.sparse_placeholder(tf.float32)
    tf_train_labels_tf = tf.placeholder(tf.float32, [batch_size, num_labels])
    
    tf_val_dataset = tf.sparse_placeholder(tf.float32)
    tf_val_labels_tf = tf.placeholder(tf.float32, [batch_size, num_labels])
    
    tf_test_dataset = tf.sparse_placeholder(tf.float32)
    tf_test_labels_tf = tf.placeholder(tf.float32, [batch_size, num_labels])
    #'''
    
    rho_tf = tf.placeholder(tf.float32,[1],'rho_tf')
    lamb_tf = tf.placeholder(tf.float32,[1],'lamb_tf') 
    alpha_tf = tf.placeholder(tf.float32,[1], 'alphat_tf')
    #tf_entire_test_dataset = tf.constant(X_test)
 
    # Variables.
    weights = tf.Variable(np.zeros((num_features, num_labels), np.float32))
    
    tf_train_labels = tf.Variable(np.zeros((batch_size, num_labels), np.float32), name='train_labels')
    assign_train_labels= tf.assign(tf_train_labels, tf_train_labels_tf)
    tf_val_labels = tf.Variable(np.zeros((batch_size, num_labels), np.float32), name='val_labels')
    assign_val_labels= tf.assign(tf_val_labels, tf_val_labels_tf)
    tf_test_labels = tf.Variable(np.zeros((batch_size, num_labels), np.float32), name='test_labels')
    assign_test_labels= tf.assign(tf_test_labels, tf_test_labels_tf)
    
    
    alpha = tf.Variable(np.zeros((1), np.float32), name='alpha')
    assign_alpha = tf.assign(alpha, alpha_tf)
    rho = tf.Variable(np.zeros((1), np.float32), name='rho')
    assign_rho = tf.assign(rho, rho_tf)
    lamb = tf.Variable(np.zeros((1), np.float32), name='lamb')
    assign_lamb = tf.assign(lamb, lamb_tf)
 
    # Training computation.
    logits_inner = tf.sparse_tensor_dense_matmul(tf_train_dataset, weights)
    #g1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_inner))
    g1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_inner))
    g2 = 0.5 * tf.exp(alpha) * tf.reduce_mean(tf.square(weights))
    g = g1 + g2
    dgdv = tf.gradients(g, weights)
    
    logits_outer = tf.sparse_tensor_dense_matmul(tf_val_dataset, weights)
    #f = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf_val_labels, logits=logits_outer))
    f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_val_labels, logits=logits_outer))
 
    gvnorm = 0.5 * rho * norm_sq_sum2(dgdv) 
    lamb_g = lamb * g

    loss_u = f + gvnorm + lamb_g
    loss_v = f + gvnorm + lamb_g
    
    opt_u = tf.train.AdamOptimizer(lr_outer)
    opt_v = tf.train.AdamOptimizer(lr_inner)
    # Optimizer.
    train_u = opt_u.minimize(loss_u, var_list = alpha)
    train_v = opt_v.minimize(loss_v, var_list = weights)
    
    
    tgrad_and_var = opt_u.compute_gradients(loss_u, var_list=alpha)
    hunorm = tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var])
    
    tgrad_and_var = opt_v.compute_gradients(loss_v, var_list=weights)
    hvnorm = tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var])
    # Predictions for the training, validation, and test data.
    #train_prediction = tf.nn.softmax(logits_inner)
    #test_prediction = tf.nn.softmax(tf.matmul(tf_entire_test_dataset, weights))
    logits_test = tf.sparse_tensor_dense_matmul(tf_test_dataset, weights)
    #test_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf_test_labels, logits=logits_test)) 
    test_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_test_labels, logits=logits_test))
    
 
with tf.Session(graph=graph) as session:
    loss_total = np.zeros(num_steps)
    time_total = np.zeros(num_steps)
    for it_idx in range(times):
        # initialize weights and biases
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
        '''
        X_test_a, X_test_b = X_test.nonzero()
        X_test_indices = []
        for tr_i in range(len(X_test_a)):
            X_test_indices.append([int(X_test_a[tr_i]), int(X_test_b[tr_i])])
        X_test_data = np.array(X_test.data , dtype=np.float32) 
        X_test_indices = np.array(X_test_indices, dtype=np.int64)
        X_test_shape = np.array(X_test.shape, dtype=np.int64)
        '''
        y_test = Y_all_new[2*val: ]
        zero_val = np.argwhere(y_test == 0).flatten()
        y_test[zero_val] = -1
        
        
        y_train = y_train.reshape([y_train.shape[0], 1])
        y_val = y_val.reshape([y_val.shape[0], 1])
        #y_test = y_test.reshape([y_test.shape[0], 1])
        
        print("loaded")

        print X_train.shape
        print X_val.shape
        print y_train.shape
        print y_val.shape

        
        tf.global_variables_initializer().run()
        rho_i = np.ones((1)) * 1E0
        lamb_i = np.ones((1)) * 1E0
        alpha_i = np.ones((1)) * 0
        eps_t = 1E0
        print("Initialized")
        cum_time = 0
        for step in range(num_steps):
            
            session.run(assign_alpha, feed_dict={alpha_tf:alpha_i})
            session.run(assign_rho, feed_dict={rho_tf:rho_i})
            session.run(assign_lamb, feed_dict={lamb_tf:lamb_i})
            
            session.run(assign_train_labels, feed_dict={tf_train_labels_tf:y_train})
            session.run(assign_val_labels, feed_dict={tf_val_labels_tf:y_val})
            
            feed_dict = {tf_train_dataset: tf.SparseTensorValue(X_train_indices, X_train_data, X_train_shape), tf_val_dataset: tf.SparseTensorValue(X_val_indices, X_val_data, X_val_shape)}
            
            # run one step of computation
            tick = time.time()
            for it in range(1):
                session.run(train_u, feed_dict = feed_dict)
                session.run(train_v, feed_dict = feed_dict)
    
            st_t = time.time() - tick
                            
            #l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = session.run([f, gvnorm, lamb_g, alpha, rho, lamb, g1, g2, hunorm, hvnorm], feed_dict=feed_dict)
            l1, l2, l3, l4, l9, l10 = session.run([f, g, lamb_g, alpha, hunorm, hvnorm], feed_dict = feed_dict)
            #print l9**2 + l10**2, eps_t**2
            #print "fn", l1, l2[0], l3[0], l1+ l2[0]+ l3[0]
            if (l9**2 + l10**2 < eps_t**2):
                rho_i *= 2
                lamb_i /= 2
                eps_t /= 2
                #print "updated", rho_i
                
            alpha_i = l4
            
            #'''
            if (step % 1 == 0):
                
                w = weights.eval()
                w = w.flatten()
                cost_3 = _logistic_loss(w, X_test, y_test, 0.)
                
                #print step, ". ", st_t, ", ", cost1, ", ", l4[0], ", ", cost_2, ", ", cost_3
                #print step, ". ", st_t, ", ", acc1, ", ", cost_3, ", ", cost1, ", ", pred_2, ", ", cost_2, ", ", l4[0]
                print step, ". ", st_t, ", ", cost_3, ", ", l4[0], ",", l1
                cum_time += st_t
                loss_total[step] += cost_3
                time_total[step] += cum_time
                          
    st_loss = ""
    for idx in range(num_steps):
        st_loss += str(float(loss_total[idx])/times) + ", "
    print st_loss, "\n\n"
    st_time = ""
    for idx in range(num_steps):
        st_time += str(float(time_total[idx])/times) + ", "    
    print st_time
    
print("OUR")