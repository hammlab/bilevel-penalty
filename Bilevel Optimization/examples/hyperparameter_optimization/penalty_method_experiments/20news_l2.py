# -*- coding: utf-8 -*-
"""
Created on Sun May 13 10:42:44 2018

@author: Akshay
"""

# load some data
from sklearn import datasets, linear_model
import tensorflow as tf
import numpy as np
import keras
import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
categories = None
#remove = ('headers', 'footers', 'quotes')

print("loading")
data_train_f = fetch_20newsgroups(subset='train', categories=categories, shuffle=False, random_state=42)#, remove=remove)
data_test_f = fetch_20newsgroups(subset='test', categories=categories, shuffle=False, random_state=42)#, remove=remove)
#print('data loaded')

target_names_f = data_train_f.target_names
y_train_f, y_test_f = data_train_f.target, data_test_f.target
vectorizer_f = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train_f = vectorizer_f.fit_transform(data_train_f.data)
#####
#X_train = X_train.todense()
#print("n_samples: %d, n_features: %d" % X_train_f.shape)
X_test_f = vectorizer_f.transform(data_test_f.data)
#####
#X_test = X_test.todense()
#print("n_samples: %d, n_features: %d" % X_test_f.shape)

#####
X_all_f = vstack([X_train_f, X_test_f])
#X_all = np.concatenate(([X_train, X_test]), axis = 0)

# binarize labels
y_train_f[data_train_f.target < 10] = -1
y_train_f[data_train_f.target >= 10] = 1
y_test_f[data_test_f.target < 10] = -1
y_test_f[data_test_f.target >= 10] = 1
  
Y_all_f = np.concatenate([y_train_f, y_test_f], axis = 0)
#####
#Y_all = Y_all.reshape(18846, 1)
shuff_idx_f = np.load("20news_shuff_idx.npy")
X_all_f = X_all_f[shuff_idx_f]
Y_all_f = Y_all_f[shuff_idx_f]

val_f = int(X_all_f.shape[0]/3)     
 
X_train_f = X_all_f[:val_f]
y_train_f = Y_all_f[:val_f]
      
X_val_f = X_all_f[val_f: 2*val_f]
y_val_f = Y_all_f[val_f: 2*val_f]

X_test_f = X_all_f[2*val_f: ]
y_test_f = Y_all_f[2*val_f: ]

#################
print("loading again")
data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=False, random_state=42)#, remove=remove)
data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=False, random_state=42)#, remove=remove)

target_names = data_train.target_names
y_train, y_test = data_train.target, data_test.target
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
#####
X_train = X_train.todense()
#print("n_samples: %d, n_features: %d" % X_train_f.shape)
X_test = vectorizer.transform(data_test.data)
#####
X_test = X_test.todense()
#print("n_samples: %d, n_features: %d" % X_test_f.shape)

#####
#X_all_f = vstack([X_train_f, X_test_f])
X_all = np.concatenate(([X_train, X_test]), axis = 0)

# binarize labels
y_train[data_train.target < 10] = 0
y_train[data_train.target >= 10] = 1
y_test[data_test.target < 10] = 0
y_test[data_test.target >= 10] = 1
  
Y_all = np.concatenate([y_train, y_test], axis = 0)
#####
#Y_all = Y_all.reshape(18846, 1)
shuff_idx = np.load("20news_shuff_idx.npy")
X_all = X_all[shuff_idx]
Y_all = Y_all[shuff_idx]

val = int(X_all.shape[0]/3)     
 
X_train = X_all[:val]
y_train = Y_all[:val]
      
X_val = X_all[val: 2*val]
y_val = Y_all[val: 2*val]

X_test = X_all[2*val: ]
y_test = Y_all[2*val: ]

y_train = y_train.reshape([y_train.shape[0], 1])
y_val = y_val.reshape([y_val.shape[0], 1])

print("loaded")

# number of features
num_features = X_train.shape[1]
# number of target labels
num_labels = 2
# learning rate (alpha)
batch_size = 2048#y_train.shape[0] - 1#64#512
# number of epochs
num_steps = 50#200
lr_outer = 1E0#0.2
lr_inner = 1E-2#2*1E-2
rho_i = np.ones((1)) * 1E0
lamb_i = np.ones((1)) * 1E0
alpha_i = np.ones((1)) * 0
eps_t = 1E0            
highest = 0
best = 0

y_train = keras.utils.to_categorical(y_train, num_labels)
y_val = keras.utils.to_categorical(y_val, num_labels)
#y_test = keras.utils.to_categorical(y_test, num_labels)      

print X_train.shape
print X_val.shape
print y_train.shape
print y_val.shape

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
 
    # Inputs
    #'''
    a = tf.sparse_placeholder(tf.float32, [y_train.shape[0], num_features])
    tf_train_dataset_tf = tf.placeholder(tf.float32, [batch_size, num_features])
    tf_train_labels_tf = tf.placeholder(tf.float32, [batch_size, num_labels])
    tf_val_dataset_tf = tf.placeholder(tf.float32, [batch_size, num_features])
    tf_val_labels_tf = tf.placeholder(tf.float32, [batch_size, num_labels])
    #'''
    '''
    tf_train_dataset = tf.placeholder(tf.float32, [batch_size, num_features])
    tf_train_labels = tf.placeholder(tf.float32, [batch_size, num_labels])
    tf_val_dataset = tf.placeholder(tf.float32, [batch_size, num_features])
    tf_val_labels = tf.placeholder(tf.float32, [batch_size, num_labels])
    '''
    
    rho_tf = tf.placeholder(tf.float32,[1],'rho_tf')
    lamb_tf = tf.placeholder(tf.float32,[1],'lamb_tf') 
    alpha_tf = tf.placeholder(tf.float32,[1], 'alphat_tf')
    #tf_entire_test_dataset = tf.constant(X_test)
 
    # Variables.
    weights = tf.Variable(np.zeros((num_features, num_labels), np.float32))
    #'''
    tf_train_dataset = tf.Variable(np.zeros((batch_size, num_features), np.float32), name='train_data')
    assign_train_dataset= tf.assign(tf_train_dataset, tf_train_dataset_tf)
    tf_train_labels = tf.Variable(np.zeros((batch_size, num_labels), np.float32), name='train_labels')
    assign_train_labels= tf.assign(tf_train_labels, tf_train_labels_tf)
    tf_val_dataset = tf.Variable(np.zeros((batch_size, num_features), np.float32), name='val_data')
    assign_val_dataset = tf.assign(tf_val_dataset, tf_val_dataset_tf)
    tf_val_labels = tf.Variable(np.zeros((batch_size, num_labels), np.float32), name='val_labels')
    assign_val_labels= tf.assign(tf_val_labels, tf_val_labels_tf)
    #'''
    
    alpha = tf.Variable(np.zeros((1), np.float32), name='alpha')
    assign_alpha = tf.assign(alpha, alpha_tf)
    rho = tf.Variable(np.zeros((1), np.float32), name='rho')
    assign_rho = tf.assign(rho, rho_tf)
    lamb = tf.Variable(np.zeros((1), np.float32), name='lamb')
    assign_lamb = tf.assign(lamb, lamb_tf)
 
    # Training computation.
    logits_inner = tf.matmul(tf_train_dataset, weights)
    g1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_inner))
    #g1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_inner))
    g2 = tf.exp(alpha) * tf.nn.l2_loss(weights)
    g = g1 + g2
    dgdv = tf.gradients(g, weights)
    
    logits_outer = tf.matmul(tf_val_dataset, weights)
    f = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf_val_labels, logits=logits_outer))
    #f = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_val_labels, logits=logits_outer))
 
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
 
with tf.Session(graph=graph) as session:
    # initialize weights and biases
    tf.global_variables_initializer().run()
    print("Initialized")
 
    for step in range(num_steps):
        # pick a randomized offset
        #ind_train = X_train.shape[0]
        ind_train = np.random.choice(X_train.shape[0], size=(batch_size), replace=False)
        #ind_val = X_val.shape[0]
        ind_val = np.random.choice(X_val.shape[0], size=(batch_size), replace=False)
        
        # Generate a minibatch.
        #'''
        batch_data_train = X_train[ind_train]
        batch_labels_train = y_train[ind_train]
        batch_data_val = X_val[ind_val]
        batch_labels_val = y_val[ind_val]
        #'''
        '''
        batch_data_train = X_train
        batch_labels_train = y_train
        batch_data_val = X_val
        batch_labels_val = y_val
        #'''
 
    
        # Prepare the feed dict
        feed_dict = {tf_train_dataset : batch_data_train, tf_train_labels : batch_labels_train, tf_val_dataset : batch_data_val, tf_val_labels : batch_labels_val}
        
        session.run(assign_alpha, feed_dict={alpha_tf:alpha_i})
        session.run(assign_rho, feed_dict={rho_tf:rho_i})
        session.run(assign_lamb, feed_dict={lamb_tf:lamb_i})
        
        #'''
        session.run(assign_train_dataset, feed_dict={tf_train_dataset_tf:batch_data_train})
        session.run(assign_train_labels, feed_dict={tf_train_labels_tf:batch_labels_train})
        session.run(assign_val_dataset, feed_dict={tf_val_dataset_tf:batch_data_val})
        session.run(assign_val_labels, feed_dict={tf_val_labels_tf:batch_labels_val})
        #'''
        
        # run one step of computation
        tick = time.time()
        for it in range(1):
            #print "here"
            '''
            session.run(train_u, feed_dict = feed_dict)
            session.run(train_v, feed_dict = feed_dict)
            #'''
            #'''
            session.run(train_u)
            session.run(train_v)
            #'''
        st_t = time.time() - tick
                        
        #l1, l2, l3, l4, l5, l6, l7, l8, l9, l10 = session.run([f, gvnorm, lamb_g, alpha, rho, lamb, g1, g2, hunorm, hvnorm], feed_dict=feed_dict)
        l4, l9, l10 = session.run([alpha, hunorm, hvnorm], feed_dict=feed_dict)
        
        #acc = accuracy(test_prediction.eval(), y_test)
        #if acc > highest:
        #    highest = acc
        #    best = l4
        
        if (l9**2 + l10**2 < eps_t**2):
            rho_i *= 2
            lamb_i /= 2
            eps_t /= 2
            
        alpha_i = l4
        
        if (step % 1 == 0):
            '''
            print "Time for epoch: ", step, " ", time.time() - tick
            
            #print("Minibatch accuracy: {:.1f}%".format(accuracy(predictions, batch_labels_train)))
            print "f = ", l1, " gvnorm = ", l2, " lamb_g = ", l3, " alpha = ", l4, " rho = ", l5, " lamb = ", l6, " g1 = ", l7, " g2 = ", l8, l9, l10
            #print("\nTest accuracy: {:.1f}%".format(accuracy(test_prediction.eval(), y_test)))  
            #print highest, best
            #y_test = y_test.reshape(7532, 1)      
            #y_train = y_train.reshape(11314, 1)      
            #print y_train_f.shape, y_test_f.shape   
            print "step:", step
            '''
            clf = linear_model.LogisticRegression(solver='lbfgs', C=np.exp(-l4[0]), fit_intercept=False, tol=1e-22, max_iter=500)
            clf.fit(X_train_f, y_train_f)
            cost_1 = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_val_f, y_val_f, 0.)
            
            cost_2 = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_test_f, y_test_f, 0.)
            
            print st_t, ",", cost_1, ",", cost_2, l4[0]
    
    
     
        
        