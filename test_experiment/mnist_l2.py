# -*- coding: utf-8 -*-
"""
Created on Sun May 13 10:42:44 2018

@author: Akshay
"""

# load some data
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
from sklearn.utils.extmath import safe_sparse_dot
from  scipy.misc import logsumexp
from sklearn.preprocessing import LabelBinarizer

def norm_sq_sum2(xs):
    return tf.reduce_sum([tf.reduce_sum(tf.square(t)) for t in xs])

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
        print 'size wrong', dim
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
Y_test_multi = y_test
sample_weight_test = np.ones(X_test.shape[0])


# number of features
num_features = X_train_all.shape[1]
# number of target labels
num_labels = 10
# number of epochs
num_epochs = 500
lr_outer = 8*1E-1#7*1E-1#1
lr_inner = 3*1E-2#3*1E-2#1E-2


best = np.ones((num_features, num_labels))
times = 5
#y_train = keras.utils.to_categorical(y_train, num_labels)
#y_test = keras.utils.to_categorical(y_test, num_labels)      
#print y_train.shape
#print y_test.shape

# initialize a tensorflow graph
graph = tf.Graph()
 
with graph.as_default():
    """
    defining all the nodes
    """
 
    # Inputs
    rho_tf = tf.placeholder(tf.float32,[1],'rho_tf')
    lamb_tf = tf.placeholder(tf.float32,[1],'lamb_tf') 
    alpha_tf = tf.placeholder(tf.float32,[num_features, num_labels], 'alphat_tf')
   
    train_dataset_tf = tf.placeholder(tf.float32, [train_size, num_features], 'train_data')
    train_labels_tf = tf.placeholder(tf.float32, [train_size, 10], 'train_labels')
    val_dataset_tf = tf.placeholder(tf.float32, [val_size, num_features], 'val_data')
    val_labels_tf = tf.placeholder(tf.float32, [val_size, 10], 'val_labels')
    
    tf_entire_test_dataset = tf.constant(X_test)
    tf_entire_test_labels = tf.constant(y_test)
 
    # Variables.
    #weights = tf.Variable(tf.truncated_normal([num_features, num_labels]))
    weights = tf.Variable(np.zeros((num_features, num_labels), np.float32))
    alpha = tf.Variable(np.ones((num_features, num_labels), np.float32), name='alpha')
    assign_alpha = tf.assign(alpha, alpha_tf)
    rho = tf.Variable(np.zeros((1), np.float32), name='rho')
    assign_rho = tf.assign(rho, rho_tf)
    lamb = tf.Variable(np.zeros((1), np.float32), name='lamb')
    assign_lamb = tf.assign(lamb, lamb_tf)
    tf_entire_train_dataset = tf.Variable(np.zeros((train_size, num_features), np.float32), name='train_data')
    assign_tf_entire_train_dataset = tf.assign(tf_entire_train_dataset, train_dataset_tf)
    tf_entire_train_labels = tf.Variable(np.zeros((train_size, 10), np.float32), name='train_labels')
    assign_tf_entire_train_labels = tf.assign(tf_entire_train_labels, train_labels_tf)
    
    tf_entire_val_dataset = tf.Variable(np.zeros((val_size, num_features), np.float32), name='val_data')
    assign_tf_entire_val_dataset = tf.assign(tf_entire_val_dataset, val_dataset_tf)
    tf_entire_val_labels = tf.Variable(np.zeros((val_size, 10), np.float32), name='val_labels')
    assign_tf_entire_val_labels = tf.assign(tf_entire_val_labels, val_labels_tf)
 
    # Training computation.
    logits_inner = tf.matmul(tf_entire_train_dataset, weights)
    g1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_entire_train_labels, logits=logits_inner))
    #g1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_inner))
    #g2 = tf.reduce_sum(tf.multiply(tf.exp(alpha), tf.nn.l2_loss(weights)))
    g2 = 0.5 * tf.reduce_mean(tf.multiply(tf.exp(alpha), tf.square(weights)))
    g = g1 + g2
    dgdv = tf.gradients(g, weights)
    
    logits_outer = tf.matmul(tf_entire_val_dataset, weights)
    f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_entire_val_labels, logits=logits_outer))
    #f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_test_labels, logits=logits_outer))
 
    gvnorm = 0.5 * rho * norm_sq_sum2(dgdv) 
    lamb_g = lamb * g

    loss_u = f + gvnorm + lamb_g
    loss_v = f + gvnorm + lamb_g
    
    # Optimizer.
    opt_u = tf.train.AdamOptimizer(lr_outer)
    opt_v = tf.train.AdamOptimizer(lr_inner)
    # Optimizer.
    train_u = opt_u.minimize(loss_u, var_list = alpha)
    train_v = opt_v.minimize(loss_v, var_list = weights)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits_inner)
    test_prediction = tf.nn.softmax(tf.matmul(tf_entire_test_dataset, weights))
    
    logits_test = tf.matmul(tf_entire_test_dataset, weights)
    test_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf_entire_test_labels, logits=logits_test))
    
    logits_val = tf.matmul(tf_entire_val_dataset, weights)
    val_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf_entire_val_labels, logits=logits_val))
    
    tgrad_and_var = opt_u.compute_gradients(loss_u, var_list=alpha)
    hunorm = tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var])
    
    tgrad_and_var = opt_v.compute_gradients(loss_v, var_list=weights)
    hvnorm = tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var])
 
with tf.Session(graph=graph) as session:
    # initialize weights and biases
    print("Initialized")
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
    
        highest = 0
        lowest = 100000
        rho_i = np.ones((1)) * 1E0
        lamb_i = np.ones((1)) * 1E0
        eps_t = 1E-2 #1E-2 for small dataset                
        alpha_i = np.zeros((num_features, num_labels))
        tf.global_variables_initializer().run()
        cum_time = 0
        for epoch in range(num_epochs):
            session.run(assign_alpha, feed_dict={alpha_tf:alpha_i})
            session.run(assign_rho, feed_dict={rho_tf:rho_i})
            session.run(assign_lamb, feed_dict={lamb_tf:lamb_i})
            
            session.run(assign_tf_entire_train_dataset, feed_dict={train_dataset_tf:X_train})
            session.run(assign_tf_entire_train_labels, feed_dict={train_labels_tf:y_train})
            session.run(assign_tf_entire_val_dataset, feed_dict={val_dataset_tf:X_val})
            session.run(assign_tf_entire_val_labels, feed_dict={val_labels_tf:y_val})
            
            tick = time.time()
            # run one step of computation
            for it in range(1):
                #print "here"
                session.run(train_u)
                session.run(train_v)
            st_t = time.time() - tick
            cum_time += st_t                        
                    
            l1, l2, l3, l4, l5, l6, l7, l8, predictions, l9, l10 = session.run([f, gvnorm, lamb_g, alpha, rho, lamb, g1, g2, train_prediction, hunorm, hvnorm])
            acc = val_loss.eval()#accuracy(test_prediction.eval(), y_test)
            if acc < lowest:
                lowest = acc
            
            if (l9**2 + l10**2 < eps_t**2):
                rho_i *= 2
                lamb_i /= 2
                eps_t /= 2
                print "Updated", rho_i
            alpha_i = l4
             
            '''
            print("Train accuracy: {:.1f}%".format(accuracy(predictions, y_train)))
            print "f = ", l1, " gvnorm = ", l2, " lamb_g = ", l3, " rho = ", l5, " lamb = ", l6, " g1 = ", l7, " g2 = ", l8, l9, l10
            print("Test accuracy: {:.1f}%".format(accuracy(test_prediction.eval(), y_test)))  
            print "ValLoss:", val_loss.eval()
            print "TestLoss:", test_loss.eval()
            print "Highest = ", highest#, best.shape, best
            print "Norm W = ", tf.norm(alpha).eval()
            print "\n"
            '''   
            w_fin = weights.eval()
            w_fin = w_fin.T
            w_fin = w_fin.flatten()
            loss = _multinomial_loss(w_fin, X_test, Y_test_multi, np.zeros(w_fin.size), sample_weight_test)[0]
            print epoch, test_loss.eval(), loss, val_loss.eval(), lowest
            total_loss[epoch] += test_loss.eval()
            total_time[epoch] += cum_time
                      
    print total_loss/times
    print "\n\n"
    print total_time/times
    
    st_loss = ""
    for idx in range(len(total_loss)):
        st_loss += str(float(total_loss[idx])/times) + ", "
    print(st_loss, "\n\n")
    st_time = ""
    for idx in range(len(total_time)):
        st_time += str(float(total_time[idx])/times) + ", "
    print(st_time)
    print("OUR")     