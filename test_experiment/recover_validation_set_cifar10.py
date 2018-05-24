# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:12:54 2018

@author: Akshay
"""

from __future__ import print_function
import keras
from keras.datasets import cifar10, cifar100
import numpy as np
import tensorflow as tf
from cleverhans.utils_tf import model_train, model_eval, batch_eval, model_argmax
import time

def norm_sq_sum2(xs):
    return tf.reduce_sum([tf.reduce_sum(tf.square(t)) for t in xs])

def norm_sq_sum(xs):
    total = 0.
    for i in range(len(xs)):
        if xs[i] is not None:
            total += tf.reduce_sum(tf.square(xs[i]))
        else:
            print(xs[i])            
    return total

num_classes_train = 10
num_classes_test = 10

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
only_y = y_train
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#'''
ind_test = []
for i in range(len(y_test)):
    if y_test[i] == 0:
        ind_test.append(i)
    elif y_test[i] == 3:
        ind_test.append(i)

x_test = x_test[ind_test]
y_test = y_test[ind_test]
'''
three = np.argwhere(y_test == 3).flatten()
y_test[three] = 1       
# '''     
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes_train)
y_test = keras.utils.to_categorical(y_test, num_classes_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

X_train = x_train
Y_train = y_train
X_test = x_test
Y_test = y_test
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

num_features = X_train.shape[1]
lr_outer = 1E-1#1E-3
lr_inner = 1E-4#1E-4
rho_i = np.ones((1)) * 1E0
lamb_i = np.ones((1)) * 1E0
eps_t = 1E0 
batch_size = 128

tf.reset_default_graph()
ss = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

x_train_tf = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='x_train')
y_train_tf = tf.placeholder(tf.float32, shape=(None, num_classes_train), name='y_train')
x_test_tf = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='x_test')
y_test_tf = tf.placeholder(tf.float32, shape=(None, num_classes_test), name='y_test')

rho_tf = tf.placeholder(tf.float32, [1],'rho_tf')
lamb_tf = tf.placeholder(tf.float32, [1],'lamb_tf') 

alpha_atanh_tf = tf.placeholder(tf.float32, [batch_size], 'alphat_tf')
alpha_atanh = tf.Variable(np.ones((batch_size), np.float32), name='alpha')
alpha= 0.5*(tf.tanh(alpha_atanh) + 1.)
assign_alpha_atanh = tf.assign(alpha_atanh, alpha_atanh_tf)

rho = tf.Variable(np.ones((1), np.float32), name='rho')
assign_rho = tf.assign(rho, rho_tf)
lamb = tf.Variable(np.ones((1), np.float32), name='lamb')
assign_lamb = tf.assign(lamb, lamb_tf)

def make_classifier(ins, nh=100, trainable=True):
    W1 = tf.get_variable('W1',[5,5,3,64], initializer=tf.random_normal_initializer(stddev = 0.01), trainable = trainable)
    b1 = tf.get_variable('b1',[64],initializer=tf.constant_initializer(0.1), trainable = trainable)
    c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ins, W1, strides=[1,1,1,1], padding='SAME'),b1))
    p1 = tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    W2 = tf.get_variable('W2',[5,5,64,64],initializer=tf.random_normal_initializer(stddev = 0.01), trainable = trainable)
    b2 = tf.get_variable('b2',[64],initializer=tf.constant_initializer(0.1), trainable = trainable)
    c2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p1, W2, strides=[1,1,1,1], padding='SAME'), b2))
    p2 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    a2 = tf.reshape(p2,[-1,8*8*64])
    W3 = tf.get_variable('W3',[8*8*64,nh],initializer=tf.random_normal_initializer(stddev = 0.01), trainable = trainable)
    b3 = tf.get_variable('b3',[nh],initializer=tf.constant_initializer(0.1), trainable = trainable)
    a3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(a2,W3),b3))
    return a3

def make_classifier2(ins, nh=100):
    W1 = tf.get_variable('W1',[3,3,3,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[64],initializer=tf.constant_initializer(0.1))
    c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ins, W1, strides=[1,1,1,1], padding='SAME'),b1))
    
    W2 = tf.get_variable('W2',[3,3,64,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b2 = tf.get_variable('b2',[64],initializer=tf.constant_initializer(0.1))
    c2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(c1, W2, strides=[1,1,1,1], padding='SAME'),b2))
    
    p1 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    d1 = tf.layers.dropout(p1, 0.25)
    
    W3 = tf.get_variable('W3',[3,3,64,128],initializer=tf.random_normal_initializer(stddev=0.01))
    b3 = tf.get_variable('b3',[128],initializer=tf.constant_initializer(0.1))
    c3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(d1, W3, strides=[1,1,1,1], padding='SAME'),b3))
    
    W4 = tf.get_variable('W4',[3,3,128,128],initializer=tf.random_normal_initializer(stddev=0.01))
    b4 = tf.get_variable('b4',[128],initializer=tf.constant_initializer(0.1))
    c4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(c3, W4, strides=[1,1,1,1], padding='SAME'),b4))
    
    p2 = tf.nn.max_pool(c4, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    d2 = tf.layers.dropout(p2, 0.25)
    
    W5 = tf.get_variable('W5',[3,3,128,256],initializer=tf.random_normal_initializer(stddev=0.01))
    b5 = tf.get_variable('b5',[256],initializer=tf.constant_initializer(0.1))
    c5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(d2, W5, strides=[1,1,1,1], padding='SAME'),b5))
    
    W6 = tf.get_variable('W6',[3,3,256,256],initializer=tf.random_normal_initializer(stddev=0.01))
    b6 = tf.get_variable('b6',[256],initializer=tf.constant_initializer(0.1))
    c6 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(c5, W6, strides=[1,1,1,1], padding='SAME'),b6))
    
    p3 = tf.nn.max_pool(c6, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    d3 = tf.layers.dropout(p3, 0.25)
    
    a2 = tf.reshape(d3, [-1, 4*4*256])
    
    W7 = tf.get_variable('W7',[4*4*256, 1024],initializer=tf.random_normal_initializer(stddev=0.01))
    b7 = tf.get_variable('b7',[1024],initializer=tf.constant_initializer(0.1))
    a3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(a2,W7), b7))
    
    W8 = tf.get_variable('W8',[1024, 512],initializer=tf.random_normal_initializer(stddev=0.01))
    b8 = tf.get_variable('b8',[512],initializer=tf.constant_initializer(0.1))
    a4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(a3, W8), b8))
    
    W9 = tf.get_variable('W9',[512, nh],initializer=tf.random_normal_initializer(stddev=0.01))
    b9 = tf.get_variable('b9',[nh],initializer=tf.constant_initializer(0.1))
    a5 = tf.nn.relu(tf.nn.bias_add(tf.matmul(a4, W9), b9))

    return a5

with tf.variable_scope('cls', reuse=False):
    a5_tr = make_classifier(x_train_tf)
W4_tr =  tf.Variable(np.ones((100, num_classes_train), np.float32))
b4_tr =  tf.Variable(np.ones((num_classes_train), np.float32))
cls_train = tf.nn.bias_add(tf.matmul(a5_tr, W4_tr),b4_tr)
    
var_cls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'cls')
logits_inner = cls_train
g = tf.reduce_mean(tf.multiply(alpha, tf.nn.softmax_cross_entropy_with_logits(logits = logits_inner,labels = y_train_tf)))/ tf.reduce_sum(alpha)

#dgdv = tf.gradients(g, var_cls)
dgdv = tf.gradients(g, [var_cls[0], var_cls[1], var_cls[2], var_cls[3], var_cls[4], var_cls[5], W4_tr, b4_tr])
gvnorm = 0.5 * rho * norm_sq_sum(dgdv) 
lamb_g = lamb * g

with tf.variable_scope('cls', reuse=True):
    a5_te = make_classifier(x_test_tf)
#W4_te =  tf.Variable(np.ones((100, num_classes_test), np.float32))
#b4_te =  tf.Variable(np.ones((num_classes_test), np.float32))
#cls_test = tf.nn.bias_add(tf.matmul(a5_te, W4_te), b4_te)
cls_test = tf.nn.bias_add(tf.matmul(a5_te, W4_tr),b4_tr)


logits_outer = cls_test
f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits_outer, labels=y_test_tf))

loss = f + gvnorm + lamb_g

opt_u = tf.train.AdamOptimizer(lr_outer)
opt_v = tf.train.AdamOptimizer(lr_inner)
# Optimizer.
#print([alpha, W4_te, b4_te])
#print("\n\n")
#train_u = opt_u.minimize(loss, var_list = [alpha_atanh, W4_te, b4_te])
train_u = opt_u.minimize(loss, var_list = alpha_atanh)
#print(var_cls)
train_v = opt_v.minimize(loss, var_list = [var_cls[0], var_cls[1], var_cls[2], var_cls[3], var_cls[4], var_cls[5], W4_tr, b4_tr])

accuracy_g = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_train_tf, 1), tf.argmax(logits_inner, 1)), tf.float32))
accuracy_f = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_test_tf, 1), tf.argmax(logits_outer, 1)), tf.float32))


#tgrad_and_var = opt_u.compute_gradients(loss, var_list=[alpha_atanh, W4_te, b4_te])
tgrad_and_var = opt_u.compute_gradients(loss, var_list = alpha_atanh)
hunorm = tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var])

tgrad_and_var = opt_v.compute_gradients(loss, var_list=[var_cls[0], var_cls[1], var_cls[2], var_cls[3], var_cls[4], var_cls[5], W4_tr, b4_tr])
hvnorm = tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var])

tf.global_variables_initializer().run()
alpha_atan = np.ones((X_train.shape[0]))  
# Normalize
alpha = 0.5*(np.tanh(alpha_atan)+1.)
alpha = 0.5*alpha/np.mean(alpha)
alpha = np.maximum(.00001,np.minimum(.99999,alpha))
alpha_atan = np.arctanh(2.*alpha-1.)
for epoch in range(1000):

    nb_batches = int(np.floor(float(X_train.shape[0]) / batch_size))
    index_shuf = np.arange(X_train.shape[0])
    np.random.shuffle(index_shuf)
    
    f_t = 0
    gvnorm_t = 0
    lamb_g_t = 0
    g_t = 0
    
    #print("here")
    for batch in range(nb_batches):
        #print("batch ", batch, " of ", nb_batches)
        ind = range(batch_size*batch, min(batch_size*(1+batch), X_train.shape[0]))
        ind_test = np.random.choice(X_test.shape[0], size=(batch_size), replace=False)

        #print(X_train[index_shuf[ind]].shape)
        feed_dict = {x_train_tf: X_train[index_shuf[ind]], y_train_tf:Y_train[index_shuf[ind]], x_test_tf:X_test[ind_test], y_test_tf:Y_test[ind_test]}
        
        ss.run(assign_alpha_atanh, feed_dict={alpha_atanh_tf:alpha_atan[index_shuf[ind]]})
        ss.run(assign_rho, feed_dict={rho_tf:rho_i})
        ss.run(assign_lamb, feed_dict={lamb_tf:lamb_i})
        
        for it in range(1):
            ss.run(train_v, feed_dict=feed_dict)
        ss.run(train_u, feed_dict=feed_dict)
            
        f_t, g_t, gvnorm_t, lamb_g_t, tmp_alpha_atan, hunorm_t, hvnorm_t = ss.run([f, g, gvnorm, lamb_g, alpha_atanh, hunorm, hvnorm], feed_dict = feed_dict)
        alpha_atan[index_shuf[ind]] = tmp_alpha_atan
        
        if (hunorm_t**2 + hvnorm_t**2 < eps_t**2):
            print("updated rho")
            rho_i *= 2
            lamb_i /= 2
            eps_t /= 2
        
    print("here1", epoch)
    print("f = ", f_t)
    print("g = ", g_t)
    print("gvnorm = ", gvnorm_t)
    print("lamb_g = ", lamb_g_t)
    
    rho_t = rho.eval()
    lamb_t = lamb.eval()
    
    print(rho_t)
    print(lamb_t)
    
    print('acc_tr = %f'%(model_eval(ss, x_train_tf, y_train_tf, logits_inner, X_train, Y_train, args={'batch_size':batch_size})))
    print('acc = %f'%(model_eval(ss, x_test_tf, y_test_tf, logits_outer, X_test, Y_test, args={'batch_size':batch_size})))
    
    if True:
        alpha = 0.5 * (np.tanh(alpha_atan)+1.) # scale to beteen [0 1] from [-1 1]
        alpha = 0.5 * X_train.shape[0] * alpha / sum(alpha)
        alpha = np.maximum(.00001, np.minimum(.99999, alpha))
        alpha_atan = np.arctanh(0.99999 * (2. * alpha - 1.))
    
    pts = np.argsort(alpha).flatten()
    correct = 0
    for i in range(1, 10001):
        if only_y[pts[-i]] == 0 or only_y[pts[-i]] == 3:
            correct += 1
    print(alpha[pts[-10000:]])
    print(len(np.where(alpha>0.1)[0])/np.float(X_train.shape[0]))
    print(correct)
    print("\n")
#'''