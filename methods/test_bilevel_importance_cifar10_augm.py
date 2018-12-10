from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import sys
sys.path.append('/home/hammj/Dropbox/Research/AdversarialLearning/codes/lib/cleverhans-master')

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

#from cleverhans.utils_mnist import data_mnist
import keras
from keras.datasets import cifar10

from bilevel_importance_augm import bilevel_importance
import time

from augment import augment

#############################################################################################################

lr_u = 1E-2
lr_v = 1E-4

nepochs = 400
niter = 1
batch_size = 100

rho_0 = 1E-1
lamb_0 = 1E-1
eps_0 = 1E0

c_rho = 2
c_lamb = 0.5
c_eps = 0.5

height = 32
width = 32
nch = 3
num_class = 10

frac_noisy = 0.5

istraining_tf = tf.placeholder_with_default(True,shape=())

'''
def make_classifier(ins,d=128,num_class=num_class):
    W1 = tf.get_variable('W1',[5,5,3,32],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[32],initializer=tf.constant_initializer(0.0))
    c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ins, W1, strides=[1,1,1,1], padding='SAME'),b1))
    p1 = tf.nn.max_pool(c1, ksize=[1,3,3,1],strides=[1,2,2,1], padding='SAME')
 
    W2 = tf.get_variable('W2',[5,5,32,32],initializer=tf.random_normal_initializer(stddev=0.01))
    b2 = tf.get_variable('b2',[32],initializer=tf.constant_initializer(0.0))
    c2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p1, W2, strides=[1,1,1,1], padding='SAME'), b2))
    p2 = tf.nn.max_pool(c2, ksize=[1,3,3,1],strides=[1,2,2,1], padding='SAME')
 
    W3 = tf.get_variable('W3',[5,5,32,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b3 = tf.get_variable('b3',[64],initializer=tf.constant_initializer(0.0))
    c3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p2, W3, strides=[1,1,1,1], padding='SAME'), b3))
    p3 = tf.nn.max_pool(c3, ksize=[1,3,3,1],strides=[1,2,2,1], padding='SAME')
    a3 = tf.reshape(p3,[-1,4*4*64])
    
    W4 = tf.get_variable('W4',[4*4*64,d],initializer=tf.random_normal_initializer(stddev=0.01))
    b4 = tf.get_variable('b4',[d],initializer=tf.constant_initializer(0.0))
    a4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(a3,W4),b4))
    a4 = tf.layers.dropout(a4, rate=.5, training=istraining_tf)
    #a7 = tf.nn.dropout(a7,keep_prob=1.0)

    W5 = tf.get_variable('W5',[d,num_class],initializer=tf.random_normal_initializer(stddev=0.01))
    b5 = tf.get_variable('b5',[num_class],initializer=tf.constant_initializer(0.0))
    a5 = tf.matmul(a4,W5) + b5

    out = a5
    return out
'''


def make_classifier(ins,d1=512,d2=256,num_class=num_class):
    W1 = tf.get_variable('W1',[3,3,3,48],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[48],initializer=tf.constant_initializer(0.0))
    c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ins, W1, strides=[1,1,1,1], padding='SAME'),b1))
    W2 = tf.get_variable('W2',[3,3,48,48],initializer=tf.random_normal_initializer(stddev=0.01))
    b2 = tf.get_variable('b2',[48],initializer=tf.constant_initializer(0.0))
    c2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(c1, W2, strides=[1,1,1,1], padding='SAME'),b2))
    p2 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    p2 = tf.layers.dropout(p2, rate=0.25, training=istraining_tf)
    #p2 = tf.nn.dropout(p2,keep_prob=1.0)

    W3 = tf.get_variable('W3',[3,3,48,96],initializer=tf.random_normal_initializer(stddev=0.01))
    b3 = tf.get_variable('b3',[96],initializer=tf.constant_initializer(0.0))
    c3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p2, W3, strides=[1,1,1,1], padding='SAME'),b3))
    W4 = tf.get_variable('W4',[3,3,96,96],initializer=tf.random_normal_initializer(stddev=0.01))
    b4 = tf.get_variable('b4',[96],initializer=tf.constant_initializer(0.0))
    c4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(c3, W4, strides=[1,1,1,1], padding='SAME'),b4))
    p4 = tf.nn.max_pool(c4, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    p4 = tf.layers.dropout(p4, rate=0.25, training=istraining_tf)
    #p4 = tf.nn.dropout(p4,keep_prob=1.0)

    W5 = tf.get_variable('W5',[3,3,96,192],initializer=tf.random_normal_initializer(stddev=0.01))
    b5 = tf.get_variable('b5',[192],initializer=tf.constant_initializer(0.0))
    c5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p4, W5, strides=[1,1,1,1], padding='SAME'),b5))
    W6 = tf.get_variable('W6',[3,3,192,192],initializer=tf.random_normal_initializer(stddev=0.01))
    b6 = tf.get_variable('b6',[192],initializer=tf.constant_initializer(0.0))
    c6 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(c5, W6, strides=[1,1,1,1], padding='SAME'),b6))
    p6 = tf.nn.max_pool(c6, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    p6 = tf.layers.dropout(p6, rate=0.25, training=istraining_tf)
    #p6 = tf.nn.dropout(p6,keep_prob=1.0)
    a6 = tf.reshape(p6,[-1,4*4*192])
    
    W7 = tf.get_variable('W7',[4*4*192,d1],initializer=tf.random_normal_initializer(stddev=0.01))
    b7 = tf.get_variable('b7',[d1],initializer=tf.constant_initializer(0.0))
    a7 = tf.nn.relu(tf.nn.bias_add(tf.matmul(a6,W7),b7))
    a7 = tf.layers.dropout(a7, rate=0.5, training=istraining_tf)
    #a7 = tf.nn.dropout(a7,keep_prob=1.0)

    W8 = tf.get_variable('W8',[d1,d2],initializer=tf.random_normal_initializer(stddev=0.01))
    b8 = tf.get_variable('b8',[d2],initializer=tf.constant_initializer(0.0))
    a8 = tf.nn.relu(tf.nn.bias_add(tf.matmul(a7,W8),b8))
    a8 = tf.layers.dropout(a8, rate=0.5, training=istraining_tf)
    #a8 = tf.nn.dropout(a8,keep_prob=1.0)

    W9 = tf.get_variable('W9',[d2,num_class],initializer=tf.random_normal_initializer(stddev=0.01))
    b9 = tf.get_variable('b9',[num_class],initializer=tf.constant_initializer(0.0))
    a9 = tf.matmul(a8,W9) + b9
    #a9 = tf.matmul(a7,W9) + b9

    out = a9
    return out


def main(argv=None):

    #shape = [FLAGS.batch_size,img_rows,img_cols,channels]
    tf.set_random_seed(1234)
    sess = tf.Session()

    ## Read data
    #X_train, Y_train, X_test, Y_test = data_mnist(train_start=0, train_end=60000,test_start=0,test_end=10000)
    #Ntrain = X_train.shape[0]
    #Ntest = X_test.shape[0]
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    Ntrain = X_train.shape[0]
    X_train = X_train.astype('float32') / 255.
    Y_train = keras.utils.to_categorical(Y_train, num_class)
    Ntest = np.int(X_test.shape[0]/batch_size)*batch_size
    X_test = X_test[:Ntest].astype('float32') / 255.
    Y_test = keras.utils.to_categorical(Y_test[:Ntest], num_class)
    
    ## Noisy data
    Ntrain_new = 40000
    X_train_new = X_train[:Ntrain_new]
    # Random label
    Nnoisy = np.int(frac_noisy*Ntrain_new)
    ind_noisy = np.random.choice(Ntrain_new,Nnoisy,replace=False)
    label_noisy = np.random.choice(num_class,Nnoisy,replace=True)   
    y_all = np.argmax(Y_train[:Ntrain_new],1)
    y_all[ind_noisy] = label_noisy
    Y_train_new = np.zeros((Ntrain_new,num_class))
    Y_train_new[np.arange(Ntrain_new),y_all] = 1

    Nval = 10000
    X_val = X_train[Ntrain_new:Ntrain_new+Nval,:]
    Y_val = Y_train[Ntrain_new:Ntrain_new+Nval,:]
    
    ## Define model
    x_train_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_train_tf = tf.placeholder(tf.float32, shape=(batch_size,num_class))
    x_test_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_test_tf = tf.placeholder(tf.float32, shape=(batch_size,num_class))

    x_train_augm_tf, _ = augment(x_train_tf, y_train_tf, horizontal_flip=True, rotate=15, crop_probability=1., crop_min_percent=0.8, crop_max_percent=1.2)
    x_test_augm_tf, _ = augment(x_test_tf, y_test_tf, horizontal_flip=True, rotate=15, crop_probability=1., crop_min_percent=0.8, crop_max_percent=1.2)


    with tf.variable_scope('cls',reuse=False):
        cls_train = make_classifier(x_train_tf)
    var_cls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='cls')
    with tf.variable_scope('cls',reuse=True):
        cls_train_augm = make_classifier(x_train_augm_tf)        
        cls_test = make_classifier(x_test_tf)
        cls_test_augm = make_classifier(x_test_augm_tf)

    saver_cls = tf.train.Saver(var_cls,max_to_keep=None)

    if False:
        pass
    
    ## Train with clean+val data
    if False:
        pass

    ## Train with clean+noisy data
    if False:
        pass
        
    ## Train with clean+noisy+val data
    if False:
        pass


    #########################################################################################################
    ## Bilevel training
    #########################################################################################################

    bl_imp = bilevel_importance(sess,x_train_tf,x_train_augm_tf,x_test_tf,x_test_augm_tf,y_train_tf,y_test_tf,
        cls_train,cls_train_augm,cls_test,cls_test_augm,var_cls,
        batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps,istraining_tf)

    sess.run(tf.global_variables_initializer())
    if False: ## Train with val data
        bl_imp.train_simple(X_val,Y_val,50) ## Train with val data
    if True: ## Train with clean+noisy data
        bl_imp.train_simple(np.concatenate((X_val,X_train_new),0),np.concatenate((Y_val,Y_train_new),0),100,X_test,Y_test)
    
    if False:
        importance_atanh = np.zeros((Ntrain_new))    
    else:
        # Predictions based on valid-pretrained model?
        _,y_train_init = bl_imp.eval_simple(X_train_new,Y_train_new)
        importance_atanh = np.ones((Ntrain_new))*np.arctanh(2.*0.4-1.)
        ind_correct = np.where(np.argmax(Y_train_new)==y_train_init)[0]
        importance_atanh[ind_correct] = np.arctanh(2.*0.6-1.)

    if True:
        for epoch in range(nepochs):
            #tick = time.time()        
            nb_batches = int(np.floor(float(Ntrain_new)/batch_size))
            ind_shuf = np.arange(Ntrain_new)
            np.random.shuffle(ind_shuf)
            for batch in range(nb_batches):
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntrain_new))
                #if len(ind)<FLAGS.batch_size:
                #    ind.extend([np.random.choice(Ntrain,FLAGS.batch_size-len(ind))])
                ind_val = np.random.choice(Nval,size=(batch_size),replace=False)
                ind_tr = ind_shuf[ind_batch]
                f,gvnorm,lamb_g,timp_atanh = bl_imp.train(X_train_new[ind_tr,:],Y_train_new[ind_tr,:],X_val[ind_val,:],Y_val[ind_val,:],importance_atanh[ind_tr],niter)
                importance_atanh[ind_tr] = timp_atanh

            ## Renormalize importance_atanh
            if True:
                importance = 0.5*(np.tanh(importance_atanh)+1.)
                importance = 0.5*importance/np.mean(importance)
                importance = np.maximum(.00001,np.minimum(.99999,importance))
                importance_atanh = np.arctanh(2.*importance-1.)
            
            if epoch%10==0:
                rho_t,lamb_t,eps_t = sess.run([bl_imp.bl.rho_t,bl_imp.bl.lamb_t,bl_imp.bl.eps_t])
                print('epoch %d (rho=%f, lamb=%f, eps=%f): h=%f + %f + %f= %f'%
                    (epoch,rho_t,lamb_t,eps_t,f,gvnorm,lamb_g,f+gvnorm+lamb_g))
                print('test acc:')
                bl_imp.eval_simple(X_test,Y_test)
            
        saver_cls.save(sess,'./results/cls_importance_cifar10_augm_%3.2f.ckpt'%(frac_noisy))
        importance = 0.5*(np.tanh(importance_atanh)+1.)
        np.save('./results/importance_cifar10_augm_%3.2f.npy'%(frac_noisy),importance)
        #np.save(os.path.join(result_dir,')),tX)        
    else:
        pass
        saver_cls.restore(sess,'./results/cls_importance_cifar10_augm_%3.2f.ckpt'%(frac_noisy))
        importance = np.load('./results/importance_cifar10_augm_%3.2f.npy'%(frac_noisy))

    ## See the histogram
    #fig = plt.figure()
    #plt.hist(importance,100)
    #plt.show()
            

    ## Retrain and measure test error
    ## Train with est'd clean+val data
    if False:
        ind = np.where(importance>0.5)[0]
        sess.run(tf.global_variables_initializer())    
        acc_train,_ = bl_imp.train_simple(np.concatenate((X_val,X_train_new[ind,:]),0), np.concatenate((Y_val,Y_train_new[ind,:]),0),200,X_test,Y_test)
        #acc_test,_ = bl_imp.eval_simple(X_test,Y_test)

    sess.close()

    return 

## frac_noisy = 0.3:70% -> 82%,  0.5:66% -> 80% ?


##############################################################################################################

if __name__ == '__main__':

    tf.app.run()


