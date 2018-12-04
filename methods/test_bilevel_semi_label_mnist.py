## test_bilevel_semi_label_mnist.py

## min_Y ErrVal(w) s.t. w=argmin ErrTr(Y;w)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import sys
#sys.path.append('/home/hammj/Dropbox/Research/AdversarialLearning/codes/lib/cleverhans-master')
sys.path.append('/home/hammj/Dropbox/Research/AdversarialLearning/codes/script/bilvel')

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import keras
#from keras.datasets import cifar10, cifar100
from keras.datasets import mnist


#from cleverhans.utils_mnist import data_mnist

from bilevel_semi_label import *

import time

lr_u = 1E-1
lr_v = 1E-3

nepochs = 401
niter = 1
batch_size = 500#128

rho_0 = 1E0#0.#1E0
lamb_0 = 1E0
eps_0 = 1E0

c_rho = 2
c_lamb = 0.5
c_eps = 0.5


def make_classifier1(ins,K=10):
    d = np.prod(ins.shape[1:])
    W1 = tf.get_variable('W1',[d,K],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[K],initializer=tf.constant_initializer(0.0))
    return tf.nn.bias_add(tf.matmul(tf.reshape(ins,[-1,d]),W1),b1)

def make_classifier2(ins,K=10,nh=100):
    W1 = tf.get_variable('W1',[5,5,1,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[64],initializer=tf.constant_initializer(0.0))
    c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ins, W1, strides=[1,1,1,1], padding='SAME'),b1))
    p1 = tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    W2 = tf.get_variable('W2',[5,5,64,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b2 = tf.get_variable('b2',[64],initializer=tf.constant_initializer(0.0))
    c2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p1, W2, strides=[1,1,1,1], padding='SAME'), b2))
    p2 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    a2 = tf.reshape(p2,[-1,7*7*64])
    W3 = tf.get_variable('W3',[7*7*64,nh],initializer=tf.random_normal_initializer(stddev=0.01))
    b3 = tf.get_variable('b3',[nh],initializer=tf.constant_initializer(0.0))
    a3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(a2,W3),b3))
    W4 = tf.get_variable('W4',[nh,K],initializer=tf.random_normal_initializer(stddev=0.01))
    b4 = tf.get_variable('b4',[K],initializer=tf.constant_initializer(0.0))
    out = tf.nn.bias_add(tf.matmul(a3,W4),b4)
    #reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)
    return out


def main(argv=None):

    tf.set_random_seed(1234)
    sess = tf.Session()

    ## Read data
    height = 28
    width = 28
    nch = 1
    nclass = 2
    #num_class = 10
    
    ## Load data
    #tX, tY, tX_test, tY_test = data_mnist(train_start=0, train_end=60000,test_start=0,test_end=10000)
    (tX, ty), (tX_test, ty_test) = mnist.load_data()
    tX = tX.reshape((-1,28,28,1)).astype('float32')/255.
    tX_test = tX_test.reshape((-1,28,28,1)).astype('float32')/255.
    tY = keras.utils.to_categorical(ty,10)
    tY_test = keras.utils.to_categorical(ty_test,10)


    Ntrain = 10000
    Nval = 500
    Ntest = 10000
    X_train = tX[:Ntrain,:]
    #Y_train = tY[:5000,:]
    Y_train = np.zeros((Ntrain,nclass))
    Y_train[np.arange(Ntrain),np.argmax(tY[:Ntrain,:],1)%nclass] = 1.
    X_val = tX[50000:(50000+Nval),:]
    #Y_val = tY[50000:50500,:]
    Y_val = np.zeros((Nval,nclass))
    Y_val[np.arange(Nval),np.argmax(tY[50000:(50000+Nval),:],1)%nclass] = 1.
    #Ntrain = X_train.shape[0]
    #Nval = X_val.shape[0]
    X_test = tX_test[:Ntest,:]
    Y_test = np.zeros((Ntest,nclass))
    Y_test[np.arange(Ntest),np.argmax(tY_test[:Ntest,:],1)%nclass] = 1.

    ## Define model
    #print('\n\nDefining models:')
    x_train_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_train_logit_tf = tf.placeholder(tf.float32, shape=(batch_size,nclass))
    x_test_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_test_tf = tf.placeholder(tf.float32, shape=(batch_size,nclass))

    with tf.variable_scope('cls',reuse=False):
        cls_train = make_classifier2(x_train_tf,nclass)
    var_cls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='cls')
    with tf.variable_scope('cls',reuse=True):
        cls_test = make_classifier2(x_test_tf,nclass)
    #print('Done')
    #saver_model = tf.train.Saver(var_model,max_to_keep=none)

    #########################################################################################################
    ## Bilevel training
    #########################################################################################################

    #print('\n\nSetting up graphs:')
    bl_semi = bilevel_semi_label(sess,x_train_tf,x_test_tf,y_train_logit_tf,y_test_tf,cls_train,cls_test,var_cls,
        batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)
    #print('Done')
    sess.run(tf.global_variables_initializer())

    bl_semi.train_simple(X_val,Y_val,200) ## Train with val data
    print('test error - val only')
    bl_semi.eval_simple(X_test,Y_test)

    if False:
        y_train_logit = np.zeros((Ntrain,nclass),np.float32)
    else:
        # Predictions based on valid-pretrained model?
        print('test error on training data')
        _,y_train_init = bl_semi.eval_simple(X_train,Y_train)
        y_train_init = np.maximum(.001,np.minimum(.999,y_train_init))
        y_train_logit = np.log(y_train_init)

    ## Metatrain
    #print('\n\nTraining start:')
    if True:
        for epoch in range(nepochs):
            #tick = time.time()        
            nb_batches = int(np.floor(float(Ntrain)/batch_size))
            ind_shuf = np.arange(Ntrain)
            np.random.shuffle(ind_shuf)
            for batch in range(nb_batches):
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntrain))
                ind_tr = ind_shuf[ind_batch]
                ind_val = np.random.choice(Nval, size=(batch_size), replace=False)
                f,gvnorm,lamb_g,ty_logit = bl_semi.train(X_train[ind_tr,:],y_train_logit[ind_tr,:],X_val[ind_val,:],Y_val[ind_val,:],niter)
                y_train_logit[ind_tr,:] = ty_logit

            if epoch%10==0:
                rho_t,lamb_t,eps_t = sess.run([bl_semi.bl.rho_t,bl_semi.bl.lamb_t,bl_semi.bl.eps_t])
                print('epoch %d (rho=%f, lamb=%f, eps=%f): h=%f + %f + %f= %f'%
                    (epoch,rho_t,lamb_t,eps_t,f,gvnorm,lamb_g,f+gvnorm+lamb_g))
            if epoch%10==0:
                acc = np.mean(np.argmax(y_train_logit,1)==np.argmax(Y_train,1))
                print('acc of y_train=%f'%(acc))

            if epoch%50==0:
                print('test error')
                acc,_ = bl_semi.eval_simple(X_test,Y_test)
                
                
    sess.close()




##############################################################################################################

if __name__ == '__main__':

    tf.app.run()



