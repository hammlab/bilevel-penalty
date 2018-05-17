## test_bilevel_l2reg_simple_mnist.py

# min_s sum_val logreg(y*x^Tw) s.t. w = argmin sum_train logreg(yx^Tw) + exp(s)*\|w\|^2


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import sys
sys.path.append('/home/hammj/Dropbox/Research/AdversarialLearning/codes/lib/cleverhans-master')

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from cleverhans.utils_mnist import data_mnist

from bilevel_l2reg_simple import bilevel_l2reg_simple

import time

lr_u = 1E-2
lr_v = 1E-3

nepochs = 201
niter = 1
batch_size = 500#128

rho_0 = 1E0
lamb_0 = 1E0
eps_0 = 1E0

c_rho = 2
c_lamb = 0.5
c_eps = 0.5


def make_classifier(ins,K=10):
    d = np.prod(ins.shape[1:])
    W1 = tf.get_variable('W1',[d,K],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[K],initializer=tf.constant_initializer(0.0))
    return tf.nn.bias_add(tf.matmul(tf.reshape(ins,[-1,d]),W1),b1)


def main(argv=None):

    tf.set_random_seed(1234)
    sess = tf.Session()

    ## Read data
    height = 28
    width = 28
    nch = 1
    nclass = 10
    #num_class = 10
    
    tX, tY, X_test, Y_test = data_mnist(train_start=0, train_end=60000,test_start=0,test_end=10000)
    X_train = tX[:50000,:]
    Y_train = tY[:50000,:]
    X_val = tX[50000:60000,:]
    Y_val = tY[50000:60000,:]
    Ntrain = X_train.shape[0]
    Nval = X_val.shape[0]
    Ntest = X_test.shape[0]

    ## Define model
    #print('\n\nDefining models:')
    x_train_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_train_tf = tf.placeholder(tf.float32, shape=(batch_size,nclass))
    x_test_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_test_tf = tf.placeholder(tf.float32, shape=(batch_size,nclass))

    with tf.variable_scope('cls',reuse=False):
        cls_train = make_classifier(x_train_tf)
    var_cls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='cls')
    with tf.variable_scope('cls',reuse=True):
        cls_test = make_classifier(x_test_tf)
    #print('Done')
    #saver_model = tf.train.Saver(var_model,max_to_keep=none)



    #########################################################################################################
    ## Bilevel training
    #########################################################################################################

    #print('\n\nSetting up graphs:')
    bl_l2reg = bilevel_l2reg_simple(sess,x_train_tf,x_test_tf,y_train_tf,y_test_tf,cls_train,cls_test,var_cls,
        batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)
    #print('Done')
    sess.run(tf.global_variables_initializer())

    ## Metatrain
    #print('\n\nTraining start:')
    if True:
        for epoch in range(nepochs):
            #tick = time.time()        
            nb_batches = int(np.floor(float(Ntrain) / batch_size))
            ind_shuf = np.arange(Ntrain)
            np.random.shuffle(ind_shuf)
            for batch in range(nb_batches):
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntrain))
                ind_tr = ind_shuf[ind_batch]
                ind_val = np.random.choice(Nval, size=(batch_size), replace=False)
                f,gvnorm,lamb_g,l2reg = bl_l2reg.train(X_train[ind_tr,:],Y_train[ind_tr,:],X_val[ind_val,:],Y_val[ind_val,:],niter)

            if epoch%10==0:
                rho_t,lamb_t,eps_t = sess.run([bl_l2reg.bl.rho_t,bl_l2reg.bl.lamb_t,bl_l2reg.bl.eps_t])
                #sig = sess.run(bl.sig)
                print('epoch %d (rho=%f, lamb=%f, eps=%f): h=%f + %f + %f= %f, l2reg=%f'%
                    (epoch,rho_t,lamb_t,eps_t,f,gvnorm,lamb_g,f+gvnorm+lamb_g,l2reg))

            if epoch%50==0:
                ## Measure test error
                nb_batches = int(np.floor(float(Ntest)/batch_size))
                acc = 0#np.nan*np.ones(nb_batches)
                for batch in range(nb_batches):
                    ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntest))
                    pred = sess.run(cls_test, {x_test_tf:X_test[ind_batch,:]})
                    acc += np.sum(np.argmax(pred,1)==np.argmax(Y_test[ind_batch,:],1))
                acc /= np.float32(nb_batches*batch_size)
                print('mean acc = %f\n'%(acc))


    # 0.93


    #########################################################################################################
    ## Single-level training
    #########################################################################################################
    ## rho=0, lamb fixed

    if True:
        lambs = [1E-2, 1E-1, 1E0, 1E1, 1E2]
        for lamb in lambs:
            print('\nlamb=%f'%(lamb))
            bl_l2reg = bilevel_l2reg_simple(sess,x_train_tf,x_test_tf,y_train_tf,y_test_tf,cls_train,cls_test,var_cls,
                batch_size,lr_u,lr_v,0.,lamb,0.,1.,1.,1.)
            sess.run(tf.global_variables_initializer())
            for epoch in range(nepochs):
                #tick = time.time()        
                nb_batches = int(np.floor(float(Ntrain)/batch_size))
                ind_shuf = np.arange(Ntrain)
                np.random.shuffle(ind_shuf)
                for batch in range(nb_batches):
                    ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntrain))
                    ind_tr = ind_shuf[ind_batch]
                    ind_val = np.random.choice(Nval,size=(batch_size),replace=False)
                    l = bl_l2reg.train_singlelevel(X_train[ind_tr,:],Y_train[ind_tr,:],X_val[ind_val,:],Y_val[ind_val,:],niter)

                if epoch%50==0:
                    ## Measure test error
                    nb_batches = int(np.floor(float(Ntest)/batch_size))
                    acc = 0#np.nan*np.ones(nb_batches)
                    for batch in range(nb_batches):
                        ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntest))
                        pred = sess.run(cls_test, {x_test_tf:X_test[ind_batch,:]})
                        acc += np.sum(np.argmax(pred,1)==np.argmax(Y_test[ind_batch,:],1))
                    acc /= np.float32(nb_batches*batch_size)
                    print('mean acc = %f\n'%(acc))









                
    sess.close()




##############################################################################################################

if __name__ == '__main__':

    tf.app.run()



