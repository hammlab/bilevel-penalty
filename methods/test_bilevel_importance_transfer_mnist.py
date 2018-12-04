## test_bilevel_importance_transfer_mnist.py

## min_{al,cls_te} ErrTest(cls_te,filt) s.t. filt,cls_tr = argmin ErrTrain(al,cls_tr,filt)



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

import logging
import os
#from cleverhans.attacks import CarliniWagnerL2
#from cleverhans.utils import AccuracyReport #pair_visual, grid_visual, 
#from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
#from cleverhans.utils_tf import model_train, model_eval, tf_model_load, model_argmax
#from cleverhans_tutorials.tutorial_models import *

#from bilevel_meta_penalty import bilevel_meta
#from bilevel_importance_transfer import bilevel_importance_transfer
from bilevel_importance_transfer_multivar import bilevel_importance_transfer
import time

#############################################################################################################

lr_al = 1E-1
lr_u = 1E-4
lr_v = 1E-4

nepochs = 1001
niter = 1
batch_size = 500

rho_0 = 1E-1
lamb_0 = 1E0
eps_0 = 1E0

c_rho = 2
c_lamb = 0.5
c_eps = 0.5

dim_filt = 128

height = 28
width = 28
nch = 1
num_class = 10

'''
def make_filt(ins,d=dim_filt): 
    W1 = tf.get_variable('W1',[3,3,1,32],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[32],initializer=tf.constant_initializer(0.0))
    c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ins, W1, strides=[1,1,1,1], padding='SAME'),b1))
    p1 = tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')        
    W2 = tf.get_variable('W2',[3,3,32,32],initializer=tf.random_normal_initializer(stddev=0.01))
    b2 = tf.get_variable('b2',[32],initializer=tf.constant_initializer(0.0))
    c2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p1, W2, strides=[1,1,1,1], padding='SAME'), b2))
    p2 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')        
    #W3 = tf.get_variable('W3',[3,3,32,32],initializer=tf.random_normal_initializer(stddev=0.01))
    #b3 = tf.get_variable('b3',[32],initializer=tf.constant_initializer(0.0))
    #c3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p3, W3, strides=[1,1,1,1], padding='SAME'), b3))
    #p3 = tf.nn.max_pool(c3, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')        
    W4 = tf.get_variable('W4',[7*7*32,d],initializer=tf.random_normal_initializer(stddev=0.01))
    b4 = tf.get_variable('b4',[d],initializer=tf.constant_initializer(0.0))
    a4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.reshape(p2,[-1,7*7*32]),W4),b4))
    out = a4
    return out
'''

def make_filt(ins,d=dim_filt):
    W1 = tf.get_variable('W1',[5,5,1,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[64],initializer=tf.constant_initializer(0.0))
    c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ins, W1, strides=[1,1,1,1], padding='SAME'),b1))
    p1 = tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    W2 = tf.get_variable('W2',[5,5,64,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b2 = tf.get_variable('b2',[64],initializer=tf.constant_initializer(0.0))
    c2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p1, W2, strides=[1,1,1,1], padding='SAME'), b2))
    p2 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    a2 = tf.reshape(p2,[-1,7*7*64])
    W3 = tf.get_variable('W3',[7*7*64,d],initializer=tf.random_normal_initializer(stddev=0.01))
    b3 = tf.get_variable('b3',[d],initializer=tf.constant_initializer(0.0))
    a3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(a2,W3),b3))
    #W4 = tf.get_variable('W4',[nh,K],initializer=tf.random_normal_initializer(stddev=0.01))
    #b4 = tf.get_variable('b4',[K],initializer=tf.constant_initializer(0.0))
    #out = tf.nn.bias_add(tf.matmul(a3,W4),b4)
    out = a3
    #reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)
    return out


def make_cls(ins,d=dim_filt,num_class=num_class): 
    W1 = tf.get_variable('W1',[d,num_class],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[num_class],initializer=tf.constant_initializer(0.0))
    a1 = tf.matmul(ins,W1) + b1
    out = a1
    return out



def main(argv=None):

    #shape = [FLAGS.batch_size,img_rows,img_cols,channels]
    tf.set_random_seed(1234)
    sess = tf.Session()
    #set_log_level(logging.DEBUG)

    ## Read data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=0, train_end=60000,test_start=0,test_end=10000)
    Ntrain = X_train.shape[0]
    Ntest = X_test.shape[0]
    
    ## Noisy data
    X_train_clean = X_train[:2500,:]
    Y_train_clean = Y_train[:2500,:]
    X_train_noise = X_train[2500:5000,:] 
    # Random label
    #Y_train_noise = Y_train[4000:5000,:]
    tind = np.random.choice(10,2500)
    Y_train_noise = np.zeros((2500,10))
    Y_train_noise[np.arange(2500),tind] = 1

    X_train_new = np.concatenate((X_train_clean,X_train_noise),0)
    Y_train_new = np.concatenate((Y_train_clean,Y_train_noise),0)    
    Ntrain_new = 5000
    
    Nval = 1000
    X_val = X_train[Ntrain_new:Ntrain_new+Nval,:]
    Y_val = Y_train[Ntrain_new:Ntrain_new+Nval,:]
    
    ## Define model
    x_train_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_train_tf = tf.placeholder(tf.float32, shape=(batch_size,num_class))
    x_test_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_test_tf = tf.placeholder(tf.float32, shape=(batch_size,num_class))

    with tf.variable_scope('filt',reuse=False):
        filt_train = make_filt(x_train_tf)
    with tf.variable_scope('filt',reuse=True):
        filt_test = make_filt(x_test_tf)        
    var_filt = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='filt')

    with tf.variable_scope('cls_train',reuse=False):
        cls_train = make_cls(filt_train)
    var_cls_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='cls_train')
    with tf.variable_scope('cls_test',reuse=False):
        cls_test = make_cls(filt_test)
    var_cls_test = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='cls_test')

    #saver_model = tf.train.Saver(var_model,max_to_keep=None)

    
    #########################################################################################################
    ## Bilevel training
    #########################################################################################################

    blit = bilevel_importance_transfer(sess,x_train_tf,x_test_tf,y_train_tf,y_test_tf,filt_train,filt_test,cls_train,cls_test,
        var_filt,var_cls_train,var_cls_test,batch_size,lr_al,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)

    ## Should I reset the model or use the pretrained model?
    sess.run(tf.global_variables_initializer())
    importance_atanh = np.zeros((Ntrain_new))    

    blit.train_lower_simple(X_train_new,Y_train_new,200)
    blit.train_upper_simple(X_val,Y_val,200)
    blit.eval_upper_simple(X_test,Y_test)

    sess.run(tf.global_variables_initializer())
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
                f,gvnorm,lamb_g,timp_atanh = blit.train(X_train_new[ind_tr,:],Y_train_new[ind_tr,:],X_val[ind_val,:],Y_val[ind_val,:],importance_atanh[ind_tr],niter)
                importance_atanh[ind_tr] = timp_atanh

            ## Renormalize importance_atanh
            if True:
                importance = 0.5*(np.tanh(importance_atanh)+1.)
                importance = 0.5*importance/np.mean(importance)
                importance = np.maximum(.00001,np.minimum(.99999,importance))
                importance_atanh = np.arctanh(2.*importance-1.)
            
            if epoch%10==0:
                rho_t,lamb_t,eps_t = sess.run([blit.bl.rho_t,blit.bl.lamb_t,blit.bl.eps_t])
                print('epoch %d (rho=%f, lamb=%f, eps=%f): h=%f + %f + %f= %f'%
                    (epoch,rho_t,lamb_t,eps_t,f,gvnorm,lamb_g,f+gvnorm+lamb_g))
                blit.eval_upper_simple(X_test,Y_test)
            
        #saver_model.save(sess,'./model_bilevel_importance_transfer_mnist.ckpt')
        #importance = 0.5*(np.tanh(importance_atanh)+1.)
        #np.save('./importance_transfer_mnist.npy',importance)
        #np.save(os.path.join(result_dir,')),tX)        
    else:
        pass
        #saver_model.restore(sess,'./model_bilevel_importance_transfer_mnist.ckpt')
        #importance = np.load('./importance_transfer_mnist.npy')

    ## See the histogram
    fig = plt.figure()
    plt.hist(importance,100)
    plt.show()
            

    ## Retrain and measure test error
    ## Train with est'd clean+val data
    if True:
        ind = np.where(importance>0.9)[0]
        sess.run(tf.global_variables_initializer())    
        acc_train,_ = blit.train_simple(np.concatenate((X_val,X_train_new[ind,:]),0), np.concatenate((Y_val,Y_train_new[ind,:]),0),100)
        acc_test,_ = blit.eval_simple(X_test,Y_test)
    
    sess.close()

    return 


##############################################################################################################

if __name__ == '__main__':

    tf.app.run()

