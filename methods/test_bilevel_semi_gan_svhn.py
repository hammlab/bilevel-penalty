## test_bilevel_semi_label_svhn.py

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
#from keras.datasets import svhn
from scipy.io import loadmat

from svhn_gan import discriminator, generator

#from cleverhans.utils_mnist import data_mnist

from bilevel_semi_gan import *

import time

lr_u = 1E-3
lr_v = 1E-3

nepochs = 101
niter = 1
batch_size = 100#128

rho_0 = 1E1
lamb_0 = 0.#1E0
eps_0 = 1E0

c_rho = 2
c_lamb = 0.5
c_eps = 0.5


height = 32
width = 32
nch = 3
num_class = 10

dim_gen = 100

istraining_ph = tf.placeholder_with_default(True,shape=())


def main(argv=None):

    tf.set_random_seed(1234)
    sess = tf.Session()

    ## Read data
    train_data = loadmat('train_32x32.mat')
    tX = train_data['X'].transpose((3,0,1,2))
    ty = train_data['y'].flatten()
    ty[ty==10] = 0
    del train_data

    test_data = loadmat('test_32x32.mat')
    tX_test = test_data['X'].transpose((3,0,1,2))
    ty_test = test_data['y'].flatten()
    ty_test[ty_test==10] = 0
    del test_data

    #(tX, ty), (tX_test, ty_test) = mnist.load_data()
    tX = 2*(tX/255.)-1
    tX_test = 2*(tX_test/255)-1
    tY = keras.utils.to_categorical(ty,num_class)
    tY_test = keras.utils.to_categorical(ty_test,num_class)

    Nval = 1000
    Ntrain = 72000  #73257
    Ntest = 26000  #26032
    X_train = tX[:Ntrain,:]
    #Y_train = tY[:5000,:]
    Y_train = np.zeros((Ntrain,num_class))
    Y_train[np.arange(Ntrain),np.argmax(tY[:Ntrain,:],1)%num_class] = 1.
    X_val = tX[Ntrain:(Ntrain+Nval),:]
    #Y_val = tY[50000:50500,:]
    Y_val = np.zeros((Nval,num_class))
    Y_val[np.arange(Nval),np.argmax(tY[Ntrain:(Ntrain+Nval),:],1)%num_class] = 1.
    #Ntrain = X_train.shape[0]
    #Nval = X_val.shape[0]
    X_test = tX_test[:Ntest,:]
    Y_test = np.zeros((Ntest,num_class))
    Y_test[np.arange(Ntest),np.argmax(tY_test[:Ntest,:],1)%num_class] = 1.

    #print(X_train.shape)
    #print(X_val.shape)
    #print(X_test.shape)
    #print(Y_train.shape)
    #print(Y_val.shape)
    #print(Y_test.shape)

    ## Define model
    #print('\n\nDefining models:')
    x_val_ph = tf.placeholder(tf.float32, shape = [batch_size, height ,width, nch], name = 'x_val_ph')
    x_train_ph = tf.placeholder(tf.float32, shape = [batch_size, height ,width, nch], name = 'x_train_ph')
    #z_ph = tf.placeholder(tf.float32, shape = [batch_size, 1, 1, dim_gen], name = 'z_ph')
    y_val_ph = tf.placeholder(tf.float32, name = 'y_val_ph', shape = [batch_size, num_class])
    y_train_ph = tf.placeholder(tf.float32, name = 'y_train_ph', shape = [batch_size, num_class])
    
    #acc_train_pl = tf.placeholder(tf.float32, [], 'acc_train_pl')
    #acc_test_pl = tf.placeholder(tf.float32, [], 'acc_test_pl')
    #acc_test_pl_ema = tf.placeholder(tf.float32, [], 'acc_test_pl')

    random_z = tf.random_uniform([batch_size, dim_gen], name='random_z')
    generator(random_z, istraining_ph, init=True)  # init of weightnorm weights
    gen_inp = generator(random_z, istraining_ph, init=False, reuse=True)

    discriminator(x_train_ph, istraining_ph, init=True)
    logits_val, _ = discriminator(x_val_ph, istraining_ph, init=False, reuse=True)
    logits_gen, feat_gen = discriminator(gen_inp, istraining_ph, init=False, reuse=True)
    logits_train, feat_train = discriminator(x_train_ph, istraining_ph, init=False, reuse=True)

    tvars = tf.trainable_variables()
    var_disc = [var for var in tvars if 'discriminator_model' in var.name]
    var_gen = [var for var in tvars if 'generator_model' in var.name]

    #########################################################################################################
    ## Bilevel training
    #########################################################################################################

    #print('\n\nSetting up graphs:')
    bl = bilevel_semi_gan(sess,x_train_ph,x_val_ph,y_train_ph,y_val_ph,
        logits_train,logits_val,logits_gen,feat_train,feat_gen,
        var_disc,var_gen,
        batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps,istraining_ph)
    #print('Done')
    sess.run(tf.global_variables_initializer())

    bl.train_simple(X_val,Y_val,300) ## Train with val data
    print('test acc - val only:')
    bl.eval_simple(X_test,Y_test)

    ## Metatrain
    #print('\n\nTraining start:')
    if True:
        for epoch in range(nepochs):
            #tick = time.time()        
            nb_batches = int(np.floor(float(Ntrain)/batch_size))
            ind_shuf = np.arange(Ntrain)
            np.random.shuffle(ind_shuf)
            for batch in range(nb_batches):
                #print('%d/%d'%(batch+1,nb_batches))
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntrain))
                ind_tr = ind_shuf[ind_batch]
                ind_val = np.random.choice(Nval, size=(batch_size), replace=False)
                #f,gvnorm,lamb_g = bl.train(X_train[ind_tr,:],X_val[ind_val,:],Y_val[ind_val,:],niter)
                if epoch<10: # Burn-in ?
                    f,g = bl.update_alternating(X_train[ind_tr,:],X_val[ind_val,:],Y_val[ind_val,:])
                    #f,g = bl.update_singlelevel(X_train[ind_tr,:],X_val[ind_val,:],Y_val[ind_val,:])                    
                else:
                    f,gvnorm,lamb_g = bl.update(X_train[ind_tr,:],X_val[ind_val,:],Y_val[ind_val,:])
                    

            if epoch<10:#%1==0:
                print('epoch %d: f=%f, g=%f'%(epoch,f,g))
            else:
                rho_t,lamb_t,eps_t = sess.run([bl.bl.rho_t,bl.bl.lamb_t,bl.bl.eps_t])
                print('epoch %d (rho=%f, lamb=%f, eps=%f): h=%f + %f + %f= %f'%
                    (epoch,rho_t,lamb_t,eps_t,f,gvnorm,lamb_g,f+gvnorm+lamb_g))

            if epoch%1==0:
                #print('epoch %d/%d'%(epoch+1,nepochs))
                #acc = np.mean(np.argmax(y_train_logit,1)==np.argmax(Y_train,1))
                #print('(unlabeled) training acc:%f'%(acc))
                #if epoch%50==0:
                #print('train acc:')
                #acc,_ = bl.eval_simple(X_train,Y_train)                
                print('test acc:')
                acc,_ = bl.eval_simple(X_test,Y_test)
                
                
    sess.close()

    ## (1000 labeled): Goal is 92% or higher. It reaches 91%, but decreases/unstable with lr=1E-3


##############################################################################################################

if __name__ == '__main__':

    tf.app.run()



