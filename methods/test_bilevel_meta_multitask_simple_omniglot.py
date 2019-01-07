
'''
u is the common network parameter
v is the individual parameter
min_u 1/N sum_i ValErr(vi(u)) s.t. vi(u) = argmin TrainErr(vi) + 0.5*gamma*\|vi-u\|^2
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import logging
import os

#from bilevel_meta_penalty_batch_increm import bilevel_meta
from bilevel_meta_multitask_simple import bilevel_meta
import time

'''
sys.path.append('/home/hammj/Dropbox/Research/AdversarialLearning/codes/lib/maml-master')
from data_generator import DataGenerator

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'omniglot', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
#flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
#flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
#flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
#flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
#flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

## Model options
#flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
#flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
#flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
#flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
#flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
#flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
#flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
#flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
#flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
#flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for 
'''

#############################################################################################################

lr_u = 1E-3
#lr_a = 1E-10
lr_v = 1E-3

nepochs = 50
niter = 20
niter_simple = 20
gamma = 1E-2

rho_0 = 1E1
lamb_0 = 1E2
eps_0 = 1E0

c_rho = 2
c_lamb = 0.5
c_eps = 0.5


height = 28
width = 28
nch = 1
#num_class = 5
n = 32460 # 1623 characters * 20 instances

nclass_per_task = 5# N-way. Same for training and testing    
ntrain_per_cls = 5 # K-shot
ntest_per_cls = 15
ntrain_per_task = nclass_per_task*ntrain_per_cls # 25
ntest_per_task = nclass_per_task*ntest_per_cls # 75
n_per_task = ntrain_per_task + ntest_per_task # 100
ntask_per_batch = np.int(1000./n_per_task)
ntask_metatrain = np.int((0.75*n*4)/n_per_task)
ntask_metatest = np.int((0.25*n)/n_per_task)
nbatch_train = int(np.floor(float(ntask_metatrain) / ntask_per_batch))
nbatch_test = int(np.floor(float(ntask_metatest) / ntask_per_batch))

print('ntask=%d'%(ntask_metatrain))
print('nbatch=%d'%(nbatch_train))
print('ntask_per_batch=%d'%(ntask_per_batch))
#ntask=649
#nbatch=64
#ntask_per_batch=10

istraining_ph = tf.placeholder_with_default(True,shape=())



'''
def make_classifier(ins,K=nclass_per_task,nh=100):
    W1 = tf.get_variable('W1',[3,3,1,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[64],initializer=tf.constant_initializer(0.0))
    c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ins, W1, strides=[1,1,1,1], padding='SAME'),b1))
    p1 = tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    W2 = tf.get_variable('W2',[3,3,64,64],initializer=tf.random_normal_initializer(stddev=0.01))
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
'''


def make_classifier2(ins,K=nclass_per_task):
    W1 = tf.get_variable('W1',[3,3,1,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[64],initializer=tf.constant_initializer(0.0))
    c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ins, W1, strides=[1,1,1,1], padding='SAME'),b1))
    p1 = tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')
    p1 = tf.layers.dropout(p1, rate=0.25, training=istraining_ph)

    W2 = tf.get_variable('W2',[3,3,64,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b2 = tf.get_variable('b2',[64],initializer=tf.constant_initializer(0.0))
    c2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p1, W2, strides=[1,1,1,1], padding='SAME'), b2))
    p2 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')
    p2 = tf.layers.dropout(p2, rate=0.25, training=istraining_ph)
    
    W3 = tf.get_variable('W3',[3,3,64,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b3 = tf.get_variable('b3',[64],initializer=tf.constant_initializer(0.0))
    c3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p2, W3, strides=[1,1,1,1], padding='SAME'), b3))
    p3 = tf.nn.max_pool(c3, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')
    p3 = tf.layers.dropout(p3, rate=0.25, training=istraining_ph)    
    
    W4 = tf.get_variable('W4',[3,3,64,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b4 = tf.get_variable('b4',[64],initializer=tf.constant_initializer(0.0))
    c4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p3, W4, strides=[1,1,1,1], padding='SAME'), b4))
    p4 = tf.nn.max_pool(c4, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID')
    p4 = tf.layers.dropout(p4, rate=0.25, training=istraining_ph)    
    a4 = tf.reshape(p4,[-1,64])
    
    W5 = tf.get_variable('W5',[64,K],initializer=tf.random_normal_initializer(stddev=0.01))
    b5 = tf.get_variable('b5',[K],initializer=tf.constant_initializer(0.0))
    out = tf.nn.bias_add(tf.matmul(a4,W5),b5)

    #reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)
    return out

'''
def make_classifier3(ins,d1=512,d2=512,K=10):#256):
    W1 = tf.get_variable('W1',[5,5,1,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable('b1',[64],initializer=tf.constant_initializer(0.0))
    c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(ins, W1, strides=[1,1,1,1], padding='SAME'),b1))
    p1 = tf.nn.max_pool(c1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    p1 = tf.layers.dropout(p1, rate=0.25, training=istraining_ph)

    W2 = tf.get_variable('W2',[5,5,64,64],initializer=tf.random_normal_initializer(stddev=0.01))
    b2 = tf.get_variable('b2',[64],initializer=tf.constant_initializer(0.0))
    c2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(p1, W2, strides=[1,1,1,1], padding='SAME'), b2))
    p2 = tf.nn.max_pool(c2, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    p2 = tf.layers.dropout(p2, rate=0.25, training=istraining_ph)
    a2 = tf.reshape(p2,[-1,7*7*64])

    W3 = tf.get_variable('W3',[7*7*64,d1],initializer=tf.random_normal_initializer(stddev=0.01))
    b3 = tf.get_variable('b3',[d1],initializer=tf.constant_initializer(0.0))
    a3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(a2,W3),b3))
    a3 = tf.layers.dropout(a3, rate=0.5, training=istraining_ph)

    #W4 = tf.get_variable('W4',[d1,d2],initializer=tf.random_normal_initializer(stddev=0.01))
    #b4 = tf.get_variable('b4',[d2],initializer=tf.constant_initializer(0.0))
    #a4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(a3,W4),b4))
    #a4 = tf.layers.dropout(a4, rate=0.5, training=istraining_ph)

    W5 = tf.get_variable('W5',[d2,K],initializer=tf.random_normal_initializer(stddev=0.01))
    b5 = tf.get_variable('b5',[K],initializer=tf.constant_initializer(0.0))
    a5 = tf.nn.bias_add(tf.matmul(a3,W5),b5)

    out = a5#3
    #reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4)
    return out
'''

def generate_splits_omniglot():
    ## Read data
    if False:
        import os
        from PIL import Image
        n = 32460    
        X = np.zeros((n,28,28),np.float32)
        y = np.zeros((n),np.int32)

        dir_data = './data/omniglot_resized'
        cnt = 0
        nclass = 0
        for d1 in os.listdir(dir_data):
            print('%s'%(d1))
            for d2 in os.listdir(dir_data+'/'+d1):
                for d3 in os.listdir(dir_data+'/'+d1+'/'+d2):
                    fname = dir_data+'/'+d1+'/'+d2+'/'+d3
                    img = Image.open(fname)
                    #X[cnt,:,:] = np.asarray(img)
                    X[cnt,:,:] = np.array(img.getdata(),np.float32).reshape((28,28))#.reshape(img.size[1],img.size[0])
                    y[cnt] = nclass
                    cnt += 1
                nclass += 1    
        assert(n==cnt)
        print(' ')
        print('n=%d'%(n)) # 32460
        print('nclass=%d'%(nclass)) #1623
        X = X.reshape(n,28,28,1)
        np.save('./x_omniglot.npy',X)
        np.save('./y_omniglot.npy',y)
    else:
        X = np.load('./x_omniglot.npy')
        y = np.load('./y_omniglot.npy')
        nclass = 1623
        n = 32460

    ## Create augmented data for 1200 classes
    Xaug = np.zeros((4,n,28,28,1),np.float32)
    Xaug[0,:] = X
    Xaug[1,:] = np.flip(X,1)
    Xaug[2,:] = np.flip(X,2)
    Xaug[3,:] = np.flip(np.flip(X,1),2)

    ######################################################################################################################
    ## Create meta datasets

    ## Sample meta-training/meta-valid/meta-test classes
    ind_cls_shuf = np.arange(nclass)
    np.random.shuffle(ind_cls_shuf)
    ind_cls_train = ind_cls_shuf[:1200]
    ind_cls_test = ind_cls_shuf[1200:]

    X_metatrain_train = np.zeros((ntask_metatrain,ntrain_per_task,height,width,nch),np.float32)
    Y_metatrain_train = np.zeros((ntask_metatrain,ntrain_per_task,nclass_per_task),np.float32)
    X_metatrain_test = np.zeros((ntask_metatrain,ntest_per_task,height,width,nch),np.float32)
    Y_metatrain_test = np.zeros((ntask_metatrain,ntest_per_task,nclass_per_task),np.float32)
    X_metatest_train = np.zeros((ntask_metatest,ntrain_per_task,height,width,nch),np.float32)
    Y_metatest_train = np.zeros((ntask_metatest,ntrain_per_task,nclass_per_task),np.float32)
    X_metatest_test = np.zeros((ntask_metatest,ntest_per_task,height,width,nch),np.float32)
    Y_metatest_test = np.zeros((ntask_metatest,ntest_per_task,nclass_per_task),np.float32)

    ## Metatrain-train/test
    for i in range(ntask_metatrain):
        ind_cls = np.random.choice(ind_cls_train,nclass_per_task,replace=False)
        ind3 = []
        tY = np.zeros((n,nclass_per_task))
        ind_rot = []
        for j in range(nclass_per_task): 
            #ind1 = np.where(np.argmax(Y_train,1)==ind_cls[j])[0]
            ind1 = np.where(y==ind_cls[j])[0]            
            ind2 = np.random.choice(ind1,ntrain_per_cls+ntest_per_cls,replace=False).tolist()
            ind3 += ind2
            # One random rotation per class
            ind_rot += np.random.choice(4,1).tolist()*(ntrain_per_cls+ntest_per_cls)
            tY[ind1,j] = 1
        #np.random.shuffle(ind3)
        tX = Xaug[ind_rot,ind3,:]
        tY = tY[ind3,:]
        ind_shuf = np.random.choice(n_per_task,n_per_task,replace=False)
        X_metatrain_train[i,:] = tX[ind_shuf[:ntrain_per_task],:]
        Y_metatrain_train[i,:] = tY[ind_shuf[:ntrain_per_task],:]
        X_metatrain_test[i,:] = tX[ind_shuf[ntrain_per_task:],:]
        Y_metatrain_test[i,:] = tY[ind_shuf[ntrain_per_task:],:]


    if False: ## Check images and labels
        import matplotlib.pyplot as plt    
        print(X_metatrain_train[0].shape)
        print(Y_metatrain_train[0])
        print(Y_metatrain_train[1])    
        print(Y_metatrain_test[0][:5,:])
        print(Y_metatrain_test[1][:5,:])    
        for i in range(10):
            plt.subplot(4,10,i+1)
            plt.imshow(np.squeeze(X_metatrain_train[0][i,:]))
            plt.subplot(4,10,i+11)
            plt.imshow(np.squeeze(X_metatrain_train[0][i+10,:]))
            plt.subplot(4,10,i+21)
            plt.imshow(np.squeeze(X_metatrain_test[0][i,:]))
            plt.subplot(4,10,i+31)
            plt.imshow(np.squeeze(X_metatrain_test[0][i+10,:]))
        plt.show()
    

    ## Metatest-train/test
    for i in range(ntask_metatest):
        ind_cls = np.random.choice(ind_cls_test,nclass_per_task,replace=False)    
        ind3 = []
        tY = np.zeros((n,nclass_per_task))
        for j in range(nclass_per_task):
            #ind1 = np.where(np.argmax(Y_train,1)==ind_cls[j])[0]
            ind1 = np.where(y==ind_cls[j])[0]                        
            ind2 = np.random.choice(ind1,ntrain_per_cls+ntest_per_cls,replace=False).tolist()
            ind3 += ind2
            tY[ind1,j] = 1
        np.random.shuffle(ind3)
        X_metatest_train[i,:] = X[ind3[:ntrain_per_task],:]
        Y_metatest_train[i,:] = tY[ind3[:ntrain_per_task],:]
        X_metatest_test[i,:] = X[ind3[ntrain_per_task:],:]
        Y_metatest_test[i,:] = tY[ind3[ntrain_per_task:],:]

    return [X_metatrain_train, Y_metatrain_train, X_metatrain_test, Y_metatrain_test, 
            X_metatest_train, Y_metatest_train, X_metatest_test, Y_metatest_test]


############################################################################################################################


def main(argv=None):

    tf.set_random_seed(1234)
    sess = tf.Session()
    print('\n\nGenerating data:')
    X_metatrain_train, Y_metatrain_train, X_metatrain_test, Y_metatrain_test, \
            X_metatest_train, Y_metatest_train, X_metatest_test, Y_metatest_test \
            = generate_splits_omniglot()
    print('Done')
    
        
    ## Define models
    print('\n\nDefining models:')
    x_train_ph = tf.placeholder(tf.float32, shape=(ntask_per_batch,ntrain_per_task,height,width,nch))
    y_train_ph = tf.placeholder(tf.float32, shape=(ntask_per_batch,ntrain_per_task,nclass_per_task))
    x_test_ph = tf.placeholder(tf.float32, shape=(ntask_per_batch,ntest_per_task,height,width,nch))
    y_test_ph = tf.placeholder(tf.float32, shape=(ntask_per_batch,ntest_per_task,nclass_per_task))

    #filt_train = [[] for i in range(ntask_per_batch)]
    #filt_test = [[] for i in range(ntask_per_batch)]    
    cls_train = [[] for i in range(ntask_per_batch)]
    cls_test = [[] for i in range(ntask_per_batch)]    
    with tf.variable_scope('cls',reuse=False):
        cls0 = make_classifier2(tf.zeros((ntrain_per_task,height,width,nch)))
    var_cls0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='cls')
    var_cls = [[] for i in range(ntask_per_batch)]
    #var_incr =  [[] for i in range(ntask_per_batch)]
    for i in range(ntask_per_batch):
        with tf.variable_scope('cls'+str(i),reuse=False):
            #cls_train[i] = make_classifier2(tf.squeeze(tf.slice(x_train_ph,[i,0,0,0,0],[1,-1,-1,-1,-1]),0))
            cls_train[i] = make_classifier2(x_train_ph[i])
        with tf.variable_scope('cls'+str(i),reuse=True):
            #cls_test[i] = make_classifier2(tf.squeeze(tf.slice(x_test_ph,[i,0,0,0,0],[1,-1,-1,-1,-1]),0))
            cls_test[i] = make_classifier2(x_test_ph[i])
        var_cls[i] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='cls'+str(i))
    print('Done')

    saver_cls0 = tf.train.Saver(var_cls0,max_to_keep=None)


    #########################################################################################################
    ## Bilevel training
    #########################################################################################################
    
    print('\n\nSetting up graphs:')
    blmt = bilevel_meta(sess,x_train_ph,x_test_ph,y_train_ph,y_test_ph,
        cls_train,cls_test,var_cls0,var_cls,
        ntask_per_batch,ntrain_per_task,ntest_per_task,nclass_per_task,
        gamma,lr_u,lr_v,
        rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps,istraining_ph)

    print('Done')
    ## Should I reset the model or use the pretrained model?
    sess.run(tf.global_variables_initializer())

    ## Metatrain
    print('\n\nMetatraining start:')
    if True:
        for epoch in range(nepochs):
            #tic = time.time()
            index_shuf = np.arange(ntask_metatrain)
            np.random.shuffle(index_shuf)
            for batch in range(nbatch_train):
                ind1 = range(ntask_per_batch*batch,min(ntask_per_batch*(1+batch),ntask_metatrain))
                ind2 = index_shuf[ind1]
                sess.run([blmt.reset_opt_v,blmt.reset_v])
                f,gvnorm,lamb_g = blmt.update(X_metatrain_train[ind2,:],Y_metatrain_train[ind2,:],X_metatrain_test[ind2,:],Y_metatrain_test[ind2,:],niter)
            #print(time.time()-tic)
            
            if epoch%1==0:
                rho_t,lamb_t,eps_t = sess.run([blmt.bl.rho_t,blmt.bl.lamb_t,blmt.bl.eps_t])
                print('epoch %d (rho=%f, lamb=%f, eps=%f): h=%f + %f + %f= %f'%
                    (epoch,rho_t,lamb_t,eps_t,f,gvnorm,lamb_g,f+gvnorm+lamb_g))

            ## Test phase is not bilvel. For each task, simply train with metatest-train and test with metatest-test.

            ## Metatrain-test
            if epoch%1==0:
                print('\nMetatrain-test:')
                accs = np.nan*np.ones(nbatch_train*ntask_per_batch)
                for batch in range(nbatch_train):
                    ind1 = range(ntask_per_batch*batch,min(ntask_per_batch*(1+batch),ntask_metatrain))
                    #blmt.init_var_cls()
                    sess.run([blmt.reset_opt_v,blmt.reset_v])
                    l1 = blmt.update_cls_simple(X_metatrain_train[ind1,:],Y_metatrain_train[ind1,:],niter_simple)
                    pred = sess.run(cls_test, {x_test_ph:X_metatrain_test[ind1,:],istraining_ph:False})
                    ## Metatrain-test error
                    for i in range(ntask_per_batch):
                        accs[ind1[i]] = np.mean(np.argmax(pred[i],1)==np.argmax(Y_metatrain_test[ind1[i],:],1))
                #print(accs)
                print('mean acc = %f\n'%(accs.mean()))
            

            ## Metatest-test 
            if epoch%1==0:
                print('\nMetatest-test:')
                accs = np.nan*np.ones(nbatch_test*ntask_per_batch)
                for batch in range(nbatch_test):
                    ind1 = range(ntask_per_batch*batch,min(ntask_per_batch*(1+batch),ntask_metatest))
                    sess.run([blmt.reset_opt_v,blmt.reset_v])
                    l1 = blmt.update_cls_simple(X_metatest_train[ind1,:],Y_metatest_train[ind1,:],10)
                    pred = sess.run(cls_test, {x_test_ph:X_metatest_test[ind1,:],istraining_ph:False})
                    ## Metatest-test error
                    for i in range(ntask_per_batch):
                        accs[ind1[i]] = np.mean(np.argmax(pred[i],1)==np.argmax(Y_metatest_test[ind1[i],:],1))
                #print(accs)
                print('mean acc = %f\n'%(accs.mean()))

                accs = np.nan*np.ones(nbatch_test*ntask_per_batch)
                for batch in range(nbatch_test):
                    ind1 = range(ntask_per_batch*batch,min(ntask_per_batch*(1+batch),ntask_metatest))
                    sess.run([blmt.reset_opt_v,blmt.reset_v])
                    l1 = blmt.update_cls_simple(X_metatest_train[ind1,:],Y_metatest_train[ind1,:],20)
                    pred = sess.run(cls_test, {x_test_ph:X_metatest_test[ind1,:],istraining_ph:False})
                    ## Metatest-test error
                    for i in range(ntask_per_batch):
                        accs[ind1[i]] = np.mean(np.argmax(pred[i],1)==np.argmax(Y_metatest_test[ind1[i],:],1))
                #print(accs)
                print('mean acc = %f\n'%(accs.mean()))

                accs = np.nan*np.ones(nbatch_test*ntask_per_batch)
                for batch in range(nbatch_test):
                    ind1 = range(ntask_per_batch*batch,min(ntask_per_batch*(1+batch),ntask_metatest))
                    sess.run([blmt.reset_opt_v,blmt.reset_v])
                    l1 = blmt.update_cls_simple(X_metatest_train[ind1,:],Y_metatest_train[ind1,:],40)
                    pred = sess.run(cls_test, {x_test_ph:X_metatest_test[ind1,:],istraining_ph:False})
                    ## Metatest-test error
                    for i in range(ntask_per_batch):
                        accs[ind1[i]] = np.mean(np.argmax(pred[i],1)==np.argmax(Y_metatest_test[ind1[i],:],1))
                #print(accs)
                print('mean acc = %f\n'%(accs.mean()))

                accs = np.nan*np.ones(nbatch_test*ntask_per_batch)
                for batch in range(nbatch_test):
                    ind1 = range(ntask_per_batch*batch,min(ntask_per_batch*(1+batch),ntask_metatest))
                    sess.run([blmt.reset_opt_v,blmt.reset_v])
                    l1 = blmt.update_cls_simple(X_metatest_train[ind1,:],Y_metatest_train[ind1,:],60)
                    pred = sess.run(cls_test, {x_test_ph:X_metatest_test[ind1,:],istraining_ph:False})
                    ## Metatest-test error
                    for i in range(ntask_per_batch):
                        accs[ind1[i]] = np.mean(np.argmax(pred[i],1)==np.argmax(Y_metatest_test[ind1[i],:],1))
                #print(accs)
                print('mean acc = %f\n'%(accs.mean()))
        
            if epoch%10==0:
                #saver_cls0.save(sess,'./meta_multitask_simple_0_omniglot.ckpt')
                saver_cls0.save(sess,'meta_multitask_simple_0.1_omniglot.ckpt')
        
    sess.close()

    return 


##############################################################################################################

if __name__ == '__main__':

    tf.app.run()



