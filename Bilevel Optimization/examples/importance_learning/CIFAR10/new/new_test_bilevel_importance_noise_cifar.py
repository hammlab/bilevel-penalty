import matplotlib
matplotlib.use('Agg')
import os
import sys
sys.path.append('/home/hammj/Dropbox/Research/AdversarialLearning/codes/lib/cleverhans-master')

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from cifar10_keras_model import CIFAR
from cleverhans.utils_mnist import data_mnist
from bilevel_importance import bilevel_importance
import time
from keras.datasets import cifar10
import keras
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.utils_keras import KerasModelWrapper
#############################################################################################################

lr_u = 2
lr_v = 1E-4

nepochs = 200
niter = 1
batch_size = 200

rho_0 = 1E-2
lamb_0 = 1E-2
eps_0 = 1E1

c_rho = 1.1
c_lamb = 0.99
c_eps = 0.99

def main(argv=None):

    height = 32
    width = 32
    nch = 3
    nclass = 10
    #shape = [FLAGS.batch_size,img_rows,img_cols,channels]
    tf.set_random_seed(1234)
    sess = tf.Session()

    ## Read data
    (X_train_all, Y_train_all), (X_test, Y_test) = cifar10.load_data()
    
    X_train_all = X_train_all.astype('float32')
    X_test = X_test.astype('float32')
    X_train_all /= 255
    X_test /= 255
    
    Y_train_all = keras.utils.to_categorical(Y_train_all, nclass)
    Y_test = keras.utils.to_categorical(Y_test, nclass)
    
    points = 40000
    ## Noisy data
    X_train = X_train_all[:points]
    Y_train = Y_train_all[:points]
    X_val = X_train_all[points:] 
    Y_val = Y_train_all[points:] 
    
    Ntrain = len(X_train)
    Nval = len(X_val)
    
    corrupt = int(0.5 * len(X_train))
    for i in range(corrupt):
        Y_train[i] = np.zeros(10)
        
        #random label
        j = np.random.randint(0, 10)
        #adversarial label
        #j = (a + 1)%10
        Y_train[i][j] = 1
        
    np.save("X_train.npy", X_train)
    np.save("Y_train.npy", Y_train)
    np.save("X_val.npy", X_val)
    np.save("Y_val.npy", Y_val)
    np.save("X_test.npy", X_test)
    np.save("Y_test.npy", Y_test)
        
    #idx = np.arange(len(X_train))
    #np.random.shuffle(idx)
    #X_train = X_train[idx]
    #Y_train = Y_train[idx]
    
    ## Define model
    x_train_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_train_tf = tf.placeholder(tf.float32, shape=(batch_size,nclass))
    
    x_val_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_val_tf = tf.placeholder(tf.float32, shape=(batch_size,nclass))

    scope_model = 'cifar_classifier'
    with tf.variable_scope(scope_model, reuse=False):    
        model = CIFAR(X_train, nclass)
        
    cls_train = KerasModelWrapper(model).get_logits(x_train_tf)
    cls_test = KerasModelWrapper(model).get_logits(x_val_tf)

    #var_cls = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_model) 
    var_cls = model.trainable_weights      
    #saver_model = tf.train.Saver(var_model, max_to_keep = None)

    #########################################################################################################
    ## Bilevel training
    #########################################################################################################
    bl_imp = bilevel_importance(sess, x_train_tf, x_val_tf, y_train_tf, y_val_tf, cls_train, cls_test, var_cls,
        batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)

    sess.run(tf.global_variables_initializer())
    bl_imp.train_simple(X_val, Y_val, 100) ## Train with val data
    if False:
        importance_atanh = np.zeros((Ntrain))    
    else:
        # Predictions based on valid-pretrained model?
        print("Test Accuracy")
        _, __  = bl_imp.eval_simple(X_test, Y_test)
        print("Train Accuracy")
        _,y_train_init = bl_imp.eval_simple(X_train, Y_train)
        importance_atanh = np.ones((Ntrain))*np.arctanh(2.*0.2-1.)
        ind_correct = np.where(np.argmax(Y_train)==y_train_init)[0]
        importance_atanh[ind_correct] = np.arctanh(2.*0.8-1.)
        
    if True:
        for epoch in range(nepochs):
            #tick = time.time()        
            nb_batches = int(np.floor(float(Ntrain)/batch_size))
            ind_shuf = np.arange(Ntrain)
            np.random.shuffle(ind_shuf)
            for batch in range(nb_batches):
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch), Ntrain))
                #if len(ind)<FLAGS.batch_size:
                #    ind.extend([np.random.choice(Ntrain,FLAGS.batch_size-len(ind))])
                ind_val = np.random.choice(Nval, size=(batch_size),replace=False)
                ind_tr = ind_shuf[ind_batch]
                f,gvnorm,lamb_g,timp_atanh = bl_imp.train(X_train[ind_tr,:], Y_train[ind_tr,:], X_val[ind_val,:], Y_val[ind_val,:], importance_atanh[ind_tr], niter)
                importance_atanh[ind_tr] = timp_atanh

            ## Renormalize importance_atanh
            if True:
                importance = 0.5*(np.tanh(importance_atanh)+1.)
                importance = 0.5*importance/np.mean(importance)
                importance = np.maximum(.00001,np.minimum(.99999,importance))
                importance_atanh = np.arctanh(2.*importance-1.)
            
            if epoch%1==0:
                rho_t,lamb_t,eps_t = sess.run([bl_imp.bl.rho_t,bl_imp.bl.lamb_t,bl_imp.bl.eps_t])
                print('epoch %d (rho=%f, lamb=%f, eps=%f): h=%f + %f + %f= %f'%
                    (epoch,rho_t,lamb_t,eps_t,f,gvnorm,lamb_g,f+gvnorm+lamb_g))
                print("Test Accuracy")
                bl_imp.eval_simple(X_test,Y_test)
                print("Train Accuracy")
                bl_imp.eval_simple(X_train,Y_train)
                print("Val Accuracy")
                bl_imp.eval_simple(X_val,Y_val)
                print('mean ai=%f, mean I(ai>0.1)=%f'%(np.mean(importance),len(np.where(importance>0.1)[0])/np.float(X_train.shape[0])))
                print("\n")
            
        #saver_model.save(sess,'./model_bilevel_noise_mnist.ckpt')
        #importance = 0.5*(np.tanh(importance_atanh)+1.)
                np.save('./importance_noise.npy',importance)
        #np.save(os.path.join(result_dir,')),tX)        
    else:
        pass
        #saver_model.restore(sess,'./model_bilevel_noise_mnist.ckpt')
        #importance = np.load('./importance_noise.npy')

    ## See the histogram
    plt.figure()
    plt.hist(importance,100)
    plt.show()
            

    ## Retrain and measure test error
    ## Train with est'd clean+val data
    if True:
        ind = np.where(importance>0.1)[0]
        sess.run(tf.global_variables_initializer())    
        acc_train,_ = bl_imp.train_simple(np.concatenate((X_val, X_train[ind,:]),0), np.concatenate((Y_val, Y_train[ind,:]),0),100)
        acc_test,_ = bl_imp.eval_simple(X_test,Y_test)

    
    sess.close()

    return 


##############################################################################################################

if __name__ == '__main__':

    tf.app.run()
