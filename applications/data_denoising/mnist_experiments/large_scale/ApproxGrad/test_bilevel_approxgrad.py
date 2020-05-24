import sys
sys.path.append("../../")
import numpy as np
import tensorflow as tf
from MNIST_keras_model import mnist_model
from bilevel_importance import bilevel_importance
from keras.datasets import mnist
import keras
import time
from cleverhans.utils_keras import KerasModelWrapper
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

lr_u = 1E0
lr_v = 1E-6
lr_p = 1E-5
sig = 1E-3

nepochs = 101
niter1 = 20
niter2 = niter1
batch_size = 200

height = 28
width = 28
nch = 1
nclass = 10

tf.set_random_seed(1234)
sess=tf.Session(config=config)

for ti in range(1):
    ## Read data
    (X_train_all, Y_train_all), (X_test, Y_test) = mnist.load_data()
    
    X_train_all = X_train_all.reshape(len(X_train_all), 28, 28, 1)
    X_test = X_test.reshape(len(X_test), 28, 28, 1)
    
    X_train_all = X_train_all.astype('float32')
    X_test = X_test.astype('float32')
    X_train_all /= 255
    X_test /= 255
    
    Y_train_all = keras.utils.to_categorical(Y_train_all, nclass)
    Y_test = keras.utils.to_categorical(Y_test, nclass)
    
    points = 59000
    val_points = 1000
    
    X_train = X_train_all[:points]
    Y_train = Y_train_all[:points]
    X_val = X_train_all[points:points+val_points] 
    Y_val = Y_train_all[points:points+val_points] 
    
    Ntrain = len(X_train)
    Nval = len(X_val)
    
    corrupt = int(0.25 * len(X_train))
    for i in range(corrupt):
        curr_class = np.argmax(Y_train[i])
        Y_train[i] = np.zeros(10)
        
        #random label
        j = curr_class
        while j == curr_class:
            j = np.random.randint(0, 10)
        
        Y_train[i][j] = 1
        
    ## Define model
    x_train_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_train_tf = tf.placeholder(tf.float32, shape=(batch_size,nclass))
    
    x_val_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_val_tf = tf.placeholder(tf.float32, shape=(batch_size,nclass))

    scope_model = 'mnist_classifier'
    with tf.variable_scope(scope_model, reuse=False):    
        model = mnist_model(X_train, nclass)
        
    cls_train = KerasModelWrapper(model).get_logits(x_train_tf)
    cls_test = KerasModelWrapper(model).get_logits(x_val_tf)

    var_cls = model.trainable_weights      
    
    #########################################################################################################
    ## Bilevel training
    #########################################################################################################
    bl_imp = bilevel_importance(sess, x_train_tf, x_val_tf, y_train_tf, y_val_tf, cls_train, cls_test, var_cls,
        batch_size,sig)

    sess.run(tf.global_variables_initializer())
    
    ## Pre-train with val data
    bl_imp.train_simple(X_val, Y_val, 100) 
    
    if False:
        importance_atanh = np.ones((Ntrain))*0.8
        importance_atanh = np.arctanh(2.*importance_atanh-1.)
    else:
        # Predictions based on valid-pretrained model?
        print("Test Accuracy")
        _, __  = bl_imp.eval_simple(X_test, Y_test)
        print("Train Accuracy")
        _,y_train_init = bl_imp.eval_simple(X_train, Y_train)
        
        importance_atanh = np.ones((Ntrain))*np.arctanh(2.*0.4-1.)
        ind_correct = np.where(np.argmax(Y_train,1)==y_train_init)[0]
        importance_atanh[ind_correct] = np.arctanh(2.*0.6-1.)
        
    if True:
        time_sum = 0
        for epoch in range(1, nepochs):
            start = time.time()
            nb_batches = int(np.floor(float(Ntrain)/batch_size))
            ind_shuf = np.arange(Ntrain)
            np.random.shuffle(ind_shuf)
            
            for batch in range(nb_batches):
                
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch), Ntrain))
                ind_val = np.random.choice(Nval, size=(batch_size),replace=False)
                ind_tr = ind_shuf[ind_batch]
                
                fval, gval, hval, timp_atanh = bl_imp.train(X_train[ind_tr], Y_train[ind_tr], X_val[ind_val], Y_val[ind_val], importance_atanh[ind_tr], lr_u, lr_v, lr_p, niter1, niter2)
                importance_atanh[ind_tr] = timp_atanh
                
            if epoch % 20 == 0 and epoch != 0:
                lr_u *= 0.5
                lr_v *= 0.5

            ## Renormalize importance_atanh
            if True:
                importance = 0.5*(np.tanh(importance_atanh)+1.)
                importance = 0.5*importance/np.mean(importance)
                importance = np.maximum(.00001,np.minimum(.99999,importance))
                importance_atanh = np.arctanh(2.*importance-1.)
            
            if epoch%1==0:
                print("Test Accuracy")
                bl_imp.eval_simple(X_test,Y_test)
                

sess.close()
