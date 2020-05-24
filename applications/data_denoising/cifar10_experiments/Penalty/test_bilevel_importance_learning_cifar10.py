import sys
sys.path.append("../")
import numpy as np
import tensorflow as tf
from cifar10_keras_model import CIFAR
from bilevel_importance_aug_lag import bilevel_importance
from keras.datasets import cifar10
import keras
from cleverhans.utils_keras import KerasModelWrapper
from keras.preprocessing.image import ImageDataGenerator
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

lr_u = 3
lr_v = 1E-4

nepochs = 200
niter = 20
batch_size = 200

rho_0 = 1E-2
lamb_0 = 1E-2
eps_0 = 1E-2
nu_0 = 1E-4

c_rho = 1.1
c_lamb = 0.9
c_eps = 0.9

height = 32
width = 32
nch = 3
nclass = 10

def main(argv=None):

    tf.set_random_seed(1234)
    sess = tf.Session(config=config)

    ## Read data
    (X_train_all, Y_train_all), (X_test, Y_test) = cifar10.load_data()
    
    X_train_all = X_train_all.astype('float32')
    X_test = X_test.astype('float32')
    X_train_all /= 255
    X_test /= 255
    
    Y_train_all = keras.utils.to_categorical(Y_train_all, nclass)
    Y_test = keras.utils.to_categorical(Y_test, nclass)
    
    points = 40000
    val_points = 10000
   
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
        
        #shift 1 label
        #j = (curr_class + 1)%10
        
        Y_train[i][j] = 1
        
    
    x_train_tf = tf.placeholder(tf.float32, shape=(batch_size,height,width,nch))
    y_train_tf = tf.placeholder(tf.float32, shape=(batch_size,nclass))
    
    x_val_tf = tf.placeholder(tf.float32, shape=(None,height,width,nch))
    y_val_tf = tf.placeholder(tf.float32, shape=(None,nclass))

    scope_model = 'cifar_classifier'
    with tf.variable_scope(scope_model, reuse=False):    
        model = CIFAR(X_train, nclass)
        
    cls_train = KerasModelWrapper(model).get_logits(x_train_tf)
    cls_test = KerasModelWrapper(model).get_logits(x_val_tf)

    var_cls = model.trainable_weights 

    #########################################################################################################
    ## Bilevel training
    #########################################################################################################
    bl_imp = bilevel_importance(sess, x_train_tf, x_val_tf, y_train_tf, y_val_tf, cls_train, cls_test, var_cls,
        batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,nu_0,c_rho,c_lamb,c_eps)

    sess.run(tf.global_variables_initializer())
    
    # Pretrain lower level
    bl_imp.train_simple(X_val, Y_val, 101)
    
    if False:
        importance_atanh = np.ones((Ntrain))*0.8
        importance_atanh = np.arctanh(2.*importance_atanh-1.)
    else:
        # Predictions based on pretrained model
        print("Test Accuracy")
        _, __  = bl_imp.eval_simple(X_test, Y_test)
        print("Train Accuracy")
        _,y_train_init = bl_imp.eval_simple(X_train, Y_train)
        
        importance_atanh = np.ones((Ntrain))*np.arctanh(2.*0.2-1.)#0.2
        ind_correct = np.where(np.argmax(Y_train,1)==y_train_init)[0]
        importance_atanh[ind_correct] = np.arctanh(2.*0.8-1.)#0.8
        
    if True:
        for epoch in range(nepochs):
            nb_batches = int(np.floor(float(Ntrain)/batch_size))
            ind_shuf = np.arange(Ntrain)
            np.random.shuffle(ind_shuf)
            
            for batch in range(nb_batches):
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch), Ntrain))
                ind_tr = ind_shuf[ind_batch]
                ind_val = np.random.choice(Nval, size=(batch_size),replace=False)
                
                # Using data augmentation
                
                for train_X_batch, train_Y_batch in datagen.flow(X_train[ind_tr], Y_train[ind_tr], batch_size=len(ind_tr), shuffle=False):
                    break
                
                for val_X_batch, val_Y_batch in datagen.flow(X_val[ind_val], Y_val[ind_val], batch_size=len(ind_val), shuffle=False):
                    break
                
                f, gvnorm, gv_nu, lamb_g, timp_atanh = bl_imp.train(train_X_batch, train_Y_batch, val_X_batch, val_Y_batch, importance_atanh[ind_tr], niter)
                #f, gvnorm, gv_nu, lamb_g, timp_atanh = bl_imp.train(X_train[ind_tr], Y_train[ind_tr], X_val[ind_val], Y_val[ind_val], importance_atanh[ind_tr], niter)
                
                importance_atanh[ind_tr] = timp_atanh
                
            ## Renormalize importance_atanh
            if True:
                importance = 0.5*(np.tanh(importance_atanh)+1.)
                importance = 0.5*importance/np.mean(importance)
                importance = np.maximum(.00001,np.minimum(.99999,importance))
                importance_atanh = np.arctanh(2.*importance-1.)
            
            if epoch%1==0:
                rho_t,lamb_t,eps_t = sess.run([bl_imp.bl.rho_t,bl_imp.bl.lamb_t,bl_imp.bl.eps_t])
                print('epoch %d (rho=%f, lamb=%f, eps=%f): h=%f + %f + %f + %f= %f'%
                    (epoch,rho_t,lamb_t,eps_t,f,gvnorm,gv_nu,lamb_g,f+gvnorm+gv_nu+lamb_g))
                print("Test Accuracy")
                bl_imp.eval_simple(X_test,Y_test)
                print("Train Accuracy")
                bl_imp.eval_simple(X_train,Y_train)
                print("Val Accuracy")
                bl_imp.eval_simple(X_val,Y_val)
                print('mean ai=%f, mean I(ai>0.1)=%f'%(np.mean(importance),len(np.where(importance>0.1)[0])/np.float(X_train.shape[0])))
                print(niter, "\n")
            
    sess.close()
    return 

##############################################################################################################
if __name__ == '__main__':
    tf.app.run()
