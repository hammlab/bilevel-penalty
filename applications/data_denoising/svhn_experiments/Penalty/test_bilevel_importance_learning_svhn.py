############## Imports ##############
import sys
sys.path.append("../")
from svhn_keras_model import svhn_model
import numpy as np
import tensorflow as tf
from bilevel_importance_aug_lag import bilevel_importance
import h5py
from cleverhans.utils_keras import KerasModelWrapper
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
################ Hyperparameters #####################
lr_u = 3
lr_v = 1E-5

nepochs = 101
niter = 20
batch_size = 200

rho_0 = 1E-2
lamb_0 = 1E-2
eps_0 = 1E-2
nu_0 = 0.

c_rho = 1.1
c_lamb = 0.9
c_eps = 0.9

height = 32
width = 32
nch = 1
nclass = 10


def main(argv=None):
    
    tf.set_random_seed(1234)
    sess = tf.Session(config=config)

    for ti in range(1):

        h5f = h5py.File('SVHN_single_grey_without_extra.h5', 'r')
        
        # Load the training, test and validation set
        X_train = h5f['X_train'][:]
        Y_train = h5f['y_train'][:]
        X_test = h5f['X_test'][:]
        Y_test = h5f['y_test'][:]
        X_val = h5f['X_val'][:]
        Y_val = h5f['y_val'][:]
        
        h5f.close()
        
        #Make dataset divisible by batch size
        pts = int(len(X_train)/batch_size)
        X_train = np.array(X_train[:pts*batch_size])
        Y_train = np.array(Y_train[:pts*batch_size])
        
        Ntrain = len(X_train)
        Nval = len(X_val)
        
        corrupt = int(0.5 * len(X_train))
        for i in range(corrupt):
            curr_class = np.argmax(Y_train[i])
            Y_train[i] = np.zeros(10)
            
            #random label
            j = curr_class
            while j == curr_class:
                j = np.random.randint(0, 10)
                                   
            Y_train[i][j] = 1
            
        ## Define model
        x_train_tf = tf.placeholder(tf.float32, shape=(None,height,width,nch))
        y_train_tf = tf.placeholder(tf.float32, shape=(None,nclass))
        
        x_val_tf = tf.placeholder(tf.float32, shape=(None,height,width,nch))
        y_val_tf = tf.placeholder(tf.float32, shape=(None,nclass))
    
        scope_model = 'svhn_classifier'
        with tf.variable_scope(scope_model, reuse=False):    
            model = svhn_model(X_train, nclass)
            
        cls_train = KerasModelWrapper(model).get_logits(x_train_tf)
        cls_test = KerasModelWrapper(model).get_logits(x_val_tf)
    
        var_cls = model.trainable_weights 
    
    
        #########################################################################################################
        ## Bilevel training
        #########################################################################################################
        bl_imp = bilevel_importance(sess, x_train_tf, x_val_tf, y_train_tf, y_val_tf, cls_train, cls_test, var_cls,
            batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,nu_0,c_rho,c_lamb,c_eps)
    
        sess.run(tf.global_variables_initializer())
        
        ## Pre-Train with val data
        bl_imp.train_simple(X_val, Y_val, 200) 
        if False:
            importance_atanh = np.ones((Ntrain))*0.8
            importance_atanh = np.arctanh(2.*importance_atanh-1.)
        else:
            # Predictions based on valididation set pretrained model
            print("Test Accuracy")
            _, __  = bl_imp.eval_simple(X_test, Y_test)
            print("Train Accuracy")
            _,y_train_init = bl_imp.eval_simple(X_train, Y_train)
            
            #Initialize importance 
            importance_atanh = np.ones((Ntrain))*np.arctanh(2.*0.2-1.)
            ind_correct = np.where(np.argmax(Y_train,1)==y_train_init)[0]
            importance_atanh[ind_correct] = np.arctanh(2.*0.8-1.)
            
        if True:
            #Start Bilevel Training
            for epoch in range(nepochs):
                nb_batches = int(float(Ntrain)/batch_size)
                if Ntrain%batch_size!=0:
                    nb_batches += 1
                
                ind_shuf = np.arange(Ntrain)
                np.random.shuffle(ind_shuf)
                for batch in range(nb_batches):
                    ind_batch = range(batch_size*batch,min(batch_size*(1+batch), Ntrain))
                    ind_val = np.random.choice(Nval, size=(batch_size),replace=False)
                    ind_tr = ind_shuf[ind_batch]
                    
                    f, gvnorm, gv_nu, lamb_g, timp_atanh = bl_imp.train(X_train[ind_tr], Y_train[ind_tr], X_val[ind_val], Y_val[ind_val], importance_atanh[ind_tr], niter)
                    importance_atanh[ind_tr] = timp_atanh
    
                ## Renormalize importance_atanh
                if True:
                    importance = 0.5*(np.tanh(importance_atanh)+1.)
                    importance = 0.5*importance/np.mean(importance)
                    importance = np.maximum(.00001,np.minimum(.99999,importance))
                    importance_atanh = np.arctanh(2.*importance-1.)
                
                if epoch%1==0:
                    rho_t,lamb_t,eps_t = sess.run([bl_imp.bl.rho_t,bl_imp.bl.lamb_t,bl_imp.bl.eps_t])
                    print(corrupt,'epoch %d (rho=%f, lamb=%f, eps=%f): h=%f + %f + %f + %f= %f'%
                        (epoch,rho_t,lamb_t,eps_t,f,gvnorm, gv_nu,lamb_g,f+gvnorm+lamb_g+gv_nu))
                    print("Test Accuracy")
                    bl_imp.eval_simple(X_test,Y_test)
                    print("Train Accuracy")
                    bl_imp.eval_simple(X_train,Y_train)
                    print("Val Accuracy")
                    bl_imp.eval_simple(X_val,Y_val)
                    print('mean ai=%f, mean I(ai>0.1)=%f'%(np.mean(importance),len(np.where(importance>0.1)[0])/np.float(X_train.shape[0])))
                    print(corrupt, niter, "\n")

    sess.close()
    return 

##############################################################################################################
if __name__ == '__main__':
    tf.app.run()
