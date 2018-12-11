## Allow dropout and data augmentation
import sys
import tensorflow as tf
import numpy as np
from bilevel_penalty import *


class bilevel_importance(object):

    def __init__(self,sess,x_train_tf,x_train_augm_tf,x_val_tf,x_val_augm_tf,y_train_tf,y_val_tf,
        cls_train,cls_train_augm,cls_val,cls_val_augm,var_cls,
        batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps,istraining_tf):
    
        self.sess = sess
        self.x_train_tf = x_train_tf
        self.x_train_augm_tf = x_train_augm_tf
        self.x_val_tf = x_val_tf
        self.x_val_augm_tf = x_val_augm_tf
        self.y_train_tf = y_train_tf
        self.y_val_tf = y_val_tf
        self.cls_train = cls_train
        self.cls_train_augm = cls_train_augm
        self.cls_val = cls_val
        self.cls_val_augm = cls_val_augm
        self.var_cls = var_cls # v
        self.batch_size = batch_size
        self.istraining_tf = istraining_tf

        self.importance_atanh_tf = tf.placeholder(tf.float32,[self.batch_size],'importance_atanh_tf')
        self.importance_atanh = tf.Variable(np.zeros((self.batch_size),np.float32),name='importance_atanh')
        self.assign_importance_atanh = tf.assign(self.importance_atanh,self.importance_atanh_tf)
        importance = 0.5*(tf.tanh(self.importance_atanh) + 1.) # between 0 and 1

        self.f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_val_augm,labels=self.y_val_tf))
        self.g = tf.reduce_sum(tf.multiply(importance,tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_train_augm,labels=self.y_train_tf)))/tf.reduce_sum(importance)
        
        self.bl = bilevel_penalty(sess,self.f,self.g,self.importance_atanh,var_cls,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)
        
        self.loss_simple_augm = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_train_augm,labels=self.y_train_tf))
        self.optim_simple_augm = tf.train.AdamOptimizer(lr_v).minimize(self.loss_simple_augm,var_list=self.var_cls)
        
        #self.sess.run(self.init)
        #self.sess.run(tf.global_variables_initializer())



    def train(self, x_train, y_train, x_val, y_val, importance_atanh, niter=1):
        self.sess.run(self.assign_importance_atanh,feed_dict={self.importance_atanh_tf:importance_atanh})

        feed_dict={self.x_train_augm_tf:x_train, self.y_train_tf:y_train, self.x_val_augm_tf:x_val, self.y_val_tf:y_val}
        f,gvnorm,lamb_g = self.bl.train(feed_dict,niter)
        timp_atanh = self.sess.run(self.importance_atanh)

        return [f,gvnorm,lamb_g,timp_atanh]


    def train_simple(self, x_train, y_train, nepochs, x_test=None, y_test=None):
        if x_test is None:
            x_test = x_train
        if y_test is None:
            y_test = y_train
            
        batch_size = self.batch_size#x_train_tf.get_shape()[0]
        Ntrain = x_train.shape[0]
        nb_batches = int(np.floor(float(Ntrain)/batch_size))
        for epoch in range(nepochs):
            ind_shuf = np.arange(Ntrain)
            np.random.shuffle(ind_shuf)
            for batch in range(nb_batches):
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntrain))
                ind_tr = ind_shuf[ind_batch]                
                self.sess.run(self.optim_simple_augm,feed_dict={self.x_train_augm_tf:x_train[ind_tr,:],self.y_train_tf:y_train[ind_tr,:]})
            if epoch%10==0:
                l = self.sess.run(self.loss_simple_augm,feed_dict={self.x_train_augm_tf:x_train[ind_tr,:],self.y_train_tf:y_train[ind_tr,:],self.istraining_tf:False})
                print('epoch=%d/%d, loss=%f'%(epoch,nepochs,l))
                print('train:')
                self.eval_simple(x_train, y_train)
                print('test:')
                self.eval_simple(x_test, y_test)
        #return self.eval_simple(x_test, y_test)
        return



    def eval_simple(self, x_test, y_test):
        batch_size = self.batch_size#x_test_tf.get_shape()[0]
        Ntest = x_test.shape[0]
        nb_batches = int(np.floor(float(Ntest)/batch_size))
        acc = 0#np.nan*np.ones(nb_batches)
        preds = np.nan*np.ones(Ntest,np.int32)
        if Ntest%int(batch_size) is not 0:
            print('Warning: Data size is not a multiple of batch_size!!!!')
        for batch in range(nb_batches):
            ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntest))
            pred = self.sess.run(self.cls_val,{self.x_val_tf:x_val[ind_batch,:],self.istraining_tf:False})
            acc += np.sum(np.argmax(pred,1)==np.argmax(y_val[ind_batch,:],1))
            preds[ind_batch] = np.argmax(pred,1)
        acc /= np.float32(nb_batches*batch_size)
        print('mean acc = %f'%(acc))
        return [acc,preds]



