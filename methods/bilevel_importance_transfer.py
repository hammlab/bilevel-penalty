import sys
import tensorflow as tf
import numpy as np
from bilevel_penalty import *

## min_{al,cls_te} ErrTest(cls_te,filt) s.t. filt,cls_tr = argmin ErrTrain(al,cls_tr,filt)

class bilevel_importance_transfer(object):

    def __init__(self,sess,x_train_tf,x_test_tf,y_train_tf,y_test_tf,
            filt_train,filt_test,cls_train,cls_test,var_filt,var_cls_train,var_cls_test,
            batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps):
    
        self.sess = sess
        self.x_train_tf = x_train_tf
        self.x_test_tf = x_test_tf
        self.y_train_tf = y_train_tf
        self.y_test_tf = y_test_tf
        
        self.filt_train = filt_train
        self.filt_test = filt_test
        self.cls_train = cls_train
        self.cls_test = cls_test
        self.var_filt = var_filt # v1
        self.var_cls_train = var_cls_train # v2
        self.var_cls_test = var_cls_test # u2
        self.batch_size = batch_size
        self.importance_atanh_tf = tf.placeholder(tf.float32,[self.batch_size],'importance_atanh_tf') 
        self.importance_atanh = tf.Variable(np.zeros((self.batch_size),np.float32),name='importance_atanh') ## (u1)
        self.assign_importance_atanh = tf.assign(self.importance_atanh,self.importance_atanh_tf)
        importance = 0.5*(tf.tanh(self.importance_atanh) + 1.) # between 0 and 1

        self.f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_test,labels=self.y_test_tf))
        self.g = tf.reduce_sum(tf.multiply(importance,tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_train,labels=self.y_train_tf)))/tf.reduce_sum(importance)

        u = [self.importance_atanh]+self.var_cls_test
        #u = self.var_cls_test
        v = self.var_filt+self.var_cls_train
        
        self.bl = bilevel_penalty(sess,self.f,self.g,u,v,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)

        self.loss_upper_simple = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_test,labels=self.y_test_tf))
        self.optim_upper_simple = tf.train.AdamOptimizer(lr_u).minimize(self.loss_upper_simple,var_list=self.var_cls_test)
        ## No update of importance_atanh
        
        self.loss_lower_simple = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_train,labels=self.y_train_tf))
        self.optim_lower_simple = tf.train.AdamOptimizer(lr_v).minimize(self.loss_lower_simple,var_list=v)
        
        #self.sess.run(self.init)
        #self.sess.run(tf.global_variables_initializer())



    def train(self, x_train, y_train, x_test, y_test, importance_atanh, niter=1):
        self.sess.run(self.assign_importance_atanh,feed_dict={self.importance_atanh_tf:importance_atanh})

        feed_dict={self.x_train_tf:x_train, self.y_train_tf:y_train, self.x_test_tf:x_test, self.y_test_tf:y_test}
        f,gvnorm,lamb_g = self.bl.train(feed_dict,niter)
        timp_atanh = self.sess.run(self.importance_atanh)

        return [f,gvnorm,lamb_g,timp_atanh]



    def train_lower_simple(self, x_train, y_train, nepochs):
        batch_size = self.batch_size#x_train_tf.get_shape()[0]
        Ntrain = x_train.shape[0]
        nb_batches = int(np.floor(float(Ntrain)/batch_size))
        for epoch in range(nepochs):
            ind_shuf = np.arange(Ntrain)
            np.random.shuffle(ind_shuf)
            for batch in range(nb_batches):
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntrain))
                ind_tr = ind_shuf[ind_batch]                
                self.sess.run(self.optim_lower_simple,feed_dict={self.x_train_tf:x_train[ind_tr,:],self.y_train_tf:y_train[ind_tr,:]})
            if epoch%10==0:
                l = self.sess.run(self.loss_lower_simple,feed_dict={self.x_train_tf:x_train[ind_tr,:],self.y_train_tf:y_train[ind_tr,:]})
                print('epoch=%d/%d, loss_lower=%f'%(epoch,nepochs,l))

        return self.eval_lower_simple(x_train, y_train)


    def train_upper_simple(self, x_test, y_test, nepochs):
        batch_size = self.batch_size#x_test_tf.get_shape()[0]
        Ntest = x_test.shape[0]
        nb_batches = int(np.floor(float(Ntest)/batch_size))
        for epoch in range(nepochs):
            ind_shuf = np.arange(Ntest)
            np.random.shuffle(ind_shuf)
            for batch in range(nb_batches):
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntest))
                ind_tr = ind_shuf[ind_batch]                
                self.sess.run(self.optim_upper_simple,feed_dict={self.x_test_tf:x_test[ind_tr,:],self.y_test_tf:y_test[ind_tr,:]})
            if epoch%10==0:
                l = self.sess.run(self.loss_upper_simple,feed_dict={self.x_test_tf:x_test[ind_tr,:],self.y_test_tf:y_test[ind_tr,:]})
                print('epoch=%d/%d, loss_upper=%f'%(epoch,nepochs,l))

        return self.eval_upper_simple(x_test, y_test)


    def eval_lower_simple(self, x_train, y_train):
        batch_size = self.batch_size#x_train_tf.get_shape()[0]
        Ntrain = x_train.shape[0]
        nb_batches = int(np.floor(float(Ntrain)/batch_size))
        acc = 0#np.nan*np.ones(nb_batches)
        preds = np.nan*np.ones(Ntrain,np.int32)
        if Ntrain%int(batch_size) is not 0:
            print('Warning: Data size is not a multiple of batch_size!!!!')
        for batch in range(nb_batches):
            ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntrain))
            pred = self.sess.run(self.cls_train, {self.x_train_tf:x_train[ind_batch,:]})
            acc += np.sum(np.argmax(pred,1)==np.argmax(y_train[ind_batch,:],1))
            preds[ind_batch] = np.argmax(pred,1)
        acc /= np.float32(nb_batches*batch_size)
        print('mean acc_lower = %f'%(acc))
        return [acc,preds]



    def eval_upper_simple(self, x_test, y_test):
        batch_size = self.batch_size#x_test_tf.get_shape()[0]
        Ntest = x_test.shape[0]
        nb_batches = int(np.floor(float(Ntest)/batch_size))
        acc = 0#np.nan*np.ones(nb_batches)
        preds = np.nan*np.ones(Ntest,np.int32)
        if Ntest%int(batch_size) is not 0:
            print('Warning: Data size is not a multiple of batch_size!!!!')
        for batch in range(nb_batches):
            ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntest))
            pred = self.sess.run(self.cls_test, {self.x_test_tf:x_test[ind_batch,:]})
            acc += np.sum(np.argmax(pred,1)==np.argmax(y_test[ind_batch,:],1))
            preds[ind_batch] = np.argmax(pred,1)
        acc /= np.float32(nb_batches*batch_size)
        print('mean acc_upper = %f'%(acc))
        return [acc,preds]



