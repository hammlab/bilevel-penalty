import tensorflow as tf
import numpy as np
from bilevel_penalty_aug_lag import *

class bilevel_importance(object):

    def __init__(self,sess,x_train_tf,x_test_tf,y_train_tf,y_test_tf,cls_train,cls_test,var_cls,batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,nu_0,c_rho,c_lamb,c_eps):
    
        self.sess = sess
        self.x_train_tf = x_train_tf
        self.x_test_tf = x_test_tf
        self.y_train_tf = y_train_tf
        self.y_test_tf = y_test_tf
        self.cls_train = cls_train
        self.cls_test = cls_test
        self.var_cls = var_cls # v
        self.batch_size = batch_size

        self.importance_atanh_tf = tf.placeholder(tf.float32,[self.batch_size],'importance_atanh_tf')
        self.importance_atanh = tf.Variable(np.zeros((self.batch_size),np.float32),name='importance_atanh')
        self.assign_importance_atanh = tf.assign(self.importance_atanh,self.importance_atanh_tf)
        importance = 0.5*(tf.tanh(self.importance_atanh) + 1.) # between 0 and 1

        self.f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_test,labels=self.y_test_tf))
        self.g = tf.reduce_sum(tf.multiply(importance, tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_train,labels=self.y_train_tf)))/tf.reduce_sum(importance)
        
        self.bl = bilevel_penalty(sess,self.f,self.g,self.importance_atanh,var_cls,lr_u,lr_v,rho_0,lamb_0,eps_0,nu_0,c_rho,c_lamb,c_eps)
        
        self.loss_simple = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_train,labels=self.y_train_tf))
        self.optim_simple = tf.train.AdamOptimizer(lr_v).minimize(self.loss_simple,var_list=self.var_cls)

    def train(self, x_train, y_train, x_test, y_test, importance_atanh, niter=1, update=0):
        gvnorm = 0
        self.sess.run(self.assign_importance_atanh,feed_dict={self.importance_atanh_tf:importance_atanh})

        feed_dict={self.x_train_tf:x_train, self.y_train_tf:y_train, self.x_test_tf:x_test, self.y_test_tf:y_test}
        
        #Penalty Method
        f, gvnorm, gv_nu, lamb_g = self.bl.update(feed_dict,niter, update)
        
        timp_atanh = self.sess.run(self.importance_atanh)

        return [f, gvnorm, gv_nu, lamb_g, timp_atanh]

    def train_simple(self, x_train, y_train, nepochs):
        batch_size = self.batch_size
        Ntrain = x_train.shape[0]
        nb_batches = int(np.floor(float(Ntrain)/batch_size))
        for epoch in range(nepochs):
            ind_shuf = np.arange(Ntrain)
            np.random.shuffle(ind_shuf)
            for batch in range(nb_batches):
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntrain))
                ind_tr = ind_shuf[ind_batch]                
                self.sess.run(self.optim_simple,feed_dict={self.x_train_tf:x_train[ind_tr,:],self.y_train_tf:y_train[ind_tr,:]})
            if epoch%50==0:
                l = self.sess.run(self.loss_simple,feed_dict={self.x_train_tf:x_train[ind_tr,:],self.y_train_tf:y_train[ind_tr,:]})
                print('epoch=%d/%d, loss=%f'%(epoch,nepochs,l))

        return self.eval_simple(x_train, y_train)

    def eval_simple(self, x_test, y_test):
        batch_size = self.batch_size
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
        print('mean acc = %f'%(acc))
        return [acc,preds]