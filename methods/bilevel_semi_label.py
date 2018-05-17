import sys
import tensorflow as tf
import numpy as np
from bilevel_penalty import *


class bilevel_semi_label(object):

    def __init__(self,sess,x_train_tf,x_test_tf,y_train_logit_tf,y_test_tf,cls_train,cls_test,var_cls,batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps):
    
        self.sess = sess
        self.x_train_tf = x_train_tf
        self.x_test_tf = x_test_tf
        self.y_train_logit_tf = y_train_logit_tf
        self.y_test_tf = y_test_tf
        self.cls_train = cls_train
        self.cls_test = cls_test
        self.var_cls = var_cls # v
        self.batch_size = batch_size

        self.y_train_logit = tf.Variable(np.zeros(y_train_logit_tf.get_shape(),np.float32),name='y_train_logit')
        y_train = tf.nn.softmax(self.y_train_logit,axis=1)
        self.assign_y_train_logit = tf.assign(self.y_train_logit,self.y_train_logit_tf)

        self.f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.cls_test,labels=self.y_test_tf))
        self.g = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.cls_train,labels=y_train))
        
        self.bl = bilevel_penalty(sess,self.f,self.g,self.y_train_logit,var_cls,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)
        
        #self.loss_simple = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_train,labels=y_train))
        self.loss_simple = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_test,labels=y_test_tf))
        self.optim_simple = tf.train.AdamOptimizer(lr_v).minimize(self.loss_simple,var_list=self.var_cls)
        
        #self.sess.run(self.init)
        #self.sess.run(tf.global_variables_initializer())



    def train(self, x_train, y_train_logit, x_test, y_test, niter=1):
        self.sess.run(self.assign_y_train_logit,feed_dict={self.y_train_logit_tf:y_train_logit})

        feed_dict={self.x_train_tf:x_train,self.x_test_tf:x_test,self.y_test_tf:y_test}
        f,gvnorm,lamb_g = self.bl.train(feed_dict,niter)
        ty_train_logit = self.sess.run(self.y_train_logit)

        return [f,gvnorm,lamb_g,ty_train_logit]



    def train_simple(self, x_test, y_test, nepochs):
        batch_size = self.batch_size#x_test_tf.get_shape()[0]
        Ntest = x_test.shape[0]
        nb_batches = int(np.floor(float(Ntest)/batch_size))
        y_test_logit = np.log(y_test)
        for epoch in range(nepochs):
            ind_shuf = np.arange(Ntest)
            np.random.shuffle(ind_shuf)
            for batch in range(nb_batches):
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntest))
                ind_tr = ind_shuf[ind_batch]                
                self.sess.run(self.optim_simple,feed_dict={self.x_test_tf:x_test[ind_tr,:],self.y_test_tf:y_test[ind_tr,:]})
            if epoch%10==0:
                l = self.sess.run(self.loss_simple,feed_dict={self.x_test_tf:x_test[ind_tr,:],self.y_test_tf:y_test[ind_tr,:]})
                print('epoch=%d/%d, loss=%f'%(epoch,nepochs,l))

        return self.eval_simple(x_test, y_test)



    def eval_simple(self, x_test, y_test):
        batch_size = self.batch_size#x_test_tf.get_shape()[0]
        Ntest = x_test.shape[0]
        nclass = y_test.shape[1]
        nb_batches = int(np.floor(float(Ntest)/batch_size))
        acc = 0#np.nan*np.ones(nb_batches)
        #preds = np.nan*np.ones(Ntest,np.int32)
        preds = np.nan*np.ones((Ntest,nclass))
        if Ntest%int(batch_size) is not 0:
            print('Warning: Data size is not a multiple of batch_size!!!!')
        for batch in range(nb_batches):
            ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntest))
            tpred = self.sess.run(self.cls_test, {self.x_test_tf:x_test[ind_batch,:]})
            acc += np.sum(np.argmax(tpred,1)==np.argmax(y_test[ind_batch,:],1))
            preds[ind_batch,:] = tpred#np.argmax(pred,1)
        acc /= np.float32(nb_batches*batch_size)
        print('mean acc = %f'%(acc))
        return [acc,preds]


