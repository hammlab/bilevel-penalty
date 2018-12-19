import sys
import tensorflow as tf
import numpy as np
from bilevel_penalty import *


class bilevel_semi_gan(object):

    def __init__(self,sess,x_train_ph,x_val_ph,y_train_ph,y_val_ph,
        logits_train,logits_val,logits_gen,
        feat_train,feat_gen,
        var_disc,var_gen,
        batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps,istraining_ph):
    
        self.sess = sess
        self.x_train_ph = x_train_ph
        self.x_val_ph = x_val_ph
        self.y_train_ph = y_train_ph
        self.y_val_ph = y_val_ph
        #self.logits_train = logits_train
        #self.logits_gen = logits_gen
        self.logits_val = logits_val
        #self.var_disc = var_disc # u
        #self.var_gen = var_gen # v
        self.batch_size = batch_size
        self.istraining_ph = istraining_ph        


        ## Define losses
        '''
        loss_sup   = -Ex[log p(y|x,y<=k)]  : function of disc 
        loss_unlab = -Ex[log 1-p(y=k+1|x)] : function of disc
        loss_gen   = -Ez[log p(y=k+1|x)]   : function of disc and gen
        featmatch : function of disc and gen

        '''
        loss_sup = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_val_ph, logits=logits_val))
        loss_unlab = tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(logits_train, axis=1))) \
                        -tf.reduce_mean(tf.reduce_logsumexp(logits_train, axis=1)) 
        loss_gen = tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(logits_gen, axis=1)))
        #loss_unsup = loss_unlab + loss_gen

        m1 = tf.reduce_mean(feat_train, axis=0)
        m2 = tf.reduce_mean(feat_gen, axis=0)
        loss_featmatch = tf.reduce_mean(tf.abs(m1 - m2))

        #self.f = loss_sup + loss_unlab + loss_gen
        #self.g = loss_featmatch #- loss_gen
        self.f = loss_featmatch #- loss_gen
        self.g = loss_sup + loss_unlab + loss_gen


        #self.bl = bilevel_penalty(sess,self.f,self.g,var_disc,var_gen,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)
        self.bl = bilevel_penalty(sess,self.f,self.g,var_gen,var_disc,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)        
        
        self.loss_simple = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_val,labels=y_val_ph))
        self.optim_simple = tf.train.AdamOptimizer(lr_u).minimize(self.loss_simple,var_list=var_disc)
        
        optim_u = tf.train.AdamOptimizer(lr_u)
        optim_v = tf.train.AdamOptimizer(lr_v)

        #self.min_u = optim_u.minimize(self.f,var_list=var_disc)
        #self.min_v = optim_v.minimize(self.g,var_list=var_gen)
        self.min_u = optim_u.minimize(self.f,var_list=var_gen)
        self.min_v = optim_v.minimize(self.g,var_list=var_disc)
        
        #self.sess.run(self.init)
        #self.sess.run(tf.global_variables_initializer())



    def update(self, x_train, x_val, y_val, niter=1):

        feed_dict={self.x_train_ph:x_train,self.x_val_ph:x_val,self.y_val_ph:y_val}
        f,gvnorm,lamb_g = self.bl.update(feed_dict,niter)

        return [f,gvnorm,lamb_g]


    def update_singlelevel(self, x_train, x_val, y_val, lamb=1.):
        feed_dict={self.x_train_ph:x_train,self.x_val_ph:x_val,self.y_val_ph:y_val}
        f,g = self.bl.update_singlelevel(feed_dict,lamb)

        return [f,g]

    
    def update_alternating(self, x_train, x_val, y_val):
        feed_dict={self.x_train_ph:x_train,self.x_val_ph:x_val,self.y_val_ph:y_val}
        f,g = self.bl.update_alternating(feed_dict)

        return [f,g]



    def train_simple(self, x_test, y_test, nepochs):
        batch_size = self.batch_size#x_test_ph.get_shape()[0]
        Ntest = x_test.shape[0]
        nb_batches = int(np.floor(float(Ntest)/batch_size))
        #y_test_logit = np.log(np.clip(y_test,1E-4,1-1E-4))
        for epoch in range(nepochs):
            ind_shuf = np.arange(Ntest)
            np.random.shuffle(ind_shuf)
            for batch in range(nb_batches):
                ind_batch = range(batch_size*batch,min(batch_size*(1+batch),Ntest))
                ind_tr = ind_shuf[ind_batch]                
                self.sess.run(self.optim_simple,feed_dict={self.x_val_ph:x_test[ind_tr],self.y_val_ph:y_test[ind_tr]})
            if epoch%10==0:
                l = self.sess.run(self.loss_simple,feed_dict={self.x_val_ph:x_test[ind_tr],self.y_val_ph:y_test[ind_tr],self.istraining_ph:False})
                print('epoch=%d/%d, loss=%f'%(epoch,nepochs,l))

        return self.eval_simple(x_test, y_test)



    def eval_simple(self, x_test, y_test):
        batch_size = self.batch_size#x_test_ph.get_shape()[0]
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
            tpred = self.sess.run(self.logits_val, {self.x_val_ph:x_test[ind_batch],self.istraining_ph:False})
            acc += np.sum(np.argmax(tpred,1)==np.argmax(y_test[ind_batch],1))
            preds[ind_batch] = tpred#np.argmax(pred,1)
        acc /= np.float32(nb_batches*batch_size)
        print('mean acc = %f'%(acc))
        return [acc,preds]


