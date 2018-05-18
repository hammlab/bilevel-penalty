# min_{si} sum_val logreg(y*x^Tw) s.t. 
# w = argmin sum_train logreg(yx^Tw) + \sum_i exp(si)*wi^2

import sys
import tensorflow as tf
import numpy as np

from bilevel_penalty import *


class bilevel_l2reg_simple(object):

    def __init__(self,sess,x_train_tf,x_test_tf,y_train_tf,y_test_tf,cls_train,cls_test,var_cls,batch_size,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps):
        self.sess = sess
        self.x_train_tf = x_train_tf
        self.x_test_tf = x_test_tf
        self.y_train_tf = y_train_tf
        self.y_test_tf = y_test_tf
        self.cls_train = cls_train
        self.cls_test = cls_test
        self.var_cls = var_cls # v
        self.batch_size = batch_size

        self.sig = tf.Variable(0.,'sig')
        self.l2reg = tf.exp(self.sig)*l2norm_sq(self.var_cls)
        self.f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_test,labels=self.y_test_tf))
        self.g = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_train,labels=self.y_train_tf)) + self.l2reg

        self.bl = bilevel_penalty(sess,self.f,self.g,self.sig,var_cls,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)
 
        self.sess.run(tf.global_variables_initializer())


    def train(self,x_train,y_train,x_test,y_test,niter=1):
        feed_dict = {}
        feed_dict.update({self.x_train_tf:x_train})
        feed_dict.update({self.y_train_tf:y_train})
        feed_dict.update({self.x_test_tf:x_test})
        feed_dict.update({self.y_test_tf:y_test})
        
        f,gvnorm,lamb_g = self.bl.train(feed_dict,niter)
        
        l2reg = self.sess.run(self.l2reg,feed_dict=feed_dict)

        return [f,gvnorm,lamb_g,l2reg]


    def train_singlelevel(self,x_train,y_train,x_test,y_test,niter=1):
        feed_dict = {}
        feed_dict.update({self.x_train_tf:x_train})
        feed_dict.update({self.y_train_tf:y_train})
        feed_dict.update({self.x_test_tf:x_test})
        feed_dict.update({self.y_test_tf:y_test})
        
        l = self.bl.train_singlelevel(feed_dict,niter)

        return l


