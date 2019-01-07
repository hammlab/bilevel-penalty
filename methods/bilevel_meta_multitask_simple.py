import sys
import tensorflow as tf
import numpy as np
#from cleverhans.model import Model, CallableModelWrapper
#from cleverhans.utils_tf import model_loss

#from bilevel_penalty import *
from bilevel_penalty_multivar import *

'''
u is the common network parameter
v is the individual parameter
min_u 1/N sum_i ValErr(vi(u))], s.t. vi(u) = argmin_vi [TrainErr(vi) + 0.5*gamma*\|(vi-u)\|^2]
'''

def l2norm_sq(xs):
    return tf.reduce_sum([tf.reduce_sum(tf.square(t)) for t in xs])

def l2norm_sq_diff(xs,ys):
    return tf.reduce_sum([tf.reduce_sum(tf.square(xs[i]-ys[i])) for i in range(len(xs))])

def assign_list(xs,ys):
    return [tf.assign(xs[i],ys[i]) for i in range(len(xs))]


class bilevel_meta(object):

    def __init__(self,sess,x_train_ph,x_test_ph,y_train_ph,y_test_ph,
        cls_train,cls_test,var_cls0,var_cls,
        ntask,ntrain_per_task,ntest_per_task,nclass_per_task,
        gamma,lr_u,lr_v,
        rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps,istraining_ph):


        self.sess = sess
        self.x_train_ph = x_train_ph
        self.x_test_ph = x_test_ph
        self.y_train_ph = y_train_ph
        self.y_test_ph = y_test_ph
        #self.filt_train = filt_train
        #self.filt_test = filt_test
        self.cls_train = cls_train
        self.cls_test = cls_test
        self.var_cls0 = var_cls0 # u
        self.var_cls = var_cls # v
        self.ntask = ntask
        self.ntrain_per_task = ntrain_per_task
        self.ntest_per_task = ntest_per_task
        self.lr_u = lr_u
        #self.lr_a = lr_a
        self.lr_v = lr_v
        self.nclass_per_task = nclass_per_task
        self.gamma = gamma
        self.istraining_ph = istraining_ph

        #self.rho_t = tf.placeholder(tf.float32,[],'rho_t')
        #self.lamb_t = tf.placeholder(tf.float32,[],'lamb_t')        

        #self.a = [tf.Variable(tf.ones_like(v)) for v in self.var_cls0]
        #self.expamin = tf.reduce_min([tf.reduce_min(tf.exp(ai)) for ai in self.a])
        #self.expamean = tf.reduce_mean([tf.reduce_mean(tf.exp(ai)) for ai in self.a])
        #self.expamax = tf.reduce_max([tf.reduce_max(tf.exp(ai)) for ai in self.a])
        self.diff = tf.reduce_mean([0.5*self.gamma*l2norm_sq_diff(self.var_cls[i],self.var_cls0) for i in range(self.ntask)])


        self.f = tf.reduce_mean([tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_test[i],labels=self.y_test_ph[i,:])) for i in range(self.ntask)])
        self.g = tf.reduce_mean([tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_train[i],labels=self.y_train_ph[i,:])) for i in range(self.ntask)]) + self.diff


        #self.var_cls_flat = [item for sublist in var_cls for item in sublist]
        #print(var_cls_flat)

        #self.bl = bilevel_penalty(sess,self.f,self.g,self.var_cls0,self.var_cls_flat,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)

        self.bl = bilevel_penalty(sess,self.f,self.g,[self.var_cls0],self.var_cls,[lr_u],lr_v*np.ones(self.ntask),rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps)

        #dgdv = [tf.gradients(self.g,self.var_cls[i]) for i in range(self.ntask)]
        #self.gvnorm = 0.5*self.rho_t*tf.reduce_mean([l2norm_sq(dgdv[i]) for i in range(self.ntask)])
        #self.lamb_g = self.lamb_t*self.g

        
        #loss = self.f + self.gvnorm + self.lamb_g
        self.reset_v = [assign_list(self.var_cls[i],self.var_cls0) for i in range(self.ntask)]

        opt_v = [[] for i in range(ntask)]
        self.reset_opt_v = [[] for i in range(ntask)]
        self.min_v = [[] for i in range(ntask)]
        for i in range(ntask):
            opt_v[i] = tf.train.AdamOptimizer(self.lr_v)
            self.min_v[i] = opt_v[i].minimize(self.g,var_list=self.var_cls[i])
            self.reset_opt_v[i] = tf.variables_initializer(opt_v[i].variables())

        '''
        ## EXCLUDE var_filt for initialization
        start_vars = set(x.name for x in tf.global_variables())
        self.min_u = tf.train.AdamOptimizer(self.lr_u).minimize(loss,var_list=self.var_cls0)
        #self.min_a = tf.train.AdamOptimizer(self.lr_a).minimize(loss,var_list=self.a)
        self.min_v = tf.train.AdamOptimizer(self.lr_v).minimize(loss,var_list=self.var_cls)
        self.min_v_simple = tf.train.AdamOptimizer(self.lr_v).minimize(self.g,var_list=self.var_cls)
        #self.min_u = tf.train.GradientDescentOptimizer(self.lr_u).minimize(loss,var_list=self.var_cls0)
        #self.min_a = tf.train.GradientDescentOptimizer(self.lr_a).minimize(loss,var_list=self.a)
        #self.min_v = tf.train.GradientDescentOptimizer(self.lr_v).minimize(loss,var_list=self.var_cls)
        #self.min_v_simple = tf.train.GradientDescentOptimizer(self.lr_v).minimize(self.g,var_list=self.var_cls)
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        
        #self.init = tf.variables_initializer(var_list=[item for sublist in self.var_cls for item in sublist]+new_vars)
        self.init = tf.variables_initializer(var_list=self.var_cls0+self.a+new_vars)
        #print(self.init)
        self.sess.run(self.init)
        #self.sess.run(tf.global_variables_initializer())
        '''

    '''
    def init_var_cls(self):
        self.sess.run(self.init)
    '''


    def update(self, x_train, y_train, x_test, y_test, niter=1):
        feed_dict={self.x_train_ph:x_train, self.y_train_ph:y_train, self.x_test_ph:x_test, self.y_test_ph:y_test}
        f,gvnorm,lamb_g = self.bl.update(feed_dict,niter)

        return [f,gvnorm,lamb_g]


    def update_cls_simple(self, x_train, y_train, niter=1):
        feed_dict = {}#self.rho_t:rho_t,self.lamb_t:lamb_t}
        feed_dict.update({self.x_train_ph:x_train})
        feed_dict.update({self.y_train_ph:y_train})
        #self.sess.run([self.update_x_train,self.update_y_train],feed_dict)

        for it in range(niter):
            self.sess.run(self.min_v,feed_dict)

        feed_dict.update({self.istraining_ph:False})
        l1 = self.sess.run(self.g,feed_dict=feed_dict)

        return l1



