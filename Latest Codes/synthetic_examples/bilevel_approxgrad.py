# -*- coding: utf-8 -*-
"""
Created on Wed Sep 04 13:44:14 2019

@author: Akshay
"""
import tensorflow as tf

def flatten(var):
    return tf.concat([tf.reshape(_var, [-1]) for _var in var], axis=0)


def unflatten(var,ref_list):
    var_list = [[] for i in range(len(ref_list))]
    start = 0
    it = 0
    for list_i in ref_list:
        sh = list_i.shape
        #sh_flatten = tf.reshape(list_i, [-1]).shape[0]
        sh_flatten = tf.size(list_i)#tf.reshape(list_i, [-1]).shape[0].value
        sliced = var[start:start+sh_flatten]
        var_list[it] = tf.reshape(sliced, sh)
        start += sh_flatten
        it += 1
    return var_list

    
def _hessian_vector_product(g, v, p_flat):
        
    gv = flatten(tf.gradients(g, v))
    gvp = tf.reduce_sum(gv*p_flat)#tf.math.multiply(gv, p))
    gvvp = flatten(tf.gradients(gvp, v))
    return gvvp


def _jacobian_vector_product(g, u, v, p_flat):
        
    gv = flatten(tf.gradients(g, v))
    gvp = tf.reduce_sum(gv*p_flat)#tf.math.multiply(gv, p))
    guvp = flatten(tf.gradients(gvp, u))
    return guvp



class bilevel_approxgrad(object):

    def __init__(self,sess,f,g,u,v,lr_u,lr_v,lr_p,ulb=None,uub=None,vlb=None,vub=None,sig=1E-6):

        self.sess = sess
        self.f = f
        self.g = g
        self.u = u
        self.v = v
       
        if ulb is not None:
            self.clip_u = [tf.assign(u[i],tf.clip_by_value(u[i],ulb[i],uub[i])) for i in range(len(u))]
        else: self.clip_u = None
        if vlb is not None:
            self.clip_v = [tf.assign(v[i],tf.clip_by_value(v[i],vlb[i],vub[i])) for i in range(len(v))]
        else: self.clip_v = None
       
        self.q = [tf.get_variable('qvec'+str(i),shape=(v[i].shape),initializer=tf.zeros_initializer) for i in range(len(v))]
        self.p_flat = tf.get_variable('pvec',shape=(flatten(self.q).shape),initializer=tf.zeros_initializer)
        self.p = unflatten(self.p_flat,self.v)
        self.sig = sig

        ## min_v g
        self.min_v = tf.train.AdamOptimizer(lr_v).minimize(self.g, var_list=self.v)

        ## solve gvv*p = fv:   
        ## min_p   \|gvv*p - fv\|^2 
        self.gvvp = _hessian_vector_product(self.g, self.v, self.p_flat)
        self.fv = flatten(tf.gradients(self.f,self.v))

        self.h = tf.reduce_sum(tf.square(self.gvvp + self.sig*self.p_flat - self.fv))
            
        self.min_p = tf.train.AdamOptimizer(lr_p).minimize(self.h,var_list=self.p_flat)
        
        #Computing the
        ## min_u  [f - gv*p]:  u <- u - [fu - guv*p]
        self.gv = tf.gradients(self.g, self.v)
        self.gvp = tf.reduce_sum([tf.reduce_sum(self.gv[i]*self.p[i]) for i in range(len(self.v))])
        self.f_ = self.f - self.gvp # df_du = fu - guv*p = fu - guv*inv(gvv)*fv
        self.min_u = tf.train.AdamOptimizer(lr_u).minimize(self.f_,var_list=self.u)
        

    def update(self,feed_dict,niter1=1,niter2=1):

        ## min_v g
        for it in range(niter1):
            self.sess.run(self.min_v,feed_dict)
            if False:#self.clip_v is not None:
                self.sess.run(self.clip_v)

        ## solve gvv*p = fv    
        for it in range(niter2+1):
            self.sess.run(self.min_p,feed_dict)
        
        ## min_u  [f - gv*p]:  u <- u - [fu - guv*p]
        self.sess.run(self.min_u,feed_dict)
        if False:#self.clip_u is not None:
            self.sess.run(self.clip_u)

        fval,gval,hval = self.sess.run([self.f,self.g,self.h],feed_dict)

        return [fval,gval,hval]

