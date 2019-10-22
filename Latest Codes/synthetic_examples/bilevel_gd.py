## bilevel_gd.py


import sys
import tensorflow as tf
import numpy as np


class bilevel_gd(object):

    def __init__(self,sess,f,g,u,v,lr_u,lr_v,ulb=None,uub=None,vlb=None,vub=None):

        self.sess = sess
        self.f = f
        self.g = g

        if ulb is not None:
            self.clip_u = [tf.assign(u[i],tf.clip_by_value(u[i],ulb[i],uub[i])) for i in range(len(u))]
        else: self.clip_u = None
        if vlb is not None:
            self.clip_v = [tf.assign(v[i],tf.clip_by_value(v[i],vlb[i],vub[i])) for i in range(len(v))]
        else: self.clip_v = None

        ## min_u f
        self.min_u = tf.train.AdamOptimizer(lr_u).minimize(f,var_list=u)

        ## min_v g
        self.min_v = tf.train.AdamOptimizer(lr_v).minimize(g,var_list=v)



    def update(self,feed_dict,niter=1):

        ## min_v g
        for it in range(niter):
            self.sess.run(self.min_v,feed_dict)
            if False:#self.clip_v is not None:
                self.sess.run(self.clip_v)

        ## min_u f
        self.sess.run(self.min_u,feed_dict)
        if False:#self.clip_u is not None:
            self.sess.run(self.clip_u)

        fval,gval = self.sess.run([self.f,self.g],feed_dict)

        return [fval,gval]

