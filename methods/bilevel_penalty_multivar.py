'''

Solve min_u f(u,v) s.t. v = argmin g(u,v) by repeating
single update:
while del_u^2 + del_v^2 > eps_t^2 or maxiter
    del_v = (fv + rho_t gvv*gv + lamb_t*gv)
    v = v - lr_v*delv

    del_u = (fu + rho_t guv*gv + lamb_t*gu)            
    u = u - lr_u*del_u


rho_t *= c_rho
lamb_t *= c_lamb            
    
input-init: f, g, u, v, lr_u, lr_v
input-iter: rho_t, lamb_t, niter
output-iter: del_v, del_u, f, gvnorm, lamb_g

'''

import sys
import tensorflow as tf
import numpy as np


def l2norm_sq(xs):
    return tf.reduce_sum([tf.reduce_sum(tf.square(t)) for t in xs])

def l2norm_sq_diff(xs,ys):
    return tf.reduce_sum([tf.reduce_sum(tf.square(xs[i]-ys[i])) for i in range(len(xs))])

def l2norm_sq_adiff(a,xs,ys):
    return tf.reduce_sum([tf.reduce_sum(tf.square(tf.exp(a[i])*(xs[i]-ys[i]))) for i in range(len(xs))])

def assign_list(xs,ys):
    return [tf.assign(xs[i],ys[i]) for i in range(len(xs))]
    
def addmult_lists(x,a,y):
    tsum = []
    for i in range(len(x)):
        tsum.append(x[i]+a*y[i])
    return tsum


class bilevel_penalty(object):

    def __init__(self,sess,f,g,us,vs,lr_us,lr_vs,rho_0,lamb_0,eps_0,c_rho,c_lamb,c_eps):

        self.sess = sess
        self.f = f
        self.g = g
        
        self.c_rho = c_rho
        self.c_lamb = c_lamb
        self.c_eps = c_eps
        
        self.rho_t = tf.Variable(rho_0,'rho_t')
        self.lamb_t = tf.Variable(lamb_0,'lamb_t')        
        self.eps_t = tf.Variable(eps_0,'eps_t')
        
        v_flat = []
        for i in range(len(vs)):
            v_flat += vs[i]
        dgdv = tf.gradients(g,v_flat)
        self.gvnorm = 0.5*self.rho_t*l2norm_sq(dgdv)
        self.lamb_g = self.lamb_t*g

        h = f + self.gvnorm + self.lamb_g

        optim_u = [[] for i in range(len(us))]
        self.min_u = [[] for i in range(len(us))]
        for i in range(len(us)):
            optim_u[i] = tf.train.AdamOptimizer(lr_us[i])
            self.min_u[i] = optim_u[i].minimize(h,var_list=us[i])

        optim_v = [[] for i in range(len(vs))]
        self.min_v = [[] for i in range(len(vs))]
        for i in range(len(vs)):
            optim_v[i] = tf.train.AdamOptimizer(lr_vs[i])
            self.min_v[i] = optim_v[i].minimize(h,var_list=vs[i])

        self.hunorm = 0
        for i in range(len(us)):
            tgrad_and_var = optim_u[i].compute_gradients(h, var_list=us[i])
            self.hunorm += tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var])

        self.hvnorm = 0
        for i in range(len(vs)):
            tgrad_and_var = optim_v[i].compute_gradients(h, var_list=vs[i])
            self.hvnorm += tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var])
            
        '''
        self.min_v_simple = tf.train.AdamOptimizer(lr_v).minimize(g,var_list=v)
        self.loss_singlelevel = f+lamb_0*g
        self.min_u_singlelevel = tf.train.AdamOptimizer(lr_u).minimize(self.loss_singlelevel,var_list=u)
        self.min_v_singlelevel = tf.train.AdamOptimizer(lr_v).minimize(self.loss_singlelevel,var_list=v)        
        '''
        

    def train(self,feed_dict,niter=1):

        for it in range(niter):
            self.sess.run(self.min_v,feed_dict)
        self.sess.run(self.min_u,feed_dict)
        
        f,gvnorm,lamb_g,hunorm,hvnorm,eps_t = self.sess.run([self.f,self.gvnorm,self.lamb_g,self.hunorm,self.hvnorm,self.eps_t],feed_dict=feed_dict)

        ## Check convergence and update params
        if (hunorm**2 + hvnorm**2 < eps_t**2):
            self.sess.run([tf.assign(self.rho_t,self.c_rho*self.rho_t),tf.assign(self.lamb_t,self.c_lamb*self.lamb_t),tf.assign(self.eps_t,self.c_eps*self.eps_t)])

        return [f,gvnorm,lamb_g]

    '''
    def train_simple(self,feed_dict,niter=1):
        for it in range(niter):
            self.sess.run(self.min_v_simple,feed_dict)
        return self.sess.run(self.loss_simple,feed_dict)

    def train_singlelevel(self,feed_dict,niter=1):
        for it in range(niter):
            self.sess.run(self.min_v_singlelevel,feed_dict)
        self.sess.run(self.min_u_singlelevel,feed_dict)
        return self.sess.run(self.loss_singlelevel,feed_dict)
    '''

    '''
    def getinfo(self):
        
        return self.sess.run([self.rho_t,self.lamb_t,self.eps_t])
    '''





