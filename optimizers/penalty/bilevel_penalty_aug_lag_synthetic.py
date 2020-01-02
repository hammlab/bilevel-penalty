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

    def __init__(self,sess,f,g,u,v,lr_u,lr_v,rho_0,lamb_0,eps_0,c_rho,ulb=None,uub=None,vlb=None,vub=None):

        self.sess = sess
        self.f = f
        self.g = g
        
        self.c_rho = c_rho
        #self.c_lamb = c_lamb
        #self.c_eps = c_eps
        
        self.rho_t = tf.Variable(rho_0,'rho_t')
        self.lamb_t = tf.Variable(lamb_0,'lamb_t')        
        self.eps_t = tf.Variable(eps_0,'eps_t')
        
        self.lr_u_t = tf.Variable(lr_u,'lr_u_t')
        self.lr_v_t = tf.Variable(lr_v,'lr_v_t')

        self.dgdv = tf.gradients(g,v)
        self.gvnorm = 0.5*self.rho_t*l2norm_sq(self.dgdv)
        self.lamb_g = self.lamb_t*g

        n = len(self.dgdv)
        self.nu_t = [tf.Variable(np.zeros(self.dgdv[i].shape,dtype=np.float32)) for i in range(n)]
        self.gvnu = tf.reduce_sum([tf.reduce_sum(self.dgdv[i]*self.nu_t[i]) for i in range(n)])

        self.update_nu = [tf.assign(self.nu_t[i],self.nu_t[i]+self.rho_t*self.dgdv[i]) for i in range(n)]        
        
        if ulb is not None:
            self.clip_u = [tf.assign(u[i],tf.clip_by_value(u[i],ulb[i],uub[i])) for i in range(len(u))]
        else: self.clip_u = None
        if vlb is not None:
            self.clip_v = [tf.assign(v[i],tf.clip_by_value(v[i],vlb[i],vub[i])) for i in range(len(v))]
        else: self.clip_v = None


        h_upper = f + self.gvnu + self.gvnorm #+ self.lamb_g # No lamb_g term
        h_lower = f + self.gvnu + self.gvnorm + self.lamb_g

        optim_u = tf.train.AdamOptimizer(self.lr_u_t)
        optim_v = tf.train.AdamOptimizer(self.lr_v_t)

        self.min_u = optim_u.minimize(h_upper,var_list=u)
        self.min_v = optim_v.minimize(h_lower,var_list=v)
        
        tgrad_and_var = optim_u.compute_gradients(h_upper, var_list=u)
        self.hunorm = tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var])
        tgrad_and_var = optim_v.compute_gradients(h_lower, var_list=v)
        self.hvnorm = tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var])

        #self.del_v,_ = zip(*(optim_v.compute_gradients(loss, var_list=v)
        #self.loss_simple = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=g,labels=self.y_train_tf))

        ## singlelevel: min_u,v [f(u,v) + l*g(u,v)]
        self.lamb_ph = tf.placeholder_with_default(1.,shape=())
        self.loss_singlelevel = f + self.lamb_ph*g
        self.min_u_singlelevel = tf.train.AdamOptimizer(lr_u).minimize(self.loss_singlelevel,var_list=u)
        self.min_v_singlelevel = tf.train.AdamOptimizer(lr_v).minimize(self.loss_singlelevel,var_list=v)

        ## alternating: min_u f(u,v), min_v g(u,v)
        #self.min_u_alternating = tf.train.AdamOptimizer(lr_u).minimize(f,var_list=u)
        #self.min_v_alternating = tf.train.AdamOptimizer(lr_v).minimize(g,var_list=v)        


    def update(self,feed_dict,niter=1):

        for it in range(niter):
            self.sess.run(self.min_v,feed_dict)
            if False:#self.clip_v is not None:
                self.sess.run(self.clip_v)
        self.sess.run(self.min_u,feed_dict)
        if False:#self.clip_u is not None:
            self.sess.run(self.clip_u)
        
        f,gvnorm,lamb_g,gvnu,hunorm,hvnorm,eps_t = self.sess.run([self.f,self.gvnorm,self.lamb_g,self.gvnu,self.hunorm,self.hvnorm,self.eps_t],feed_dict)

        ## Check convergence and update params
        if (hunorm**2 + hvnorm**2 < eps_t**2):
            self.sess.run([tf.assign(self.rho_t,self.c_rho*self.rho_t),tf.assign(self.lamb_t,self.lamb_t/self.c_rho),tf.assign(self.eps_t,self.eps_t/self.c_rho)])
            self.sess.run(self.update_nu,feed_dict)
            # Reduce step size
            if True:
                self.sess.run([tf.assign(self.lr_u_t,self.lr_u_t/self.c_rho),tf.assign(self.lr_v_t,self.lr_v_t/self.c_rho)])

        return [f,gvnorm+gvnu,lamb_g] #[f,gvnorm,gvnu,lamb_g]


    def update_singlelevel(self,feed_dict,lamb):
        feed_dict[self.lamb_ph] = lamb       
        self.sess.run([self.min_u_singlelevel,self.min_v_singlelevel],feed_dict)
        if False:#self.clip_u is not None:
            self.sess.run(self.clip_u)
        if False:#self.clip_v is not None:
            self.sess.run(self.clip_v)

        return self.sess.run([self.f,self.g],feed_dict)



