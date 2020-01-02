import tensorflow as tf
import numpy as np

def l2norm_sq(xs):
    return tf.reduce_sum([tf.reduce_sum(tf.square(t)) for t in xs])

class bilevel_penalty(object):

    def __init__(self,sess,f,g,us,vs,lr_us,lr_vs,rho_0,lamb_0,eps_0,nu_0,c_rho,c_lamb,c_eps):

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
        
        self.dgdv = tf.gradients(g,v_flat)
        self.gvnorm = 0.5*self.rho_t*l2norm_sq(self.dgdv)
        self.lamb_g = self.lamb_t*g

        n = len(self.dgdv)
        self.nu_t = [tf.Variable(np.ones(self.dgdv[i].shape,dtype=np.float32)*nu_0) for i in range(n)]
        self.gvnu = tf.reduce_sum([tf.reduce_sum(self.dgdv[i]*self.nu_t[i]) for i in range(n)])
        self.update_nu = [tf.assign(self.nu_t[i],self.nu_t[i]+self.rho_t*self.dgdv[i]) for i in range(n)]        
        
        self.lr_us_t = [tf.Variable(lr_us[i],dtype=np.float32) for i in range(len(lr_us))]
        self.lr_vs_t = [tf.Variable(lr_vs[i],dtype=np.float32) for i in range(len(lr_vs))]
        self.lr_us_update = [tf.assign(self.lr_us_t[i], self.lr_us_t[i] * 0.99) for i in range(len(lr_us))] 
        self.lr_vs_update = [tf.assign(self.lr_vs_t[i], self.lr_vs_t[i] * 0.99) for i in range(len(lr_vs))] 
        
        h_upper = f + self.gvnu + self.gvnorm
        h_lower = f + self.gvnu + self.gvnorm + self.lamb_g

        optim_u = [[] for i in range(len(us))]
        self.min_u = [[] for i in range(len(us))]
        for i in range(len(us)):
            optim_u[i] = tf.train.AdamOptimizer(self.lr_us_t[i])
            self.min_u[i] = optim_u[i].minimize(h_upper,var_list=us[i])

        optim_v = [[] for i in range(len(vs))]
        self.min_v = [[] for i in range(len(vs))]
        for i in range(len(vs)):
            optim_v[i] = tf.train.AdamOptimizer(self.lr_vs_t[i])
            self.min_v[i] = optim_v[i].minimize(h_lower,var_list=vs[i])

        self.hunorm = 0
        for i in range(len(us)):
            tgrad_and_var = optim_u[i].compute_gradients(h_upper, var_list=us[i])
            self.hunorm += tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var])

        self.hvnorm = 0
        for i in range(len(vs)):
            tgrad_and_var = optim_v[i].compute_gradients(h_lower, var_list=vs[i])
            self.hvnorm += tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var])
            
    def update(self,feed_dict,niter=1):

        for it in range(niter):
            self.sess.run(self.min_v, feed_dict)
        self.sess.run(self.min_u,feed_dict)

        f,gvnorm,lamb_g,gvnu,hunorm,hvnorm,eps_t = self.sess.run([self.f,self.gvnorm,self.lamb_g,self.gvnu,self.hunorm,self.hvnorm,self.eps_t],feed_dict)
        
        ## Check convergence and update params
        if (hunorm**2 + hvnorm**2 < eps_t**2):
            self.sess.run([tf.assign(self.rho_t,self.c_rho*self.rho_t),tf.assign(self.lamb_t,self.c_lamb*self.lamb_t),tf.assign(self.eps_t,self.c_eps*self.eps_t)])
            self.sess.run(self.update_nu,feed_dict)
            self.sess.run(self.lr_us_update)
            self.sess.run(self.lr_vs_update)
            
        return [f,gvnorm, gvnu,lamb_g]