import tensorflow as tf
import numpy as np

def l2norm_sq(xs):
    return tf.reduce_sum([tf.reduce_sum(tf.square(t)) for t in xs])

class bilevel_penalty(object):

    def __init__(self,sess,f,g,u,v,lr_u,lr_v,rho_0,lamb_0,eps_0,nu_0,c_rho,c_lamb,c_eps):

        self.sess = sess
        self.f = f
        self.g = g
        
        self.c_rho = c_rho
        self.c_lamb = c_lamb
        self.c_eps = c_eps
        
        self.rho_t = tf.Variable(rho_0,'rho_t')
        self.lamb_t = tf.Variable(lamb_0,'lamb_t')        
        self.eps_t = tf.Variable(eps_0,'eps_t')
        
        self.lr_u_t = tf.Variable(lr_u,dtype=np.float32)
        self.lr_v_t = tf.Variable(lr_v,dtype=np.float32)
        self.lr_u_update = tf.assign(self.lr_u_t, self.lr_u_t * 0.9)
        self.lr_v_update = tf.assign(self.lr_v_t, self.lr_v_t * 0.9)
        
        self.v = v
        
        self.dgdv = tf.gradients(g,self.v)
        self.gvnorm = 0.5*self.rho_t*l2norm_sq(self.dgdv)
        self.lamb_g = self.lamb_t*g

        n = len(self.dgdv)
        self.nu_t = [tf.Variable(np.ones(self.dgdv[i].shape,dtype=np.float32) * nu_0) for i in range(n)]
        self.gvnu = tf.reduce_sum([tf.reduce_sum(self.dgdv[i]*self.nu_t[i]) for i in range(n)])

        self.update_nu = [tf.assign(self.nu_t[i],self.nu_t[i]+self.rho_t*self.dgdv[i]) for i in range(n)]
        
        h_upper = f + self.gvnu + self.gvnorm
        h_lower = f + self.gvnu + self.gvnorm + self.lamb_g

        self.optim_u = tf.train.AdamOptimizer(self.lr_u_t)
        self.optim_v = tf.train.AdamOptimizer(self.lr_v_t)

        self.min_v = self.optim_v.minimize(h_lower,var_list=self.v)
        self.min_u = self.optim_u.minimize(h_upper,var_list=u)
        
        tgrad_and_var_u = self.optim_u.compute_gradients(h_upper, var_list=u)
        self.hunorm = tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var_u])
        tgrad_and_var_v = self.optim_v.compute_gradients(h_lower, var_list=self.v)
        self.hvnorm = tf.reduce_sum([tf.reduce_sum(tf.square(t[0])) for t in tgrad_and_var_v])

    def update(self,feed_dict,niter=1, update=0):

        for it in range(niter):
            self.sess.run(self.min_v,feed_dict)
        self.sess.run(self.min_u,feed_dict)
        
        f, gvnorm, lamb_g, gvnu, hunorm, hvnorm, eps_t = self.sess.run([self.f,self.gvnorm,self.lamb_g,self.gvnu,self.hunorm,self.hvnorm,self.eps_t],feed_dict)

        ## Check convergence and update params
        if (hunorm**2 + hvnorm**2 < eps_t**2):
            self.sess.run([tf.assign(self.rho_t,self.c_rho*self.rho_t),tf.assign(self.lamb_t,self.c_lamb*self.lamb_t),tf.assign(self.eps_t,self.c_eps*self.eps_t)])
            self.sess.run(self.update_nu,feed_dict)
            self.sess.run(self.lr_u_update)
            self.sess.run(self.lr_v_update)

            
        return [f, gvnorm, gvnu, lamb_g]
    
    def reset_penalty_parameters(self):
        self.sess.run(self.rho_t.initializer)
        self.sess.run(self.lamb_t.initializer)
        self.sess.run(self.eps_t.initializer)
        self.sess.run(tf.variables_initializer(self.nu_t))
        self.sess.run(self.lr_u_t.initializer)
        self.sess.run(self.lr_v_t.initializer)
        
    def reset_lower_level(self):
        self.sess.run(tf.variables_initializer(self.v))
