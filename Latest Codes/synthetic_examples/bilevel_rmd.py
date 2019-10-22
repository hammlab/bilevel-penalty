## bilevel_rmd.py


import sys
import tensorflow as tf
import numpy as np


def replace_none_with_zero(l):
    return [0 if i==None else i for i in l]


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





class bilevel_rmd(object):


    def __init__(self,sess,f,g,u,v,lr_u,lr_v,ulb=None,uub=None,vlb=None,vub=None,Tmax=50):

        self.sess = sess
        self.f = f
        self.g = g
        self.u = u
        self.v = v
        n = len(v)
        m = len(u)

        if ulb is not None:
            self.clip_u = [tf.assign(u[i],tf.clip_by_value(u[i],ulb[i],uub[i])) for i in range(len(u))]
        else: self.clip_u = None
        if vlb is not None:
            self.clip_v = [tf.assign(v[i],tf.clip_by_value(v[i],vlb[i],vub[i])) for i in range(len(v))]
        else: self.clip_v = None


        #self.t = tf.get_variable('t',shape=(1),dtype=tf.int16,initializer=tf.zeros_initializer)
        self.vs = [[[] for i in range(n)] for t in range(Tmax)]
        for t in range(Tmax):
            self.vs[t] = [tf.get_variable('v'+str(t)+'_'+str(i),shape=(v[i].shape),initializer=tf.zeros_initializer) for i in range(n)]
        ## Copy current v to vs[t]
        self.write_v= [[[] for i in range(n)] for t in range(Tmax)]
        self.read_v = [[[] for i in range(n)] for t in range(Tmax)]
        for t in range(Tmax):            
            self.write_v[t] = [tf.assign(self.vs[t][i],self.v[i]) for i in range(n)]
            self.read_v[t] = [tf.assign(self.v[i],self.vs[t][i]) for i in range(n)]

        ## min_v g
        #self.min_v = tf.train.AdamOptimizer(lr_v).minimize(g,var_list=v)
        #self.min_v = tf.train.GradientDescentOptimizer(lr_v).minimize(g,var_list=v)        
        gv = tf.gradients(g,self.v)
        self.min_v = [tf.assign(self.v[i],self.v[i]-lr_v*gv[i]) for i in range(n)]

        ## RMD 
        ## Phi = v - lr_v*gv
        ## A = I - lr_v*gvv, B = -lr_v*guv

        ## q_T := fv(u,v_T), p_T:=fu(u,v_T)\\
        ## p_{t-1} = p_t + B_t q_t = p - lr_v guv*q
        ## q_{t-1} = A_t q_t = q - lr_v gvv*q
        ## return p

        #self.q = [tf.get_variable('qvec'+str(i),shape=(v[i].shape),initializer=tf.zeros_initializer) for i in range(n)]
        #self.p = [tf.get_variable('pvec'+str(i),shape=(u[i].shape),initializer=tf.zeros_initializer) for i in range(m)]
        self.q_flat = tf.get_variable('qvec',shape=(flatten(self.v).shape),initializer=tf.zeros_initializer)
        self.q = unflatten(self.q_flat,self.v)
        self.p_flat = tf.get_variable('pvec',shape=(flatten(self.u).shape),initializer=tf.zeros_initializer)
        self.p = unflatten(self.p_flat,self.u)
        
        fu = tf.gradients(f,self.u)
        fv = tf.gradients(f,self.v)
        #self.init_q = [tf.assign(self.q[i],fv[i]) for i in range(n)] # q = dfdv
        #self.init_p  = [tf.assign(self.p[i],fu[i]) for i in range(m)] # p = dfdu
        self.init_q = tf.assign(self.q_flat,flatten(fv))
        self.init_p = tf.assign(self.p_flat,flatten(fu))
        #self.init_p = [self.p[i].initializer for i in range(m)] # p = 0
        
        ## p <- p - lr_v*guv*q
        guvq = _jacobian_vector_product(g,u,v,self.q_flat)
        #self.update_p = [tf.assign(self.p[i],self.p[i]-lr_v*guvq[i]) for i in range(m)]
        self.update_p = tf.assign(self.p_flat,self.p_flat-lr_v*guvq)
        ## q <- q - lr_v*gvv*q
        gvvq = _hessian_vector_product(g,v,self.q_flat)
        #self.update_q = [tf.assign(self.q[i],self.q[i]-lr_v*gvvq[i]) for i in range(n)]
        self.update_q = tf.assign(self.q_flat,self.q_flat-lr_v*gvvq)
        
        #self.min_u = tf.train.GradientDescentOptimizer(lr_u).minimize(f,var_list=u)        
        #self.min_u = tf.train.AdamOptimizer(lr_u).minimize(f,var_list=u)
        if True:
            self.min_u = [tf.assign(self.u[i],self.u[i]-lr_u*self.p[i]) for i in range(m)]
        else:
            #opt = tf.train.AdamOptimizer(lr_u)
            opt = tf.train.GradientDescentOptimizer(lr_u)
            grads_and_vars = list(zip(self.p, u))            
            self.min_u = opt.apply_gradients(grads_and_vars)



    def update(self,feed_dict,niter=1):

        ## min_v g
        for it in range(niter):
            self.sess.run(self.min_v,feed_dict)  # v <- v - lr_v*dg/dv
            if self.clip_v is not None:
                self.sess.run(self.clip_v)
            self.sess.run(self.write_v[it],feed_dict)  # vs[t] <- v
            
        ## reverse mode
        #self.sess.run(self.init_psum)
        self.sess.run(self.init_p)
        self.sess.run(self.init_q)
        #for it in range(niter-1,0,-1): 
        #for it in range(niter-1,-1,-1): 
        for it in range(niter-2,-1,-1): 
            self.sess.run(self.read_v[it],feed_dict)  # v <- vs[t]
            #self.sess.run(self.read_v[it-1],feed_dict)  # v <- vs[t-1]
            self.sess.run(self.update_p,feed_dict) # p <- p - lr_v*guv*q
            self.sess.run(self.update_q,feed_dict) # q <- q - lr_v*gvv*q
            #self.sess.run(self.sum_p,feed_dict) # psum <= psum + p
            
        ## min_u  
        self.sess.run(self.read_v[niter-1],feed_dict) # v <- vs[T-1]
        self.sess.run(self.min_u,feed_dict)  # u <- u - lamb*psum
        if self.clip_u is not None:
            self.sess.run(self.clip_u)

        fval,gval = self.sess.run([self.f,self.g],feed_dict)

        return [fval,gval]






