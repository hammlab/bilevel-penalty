import tensorflow as tf
import numpy as np

def _hessian_vector_product(ys, xs, v):

    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")

    # First backprop
    grads = tf.gradients(ys, xs)

    assert len(grads) == length
    elemwise_products = tf.reduce_sum([
      tf.reduce_sum(tf.multiply(grad_elem, v_elem))
      for grad_elem, v_elem in zip(grads, v)
      if grad_elem is not None
    ])

    # Second backprop
    return tf.gradients(elemwise_products, xs)

class bilevel_approxgrad_multivar(object):

    def __init__(self, sess, f, g, us, vs, lr_us, lr_vs, lr_ps, sig):

        self.sess = sess
        self.f = f
        self.g = g
        
        self.lr_vs_t = [tf.Variable(lr_vs[i],dtype=np.float32) for i in range(len(lr_vs))]
        self.lr_us_t = [tf.Variable(lr_us[i],dtype=np.float32) for i in range(len(lr_us))]
        self.lr_ps_t = [tf.Variable(lr_ps[i],dtype=np.float32) for i in range(len(lr_ps))]
        
        optim_v = [[] for i in range(len(vs))]
        self.min_v = [[] for i in range(len(vs))]
        for i in range(len(vs)):
            optim_v[i] = tf.train.AdamOptimizer(self.lr_vs_t[i])
            self.min_v[i] = optim_v[i].minimize(self.g, var_list=vs[i])
            
        self.ps = [[tf.get_variable('pvec_'+str(i)+'_'+str(j),shape=(vs[i][j].shape), initializer=tf.zeros_initializer) for j in range(len(vs[i]))] for i in range(len(vs))]
        self.sig = sig     
        
        gvvps = [_hessian_vector_product(self.g, vs[i], self.ps[i]) for i in range(len(vs))]
        fvs = [tf.gradients(self.f, vs[i]) for i in range(len(vs))]
        
        self.hs = [tf.reduce_sum([tf.reduce_sum(tf.square(gvvps[i][j] + self.sig * self.ps[i][j] - fvs[i][j])) for j in range(len(vs[i]))]) for i in range(len(vs))]
        self.h = tf.reduce_sum(self.hs)
        
        optim_p = [[] for i in range(len(self.ps))]
        self.min_p = [[] for i in range(len(self.ps))]
        for i in range(len(self.ps)):
            optim_p[i] = tf.train.AdamOptimizer(self.lr_ps_t[i])
            self.min_p[i] = optim_p[i].minimize(self.hs[i], var_list=self.ps[i])
        
        #Computing the
        ## min_u  [f - gv*p]:  u <- u - [fu - guv*p]
        gvs = [tf.gradients(self.g, vs[i]) for i in range(len(vs))]
        gvps = [tf.reduce_sum([tf.reduce_sum(gvs[i][j]*self.ps[i][j]) for j in range(len(vs[i]))]) for i in range(len(vs))]
        #gvp = tf.reduce_sum(gv*self.p)
        f_ = [self.f - gvps[i] for i in range(len(vs))] # df_du = fu - guv*p = fu - guv*inv(gvv)*fv
        
        optim_u = [[] for i in range(len(us))]
        self.min_u = [[] for i in range(len(us))]
        for i in range(len(us)):
            optim_u[i] = tf.train.AdamOptimizer(self.lr_us_t[i])
            self.min_u[i] = optim_u[i].minimize(f_[i], var_list=us[i])
        
            
    def update(self, feed_dict, niter1 = 1, niter2 = 1):

        for it in range(niter1):
            self.sess.run(self.min_v, feed_dict)
            
        for it in range(niter2):            
            self.sess.run(self.min_p, feed_dict)
        
        self.sess.run(self.min_u,feed_dict)

        fval, gval, hval = self.sess.run([self.f, self.g, self.h],feed_dict)
        
        return [fval, gval, hval]