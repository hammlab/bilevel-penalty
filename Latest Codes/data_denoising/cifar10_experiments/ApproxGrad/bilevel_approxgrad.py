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


class bilevel_approxgrad(object):

    def __init__(self, sess, f, g, u, v, sig=1E-10):

        self.sess = sess
        self.f = f
        self.g = g
        
        self.lr_u = tf.placeholder(tf.float32)
        self.lr_v = tf.placeholder(tf.float32)
        self.lr_p = tf.placeholder(tf.float32)
        
        self.p = [tf.get_variable('pvec'+str(i),shape=(v[i].shape),initializer=tf.zeros_initializer) for i in range(len(v))]
        self.sig = sig

        ## min_v g
        self.optim_v = tf.train.AdamOptimizer(self.lr_v)
        self.min_v = self.optim_v.minimize(self.g, var_list=v)

        ## solve gvv*p = fv:   
        ## min_p   0.5*p'*gvv*p - p'*fv or \|gvv*p - fv\|^2 or p <- p - c*[gvv*p - fv]??
        self.gvvp = _hessian_vector_product(self.g, v, self.p)
        self.fv = tf.gradients(self.f,v)

        self.h = tf.reduce_sum([tf.reduce_sum(tf.square(self.gvvp[i] + self.sig * self.p[i] - self.fv[i])) for i in range(len(v))])
        self.min_p = tf.train.AdamOptimizer(self.lr_p).minimize(self.h,var_list=self.p)
        
        #Computing the
        ## min_u  [f - gv*p]:  u <- u - [fu - guv*p]
        gv = tf.gradients(self.g,v)
        gvp = tf.reduce_sum([tf.reduce_sum(gv[i]*self.p[i]) for i in range(len(v))])
        #gvp = tf.reduce_sum(gv*self.p)
        f_ = self.f - gvp # df_du = fu - guv*p = fu - guv*inv(gvv)*fv
        self.min_u = tf.train.AdamOptimizer(self.lr_u).minimize(f_,var_list=u)
        

    def update(self, feed_dict, lr_u, lr_v, lr_p, niter1 = 1, niter2 = 1):

        ## min_v g
        feed_dict.update({self.lr_v: lr_v})
        for it in range(niter1):
            self.sess.run(self.min_v,feed_dict)

        ## solve gvv*p = fv    
        feed_dict.update({self.lr_p:lr_p})
        for it in range(niter2):
            #h = self.sess.run(self.h, feed_dict)
            #if it%50 == 0 or it==10 or it == 20:
            #   print(it, h)
            self.sess.run(self.min_p,feed_dict)
        #print(h, "\n")
        
        ## min_u  [f - gv*p]:  u <- u - [fu - guv*p]
        feed_dict.update({self.lr_u:lr_u})
        self.sess.run(self.min_u,feed_dict)

        fval,gval,hval = self.sess.run([self.f,self.g,self.h],feed_dict)

        return [fval,gval,hval]