import sys
import tensorflow as tf
import numpy as np
#from cleverhans.model import Model, CallableModelWrapper
from cleverhans.utils_tf import model_loss

'''
Problem:
The constraint gv=0 isn't helping to reduce g...



'''

## The term (d2g/dvdu)*dg/dv = d/du |dg/dv|^2 is not correct for bilevel. 
## It's true only for minimax where g = -f.
## For reference, -(d2g/dudv)*(d2gdvdv)^-1*dfdv is the formula from implicit f. th.
## To do this, I introduce the additional loss term: -s*(dg/dv)^T*tf.stop_gradient(df/dv)
## so that df_du = dfdu -s*(d2gdudv)*dfdv


def print_shape(x):
    if isinstance(x, (list,)):
        for i in range(len(x)):
            print(x[i].get_shape())
    else:
        print(x.get_shape())

def print_shape2(sess,x):
    t = 0
    if isinstance(x, (list,)):
        for i in range(len(x)):
            s = sess.run(tf.shape(x[i]))
            t += np.prod(np.array(s))
            print(s)
    else:
        s = sess.run(tf.shape(x))
        t += np.prod(np.array(s))
        print(s)
    print('Total=',t)

def norm_sq_sum(xs):
    total = 0.
    for i in range(len(xs)):
        #for x in xs:
        total += tf.reduce_sum(tf.square(xs[i]))
    return total

def norm_sq_sum2(xs):
    return tf.reduce_sum([tf.reduce_sum(tf.square(t)) for t in xs])

def mult_lists(x,y):
    tsum = 0.
    for i in range(len(x)):
        tsum += tf.reduce_sum(tf.multiply(x[i],y[i]))
    return tsum


'''        
grads = tf.gradients(loss, tf.trainable_variables())
grads, _ = tf.clip_by_global_norm(grads, 50) # gradient clipping
grads_and_vars = list(zip(grads, tf.trainable_variables()))
train_op = optimizer.apply_gradients(grads_and_vars)

'''
class bilevel_mt(object):
    def jvp(self,ys, xs, d_xs):
        ## Jacobian-vector product: compute directional derivative dy/dx*d_xs
        v = tf.placeholder_with_default(tf.ones_like(ys), shape=ys.get_shape())
        g = tf.gradients(ys, xs, grad_ys=v)
        return tf.gradients(g, v, grad_ys=d_xs)

    def __init__(self, sess, model, var_model, batch_size, lr_upper, lr_lower, height, width, nch, num_classes, rho, sig, beta):

        self.sess = sess
        self.model = model
        self.var_model = var_model#tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope_model)        
        self.batch_size = batch_size
        self.lr_upper = lr_upper
        self.lr_lower = lr_lower
        self.height = height
        self.width = width
        self.nch = nch
        self.num_classes = num_classes
        self.beta = beta
        #self.rho = rho
        #self.sig = sig
        
        self.x_train = tf.placeholder(tf.float32,[batch_size,height,width,nch],'x_train')
        self.y_train = tf.placeholder(tf.float32,[batch_size,num_classes],'y_train')
        self.x_val = tf.placeholder(tf.float32,[batch_size,height,width,nch],'x_val')
        self.y_val = tf.placeholder(tf.float32,[batch_size,num_classes],'y_val')
        self.rho_t = tf.placeholder(tf.float32,[],'rho_t')
        self.lamb_t = tf.placeholder(tf.float32,[],'lamb_t')        

        #self.lambda_t = [tf.Variable(tf.zeros_like(self.var_model[i])) for i in range(len(self.var_model))]
        #self.lambda_t = []#[] for in len(self.var_model)]
        #for t in self.var_model:
        #    #print tf.Variable(np.zeros(t.get_shape())).get_shape()
        #    self.lambda_t.append(tf.Variable(np.zeros(t.get_shape(),np.float32)))
        #self.lambda_t = [tf.Variable(tf.zeros_like(t)) for t in self.var_model]

        self.importance_atanh_tf = tf.placeholder(tf.float32,[batch_size],'importance_atanh_tf')
        self.importance_atanh = tf.Variable(np.zeros((batch_size),np.float32),name='importance_atanh')#,validate_shape=False)
        self.importance = 0.5*(tf.tanh(self.importance_atanh) + 1.) # between 0 and 1
        self.assign_importance_atanh = tf.assign(self.importance_atanh,self.importance_atanh_tf)

        self.output_train = model.get_logits(self.x_train)
        self.output_val = model.get_logits(self.x_val)
        
        self.g = tf.reduce_sum(tf.multiply(self.importance, tf.nn.softmax_cross_entropy_with_logits(logits=self.output_train,labels=self.y_train)))/tf.sqrt(tf.reduce_sum(tf.square(self.importance)))
        #self.g = tf.reduce_sum(tf.multiply(self.importance, tf.nn.softmax_cross_entropy_with_logits(logits=self.output_train,labels=self.y_train)))/tf.reduce_sum(self.importance)
        
        dgdv = tf.gradients(self.g, self.var_model)
        
        self.f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_val,labels=self.y_val)) + self.beta * tf.reduce_mean(tf.abs(self.importance))
        self.gvnorm = 0.5 * self.rho_t * norm_sq_sum2(dgdv) # 
        self.lamb_g = self.lamb_t*self.g
        #self.loss_upper3=mult_lists(dgdv,self.lambda_t)
        #dgdv = tf.gradients(self.loss_lower,self.var_model)
        #dfdv_ = tf.gradients(self.loss_upper1+self.loss_upper2,self.var_model,stop_gradients=self.var_model)
        #loss_aug_lagr = self.loss_upper1 + self.loss_upper2 + self.loss_upper3
        #self.update_lambda_t = [tf.assign(self.lambda_t[i], self.lambda_t[i] + self.rho_t*dgdv[i]) for i in range(len(self.lambda_t))]

        loss_u = self.f + self.gvnorm + self.lamb_g
        loss_v = self.f + self.gvnorm + self.lamb_g
        
        ## EXCLUDE var_model for initialization
        start_vars = set(x.name for x in tf.global_variables())
        self.train_u = tf.train.AdamOptimizer(self.lr_upper).minimize(loss_u,var_list=self.importance_atanh)
        self.train_v = tf.train.AdamOptimizer(self.lr_lower).minimize(loss_v,var_list=self.var_model)

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        self.init = tf.variables_initializer(var_list=[self.importance_atanh]+new_vars)#+self.lambda_t)
        #print(self.init)
        self.sess.run(self.init)
        #self.sess.run(tf.global_variables_initializer())

        #print_shape(self.var_model)
        #print_shape2(sess,self.var_model)
    '''
        def update_lambda(self, x_train, y_train, x_val, y_val, importance_atanh, rho_t):
            feed_dict={self.x_train:x_train, self.y_train:y_train, self.x_val:x_val, self.y_val:y_val, self.rho_t:rho_t}
            self.sess.run(self.assign_importance_atanh,feed_dict={self.importance_atanh_tf:importance_atanh})
            self.sess.run(self.update_lambda_t,feed_dict)
    '''


    def train(self, x_train, y_train, x_val, y_val, importance_atanh, rho_t, lamb_t, niter=1):
        feed_dict={self.x_train:x_train, self.y_train:y_train, self.x_val:x_val, self.y_val:y_val, self.rho_t:rho_t, self.lamb_t:lamb_t}
        self.sess.run(self.assign_importance_atanh,feed_dict={self.importance_atanh_tf:importance_atanh})
        
        for it in range(niter):
            self.sess.run(self.train_u,feed_dict=feed_dict)
            self.sess.run(self.train_v,feed_dict=feed_dict)

        l1,l2,l3,timp_atanh = self.sess.run([self.f, self.gvnorm, self.lamb_g, self.importance_atanh],feed_dict=feed_dict)

        return [l1,l2,l3,timp_atanh]


