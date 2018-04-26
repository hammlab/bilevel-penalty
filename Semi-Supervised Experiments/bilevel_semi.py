import sys
import tensorflow as tf
import numpy as np
#from cleverhans.model import Model, CallableModelWrapper
from cleverhans.utils_tf import model_loss

## The term (d2g/dvdu)*dg/dv = d/du |dg/dv|^2 is not correct for bilevel. 
## It's true only for minimax where g = -f.
## For reference, -(d2g/dudv)*(d2gdvdv)^-1*dfdv is the formula from implicit f. th.
## To do this, I introduce the additional loss term: -s*(dg/dv)^T*tf.stop_gradient(df/dv)
## so that df_du = dfdu -s*(d2gdudv)*dfdv
def mult_lists(x,y):
    tsum = 0.
    for i in range(len(x)):
        tsum += tf.reduce_sum(tf.multiply(x[i],y[i]))
    return tsum


class bilevel_semi(object):
    def jvp(self,ys, xs, d_xs):
        ## Jacobian-vector product: compute directional derivative dy/dx*d_xs
        v = tf.placeholder_with_default(tf.ones_like(ys), shape=ys.get_shape())
        g = tf.gradients(ys, xs, grad_ys=v)
        return tf.gradients(g, v, grad_ys=d_xs)

    def __init__(self, sess, model, var_model, batch_size, lr_outer, lr_inner, height, width, nch, num_classes, sig=1.):
        self.sess = sess
        self.model = model
        #self.scope_model = scope_model
        self.var_model = var_model#tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope_model)        
        self.batch_size = batch_size
        self.lr_outer = lr_outer
        self.lr_inner = lr_inner
        self.height = height
        self.width = width
        self.nch = nch
        self.num_classes = num_classes
        #self.rho = rho
        self.sig = sig
        
        self.x_train = tf.placeholder(tf.float32,[batch_size,height,width,nch],'x_train')
        #self.y_train = tf.placeholder(tf.float32,[batch_size,num_classes],'y_train')
        self.x_val = tf.placeholder(tf.float32,[batch_size,height,width,nch],'x_val')
        self.y_val = tf.placeholder(tf.float32,[batch_size,num_classes],'y_val')

        self.importance_logit_tf = tf.placeholder(tf.float32,[batch_size,num_classes],'importance_logit_tf')
        self.importance_logit = tf.Variable(np.zeros((batch_size,num_classes),np.float32),name='importance_logit')#,validate_shape=False)
        #self.importance = 0.5*(tf.tanh(self.importance_logitatanh) + 1.) # between 0 and 1
        self.y_train = tf.nn.softmax(self.importance_logit,dim=1)
        #self.y_train = self.importance/tf.tile(tf.expand_dims(tf.reduce_sum(self.importance,1),1),[1,num_classes])

        self.assign_importance_logit = tf.assign(self.importance_logit,self.importance_logit_tf)

        self.output_train = model.get_logits(self.x_train)
        self.output_val = model.get_logits(self.x_val)
        ## The inner loss is defined on training data
        self.loss_inner = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_train,labels=self.y_train))
        #self.loss_inner = tf.reduce_mean(tf.multiply(self.importance,tf.nn.softmax_cross_entropy_with_logits(logits=self.output_train,labels=self.y_train)))
        #tgrad = tf.square(tf.gradients(self.loss_inner,self.var_model)[0]) 
        ##self.loss_gradnorm = 0.5*tf.reduce_sum(tgrad)/np.float(self.batch_size)
        ## The outer loss is defined on validation data
        self.loss_outer1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_val,labels=self.y_val))
        #self.loss_outer2 = self.rho*tf.norm(self.importance,ord=1)
        #self.loss_outer2 = self.rho*tf.reduce_mean(tf.abs(self.importance))
        # dgdv = tf.gradients(self.loss_inner,self.var_model)[0]
        # dfdv_ = tf.stop_gradient(tf.gradients(self.loss_outer1,self.var_model)[0])
        
        dgdv = tf.gradients(self.loss_inner,self.var_model)
        dfdv_ = tf.gradients(self.loss_outer1, self.var_model,stop_gradients=self.var_model)
        
        # Use stop_gradient so that it's a constant w.r.t u when differentiated w.r.t u.
        # self.loss_outer3 = -self.sig*tf.reduce_sum(tf.multiply(dgdv,dfdv_))/batch_size
        self.loss_outer3 = -self.sig*mult_lists(dgdv,dfdv_)/batch_size                                                  
        
        ## EXCLUDE var_model for initialization
        start_vars = set(x.name for x in tf.global_variables())
        self.train_inner = tf.train.AdamOptimizer(self.lr_inner).minimize(self.loss_inner, var_list=self.var_model)
        self.train_outer = tf.train.AdamOptimizer(self.lr_outer).minimize(self.loss_outer1 + self.loss_outer3, var_list=self.importance_logit)
        # new_grads_and_vars = [(g + 
        #self.train_outer_new1 = tf.train.AdamOptimizer(self.lr_outer).apply_gradients(
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        self.init = tf.variables_initializer(var_list=[self.importance_logit]+new_vars)
        #print(self.init)
        self.sess.run(self.init)
        #self.sess.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()
   
    
    def train(self, x_train, x_val, y_val, importance_logit, niter=1):
    
        feed_dict={self.x_train:x_train, self.x_val:x_val, self.y_val:y_val}
        self.sess.run(self.assign_importance_logit,feed_dict={self.importance_logit_tf:importance_logit})
        
        for it in range(niter):
            ## inner step
            self.sess.run(self.train_inner,feed_dict=feed_dict)
            ## outer step
            self.sess.run(self.train_outer,feed_dict=feed_dict)

        l1,l2,l3,timp_logit = self.sess.run([self.loss_inner,self.loss_outer1,self.loss_outer3,self.importance_logit],feed_dict=feed_dict)

        return [l1,l2,l3,timp_logit]
        

