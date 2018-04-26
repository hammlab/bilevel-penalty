import sys
import tensorflow as tf
import numpy as np
#from cleverhans.model import Model, CallableModelWrapper
from cleverhans.utils_tf import model_loss
from cleverhans.utils_keras import KerasModelWrapper
## The term (d2g/dvdu)*dg/dv = d/du |dg/dv|^2 is not correct for bilevel. 
## It's true only for minimax where g = -f.
## For reference, -(d2g/dudv)*(d2gdvdv)^-1*dfdv is the formula from implicit f. th.
## To do this, I introduce the additional loss term: -s*(dg/dv)^T*tf.stop_gradient(df/dv)


class bilevel_cs(object):

    def __init__(self, sess, model, var_model, batch_size, lr_outer, lr_inner, features, num_classes, rho=0., sig=1.):

        self.sess = sess
        self.model = KerasModelWrapper(model)
        #self.scope_model = scope_model
        self.var_model = var_model#tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope_model)        
        self.batch_size = batch_size
        self.lr_outer = lr_outer
        self.lr_inner = lr_inner
        self.features = features
        self.num_classes = num_classes
        self.rho = rho
        self.sig = sig
        
        self.x_train = tf.placeholder(tf.float32,[batch_size, features],'x_train')
        self.y_train = tf.placeholder(tf.float32,[batch_size, num_classes],'y_train')
        self.x_val = tf.placeholder(tf.float32,[batch_size, features],'x_val')
        self.y_val = tf.placeholder(tf.float32,[batch_size, num_classes],'y_val')

        self.importance_atan_tf = tf.placeholder(tf.float32,[batch_size],'importance_atan_tf')
        self.importance_atan = tf.Variable(np.zeros((batch_size),np.float32),name='importance_atan')#,validate_shape=False)
        self.importance = 0.5*(tf.tanh(self.importance_atan) + 1.) # between 0 and 1
        self.assign_importance_atan = tf.assign(self.importance_atan,self.importance_atan_tf)

        self.output_train = self.model.get_logits(self.x_train)
        self.output_val = self.model.get_logits(self.x_val)
        ## The inner loss is defined on training data
        self.loss_inner = tf.reduce_sum(tf.multiply(self.importance,tf.nn.softmax_cross_entropy_with_logits(logits=self.output_train,labels=self.y_train)))/tf.reduce_sum(self.importance)
        #self.loss_inner = tf.reduce_mean(tf.multiply(self.importance,tf.nn.softmax_cross_entropy_with_logits(logits=self.output_train,labels=self.y_train)))
        
        #tgrad = tf.square(tf.gradients(self.loss_inner,self.var_model)[0]) 
        ##self.loss_gradnorm = 0.5*tf.reduce_sum(tgrad)/np.float(self.batch_size)
        ## The outer loss is defined on validation data
        self.loss_outer1 = model_loss(self.y_val, self.output_val)
        #tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_val,labels=self.y_val))
        #self.loss_outer2 = self.rho*tf.norm(self.importance,ord=1)
        self.loss_outer2 = self.rho*tf.reduce_mean(tf.abs(self.importance))
        dgdv = tf.gradients(self.loss_inner, self.var_model)[0]
        dfdv_ = tf.stop_gradient(tf.gradients(self.loss_outer1+self.loss_outer2, self.var_model)[0])
        # Use stop_gradient so that it's a constant w.r.t u when differentiated w.r.t u.
        self.loss_outer3 = -self.sig*tf.reduce_sum(tf.multiply(dgdv,dfdv_))/batch_size
        # self.loss_outer3 = self.loss_gradnorm
        
        ## EXCLUDE var_model for initialization
        start_vars = set(x.name for x in tf.global_variables())
        self.train_inner = tf.train.AdamOptimizer(self.lr_inner).minimize(self.loss_inner,var_list=self.var_model)
        self.train_outer = tf.train.AdamOptimizer(self.lr_outer).minimize(self.loss_outer1+self.loss_outer2+self.loss_outer3,var_list=self.importance_atan)
        # new_grads_and_vars = [(g + 
        #self.train_outer_new1 = tf.train.AdamOptimizer(self.lr_outer).apply_gradients(
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        self.init = tf.variables_initializer(var_list=[self.importance_atan]+new_vars)
        #print(self.init)
        self.sess.run(self.init)
        #self.sess.run(tf.global_variables_initializer())

   
    
    def train(self, x_train, y_train, x_val, y_val, importance_atan, niter=1):
    
        feed_dict={self.x_train:x_train, self.y_train:y_train, self.x_val:x_val, self.y_val:y_val}
        self.sess.run(self.assign_importance_atan,feed_dict={self.importance_atan_tf:importance_atan})
        
        for it in range(niter):
            ## inner step
            self.sess.run(self.train_inner,feed_dict=feed_dict)
            ## outer step
            self.sess.run(self.train_outer,feed_dict=feed_dict)

        l1,l2,l3,l4,timp_atan = self.sess.run([self.loss_inner,self.loss_outer1,self.loss_outer2,self.loss_outer3,self.importance_atan],feed_dict=feed_dict)

        return [l1,l2,l3,l4,timp_atan]
        

