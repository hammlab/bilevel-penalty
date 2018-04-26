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


class singlelevel_mt(object):

    def __init__(self, sess, model, var_model, batch_size, lr_model, lr_importance, height, width, nch, num_classes, rho=0., sig=1.):

        self.sess = sess
        self.model = model
        self.var_model = var_model#tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope_model)        
        self.batch_size = batch_size
        self.lr_model = lr_model
        self.lr_importance = lr_importance
        self.height = height
        self.width = width
        self.nch = nch
        self.num_classes = num_classes
        self.rho = rho
        self.sig = sig
        
        self.x_train = tf.placeholder(tf.float32,[batch_size,height,width,nch],'x_train')
        self.y_train = tf.placeholder(tf.float32,[batch_size,num_classes],'y_train')
        self.x_val = tf.placeholder(tf.float32,[batch_size,height,width,nch],'x_val')
        self.y_val = tf.placeholder(tf.float32,[batch_size,num_classes],'y_val')
        
        self.importance_atan_tf = tf.placeholder(tf.float32,[batch_size], 'importance_atan_tf')
        self.importance_atan = tf.Variable(np.zeros((batch_size), np.float32), name='importance_atan')#,validate_shape=False)
        self.importance = 0.5*(tf.tanh(self.importance_atan) + 1.) # between 0 and 1
        self.assign_importance_atan = tf.assign(self.importance_atan, self.importance_atan_tf)

        self.output_train = model.get_logits(self.x_train)
        self.loss = tf.reduce_sum(tf.multiply(self.importance, tf.losses.softmax_cross_entropy(logits=self.output_train, onehot_labels=self.y_train)))/tf.reduce_sum(self.importance)
        
        self.output_val = model.get_logits(self.x_val)
        self.loss_val = tf.reduce_sum(tf.losses.softmax_cross_entropy(logits=self.output_val, onehot_labels=self.y_val))
        
        ## EXCLUDE var_model for initialization
        start_vars = set(x.name for x in tf.global_variables())
        self.train_model = tf.train.AdamOptimizer(self.lr_model).minimize(self.loss + self.loss_val, var_list = self.var_model)
        self.train_importance = tf.train.AdamOptimizer(self.lr_importance).minimize(self.loss + self.loss_val, var_list = self.importance_atan)

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        self.init = tf.variables_initializer(var_list=[self.importance_atan]+new_vars)

        self.sess.run(self.init)
        #self.sess.run(tf.global_variables_initializer())
   
    
    def train(self, x_train, y_train, x_val, y_val, importance_atan, niter = 1):
    
        feed_dict={self.x_train:x_train, self.y_train:y_train, self.x_val:x_val, self.y_val:y_val}
        self.sess.run(self.assign_importance_atan,feed_dict={self.importance_atan_tf:importance_atan})
        
        for it in range(niter):
            self.sess.run(self.train_model, feed_dict = feed_dict)
            self.sess.run(self.train_importance, feed_dict = feed_dict)

        [ltr, lval, timp_atan] = self.sess.run([self.loss, self.loss_val, self.importance_atan], feed_dict = feed_dict)

        return ltr, lval, timp_atan
        

