import tensorflow as tf
import numpy as np
from bilevel_penalty_aug_lag import bilevel_penalty
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from cleverhans.utils_keras import KerasModelWrapper

def make_classifier(ins,d=2048,K=2):
    W = tf.get_variable('W',[d,K],initializer=tf.random_normal_initializer(stddev=0.01))
    b = tf.get_variable('b',[K],initializer=tf.constant_initializer(0.0))
    out = tf.nn.bias_add(tf.matmul(ins,W),b)
    return out

class bilevel_poisoning(object):

    def __init__(self, sess, x_dogfish_tf, y_dogfish_tf,
                 x_base_tf, x_base_features_tf, y_base_tf, x_target_tf, y_target_tf, x_poison_tf, y_poison_tf,
                 Npoison, height, width, nch,
                 base_model, beta, mini, maxi,
                 lr_u, lr_v, rho_0, lamb_0, eps_0, nu_0, c_rho, c_lamb, c_eps):
        
        self.sess = sess
        self.x_dogfish_tf = x_dogfish_tf
        self.y_dogfish_tf = y_dogfish_tf
        self.x_base_tf = x_base_tf
        self.x_base_features_tf = x_base_features_tf
        self.y_base_tf = y_base_tf
        self.x_target_tf = x_target_tf
        self.y_target_tf = y_target_tf
        self.y_target_incorrect_tf = tf.placeholder(tf.float32, shape=(None, 2), name='y_target_incorrect_tf')
        self.x_poison_tf = x_poison_tf
        self.y_poison_tf = y_poison_tf
        
        self.min = mini
        self.max = maxi
        
        self.height = height
        self.width = width
        self.nch = nch
        
        x_rep = base_model.output
        x_rep = GlobalAveragePooling2D(name="last")(x_rep)
        model_rep = Model(inputs=base_model.input, outputs=x_rep)
        
        with tf.variable_scope('B', reuse=False):
           self.cls_train_dogfish = make_classifier(self.x_dogfish_tf)
        
        self.var_dogfish = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='B') 
        
        self.prediction_dogfish = tf.nn.softmax(self.cls_train_dogfish)
        
        self.loss_dogfish = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.cls_train_dogfish, labels=self.y_dogfish_tf))
        self.optimizer_min_dogfish = tf.train.RMSPropOptimizer(lr_v)
        self.optim_min_dogfish = self.optimizer_min_dogfish.minimize(self.loss_dogfish, var_list=self.var_dogfish)
        
        self.u = tf.get_variable('u', shape=(Npoison, height, width, 3), constraint=lambda t: tf.clip_by_value(t,self.min,self.max))
        self.assign_u = tf.assign(self.u, self.x_poison_tf)
        self.representation_u = KerasModelWrapper(model_rep).get_layer(self.u, 'last')
        
        with tf.variable_scope('B', reuse=True):
           self.cls_target = make_classifier(self.x_target_tf)
        
        with tf.variable_scope('B', reuse=True):
            self.cls_u = make_classifier(self.representation_u)
        
        self.rep_space_dist = tf.norm(self.x_base_features_tf - self.x_target_tf)
        
        costs = []
        for var in self.var_dogfish:
            costs.append(tf.nn.l2_loss(var))
        self.weight_decay = tf.add_n(costs)
        
        self.term1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_target, labels=self.y_target_incorrect_tf))
        self.term2 = tf.reduce_sum(tf.square(self.representation_u - x_target_tf))
        self.term3 = tf.reduce_sum(tf.square(self.u - x_base_tf))
        
        self.f =  self.term1 + self.term2 + beta * self.term3
        
        self.g = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.concat([self.cls_u, self.cls_train_dogfish], 0), labels=tf.concat([self.y_poison_tf, self.y_dogfish_tf], 0)))
        
        self.bl = bilevel_penalty(sess, self.f, self.g, self.u, self.var_dogfish, lr_u, lr_v, rho_0, lamb_0, eps_0, nu_0, c_rho, c_lamb, c_eps)
        
    def train(self, x_train, y_train, target_instance, target_label, base_instance, poison_instance, poison_correct_label, niter):
        
        self.sess.run(self.assign_u,feed_dict={self.x_poison_tf:poison_instance})
        
        X = np.array(x_train)
        Y = np.array(y_train)
        
        feed_dict={self.x_dogfish_tf:X, self.y_dogfish_tf:Y, self.x_target_tf: target_instance, self.y_target_tf:target_label, self.y_target_incorrect_tf:1-target_label, self.x_base_tf:base_instance, self.y_poison_tf:poison_correct_label}
        
        f, gvnorm, gv_nu, lamb_g = self.bl.update(feed_dict,niter)
        poison_instance = self.sess.run(self.u)
        
        return [f, gvnorm, gv_nu, lamb_g, poison_instance]

    def train_simple(self, x_train_features, y_train, x_test_features, y_test, nepochs):
        
        for epoch in range(nepochs):
            
            feed_dict_min = {self.x_dogfish_tf:x_train_features, self.y_dogfish_tf:y_train}
            for min_i in range(1):
                self.sess.run(self.optim_min_dogfish, feed_dict = feed_dict_min)
            
            if epoch%500==0:
                l = self.sess.run(self.loss_dogfish,feed_dict={self.x_dogfish_tf:x_train_features,self.y_dogfish_tf:y_train})
                print('epoch=%d/%d, loss=%f'%(epoch,nepochs,l))
        
        print("Accuracy simple")
        print "Train Accuracy:", self.eval_accuracy(x_train_features, y_train)
        print "Test Accuracy:", self.eval_accuracy(x_test_features, y_test)
        return self.eval_accuracy(x_test_features, y_test)
        
    def eval_accuracy(self, x_test_features, y_test):
        
        X = np.array(x_test_features)
        Y = np.array(y_test)
    
        feed_dict_test = {self.x_dogfish_tf:X}
        pred_output = self.sess.run(self.prediction_dogfish, feed_dict_test)
        acc = np.sum(np.argmax(pred_output,1)==np.argmax(Y,1))
        
        return acc / float(x_test_features.shape[0])
    
    def find_correct_example(self, X_set, Y_set, currect_index = 0):
        for i in range(currect_index, len(X_set)):
            pred_output = self.sess.run(self.prediction_dogfish, feed_dict = {self.x_dogfish_tf: X_set[i].reshape([1,2048])})
            if np.argmax(pred_output,1) == np.argmax(Y_set[i].reshape([1,2]), 1):
                return i, np.argmax(pred_output,1),np.argmax(Y_set[i].reshape([1,2]), 1)
        
            
    def find_closest_example(self, X_set, Y_set, target_index, correct_label, Npoison):
        
        labels = np.argmax(Y_set, 1)
        incorrect_indices = np.argwhere(labels != correct_label).flatten()
        dist = np.ones(len(X_set))*1E10
        for i in range(len(incorrect_indices)):
            dist[incorrect_indices[i]] = self.sess.run(self.rep_space_dist, feed_dict = {self.x_base_features_tf: X_set[incorrect_indices[i]].reshape([1,2048]), self.x_target_tf:X_set[target_index].reshape([1,2048])})
           
        dist_indices = np.argsort(dist)[:Npoison]
        return dist[dist_indices], dist_indices, np.argmax(Y_set[dist_indices].reshape([Npoison, 2]), 1)          