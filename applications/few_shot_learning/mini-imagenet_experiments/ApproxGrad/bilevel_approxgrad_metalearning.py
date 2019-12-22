import sys
sys.path.append("../../../../../optimizers/few_shot_learning")
import tensorflow as tf
import numpy as np
from bilevel_approxgrad_multivar import bilevel_approxgrad_multivar

class bilevel_meta(object):

    def __init__(self,sess,x_train_ph,x_test_ph,y_train_ph,y_test_ph,
        cls_train,cls_test,var_filt,var_cls,
        ntask,ntrain_per_task,ntest_per_task,nclass_per_task,
        lr_u,lr_v,lr_p,sig,istraining_ph):

        self.sess = sess
        self.x_train_ph = x_train_ph
        self.x_test_ph = x_test_ph
        self.y_train_ph = y_train_ph
        self.y_test_ph = y_test_ph

        self.cls_train = cls_train
        self.cls_test = cls_test
        
        self.var_filt = var_filt # u
        self.var_cls = var_cls # v

        self.ntask = ntask
        self.ntrain_per_task = ntrain_per_task
        self.ntest_per_task = ntest_per_task
        self.nclass_per_task = nclass_per_task
        
        self.lr_u = lr_u
        self.lr_v = lr_v
        self.lr_p = lr_p
        self.istraining_ph = istraining_ph

        self.f = tf.reduce_mean([tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_test[i],labels=self.y_test_ph[i,:])) for i in range(self.ntask)])
        self.g = tf.reduce_mean([tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.cls_train[i],labels=self.y_train_ph[i,:])) for i in range(self.ntask)])

        self.bl = bilevel_approxgrad_multivar(sess, self.f, self.g, [self.var_filt], self.var_cls, [lr_u], lr_v * np.ones(self.ntask), lr_p * np.ones(self.ntask), sig)

        self.min_u = tf.train.AdamOptimizer(self.lr_u).minimize(self.g, var_list=self.var_filt)

        opt_v = [[] for i in range(ntask)]
        self.reset_opt_v = [[] for i in range(ntask)]
        self.min_v = [[] for i in range(ntask)]
        for i in range(ntask):
            opt_v[i] = tf.train.AdamOptimizer(self.lr_v)
            self.min_v[i] = opt_v[i].minimize(self.g, var_list=self.var_cls[i])
            self.reset_opt_v[i] = tf.variables_initializer(opt_v[i].variables())

    def reset_v_func(self):
        for i in range(self.ntask):
            for var in self.var_cls[i]:
                self.sess.run(var.initializer)
            self.sess.run(self.reset_opt_v[i])

    def update_pretrain_u(self, x_train, y_train, niter = 1):
        feed_dict = {}
        feed_dict.update({self.x_train_u:x_train})
        feed_dict.update({self.y_train_u:y_train})

        for it in range(niter):
            self.sess.run(self.min_pretrain_u,feed_dict)

        feed_dict.update({self.istraining_ph:False})
        l1 = self.sess.run(self.pretrain_u,feed_dict=feed_dict)
        return l1

    def update(self, x_train, y_train, x_test, y_test, niter1=1, niter2 = 1):
        
        feed_dict={self.x_train_ph:x_train, self.y_train_ph:y_train, self.x_test_ph:x_test, self.y_test_ph:y_test}
        fval, gval, hval = self.bl.update(feed_dict, niter1, niter2)
        return [fval, gval, hval]

    def update_simple(self, x_train, y_train, niter=1):
        feed_dict = {}
        feed_dict.update({self.x_train_ph:x_train})
        feed_dict.update({self.y_train_ph:y_train})

        for it in range(niter):
            self.sess.run(self.min_v,feed_dict)
        self.sess.run(self.min_u,feed_dict)

        feed_dict.update({self.istraining_ph:False})
        l1 = self.sess.run(self.g,feed_dict=feed_dict)

        return l1

    def update_cls_simple(self, x_train, y_train, niter=1):
        feed_dict = {}
        feed_dict.update({self.x_train_ph:x_train})
        feed_dict.update({self.y_train_ph:y_train})

        for it in range(niter):
            self.sess.run(self.min_v,feed_dict)

        feed_dict.update({self.istraining_ph:False})
        l1 = self.sess.run(self.g,feed_dict=feed_dict)

        return l1