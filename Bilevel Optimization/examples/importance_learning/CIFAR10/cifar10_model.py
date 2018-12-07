# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 03:51:24 2018

@author: Akshay
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import tensorflow as tf
from cleverhans.model import Model
from cleverhans.utils import deterministic_dict

"""cleverhans.model.Model implementation of mnist_challenge.model.Model
This re-implementation factors variable creation apart from forward
propagation so it is possible to run forward propagation more than once
in the same model.
"""

class CIFAR(Model):

    def __init__(self, nb_classes=10):
        # NOTE: for compatibility with Madry Lab downloadable checkpoints,
        # we cannot use scopes, give these variables names, etc.
        
        
        self.W_conv1 = self._weight_variable([3, 3, 3, 48])
        self.b_conv1 = self._bias_variable([48])
        self.W_conv2 = self._weight_variable([3, 3, 48, 48])
        self.b_conv2 = self._bias_variable([48])
        
        self.W_conv3 = self._weight_variable([3, 3, 48, 96])
        self.b_conv3 = self._bias_variable([96])
        self.W_conv4 = self._weight_variable([3, 3, 96, 96])
        self.b_conv4 = self._bias_variable([96])
        
        self.W_conv5 = self._weight_variable([3, 3, 96, 192])
        self.b_conv5 = self._bias_variable([192])
        self.W_conv6 = self._weight_variable([3, 3, 192, 192])
        self.b_conv6 = self._bias_variable([192])
        
        self.W_fc1 = self._weight_variable([4 * 4 * 192, 512])
        self.b_fc1 = self._bias_variable([512])
        
        self.W_fc2 = self._weight_variable([512, 256])
        self.b_fc2 = self._bias_variable([256])
        
        self.W_fc3 = self._weight_variable([256, nb_classes])
        self.b_fc3 = self._bias_variable([nb_classes])
        Model.__init__(self, '', nb_classes, {})

    def fprop(self, x):

        output = OrderedDict()
        # first convolutional layer
        h_conv1 = tf.nn.relu(self._conv2d(x, self.W_conv1) + self.b_conv1)
        h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2) + self.b_conv2)
        h_pool1 = self._max_pool_2x2(h_conv2)

        # second convolutional layer
        h_conv3 = tf.nn.relu(self._conv2d(h_pool1, self.W_conv3) + self.b_conv3)
        h_conv4 = tf.nn.relu(self._conv2d(h_conv3, self.W_conv4) + self.b_conv4)
        h_pool2 = self._max_pool_2x2(h_conv4)
        
        # Third convolutional layer
        h_conv5 = tf.nn.relu(self._conv2d(h_pool2, self.W_conv5) + self.b_conv5)
        h_conv6 = tf.nn.relu(self._conv2d(h_conv5, self.W_conv6) + self.b_conv6)
        h_pool3 = self._max_pool_2x2(h_conv6)
        #h_dropout3 = self._dropout(h_pool3, 0.25)#0.25

        # first fully connected layer
        h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 192])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, self.W_fc1) + self.b_fc1)
        #h_dropout4 = self._dropout(h_fc1, 0.5)#0.5

        # second fully connected layer
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
        #h_dropout5 = self._dropout(h_fc2, 0.5)#0.5
        
        # output layer
        logits = tf.matmul(h_fc2, self.W_fc3) + self.b_fc3

        output = deterministic_dict(locals())
        del output["self"]
        output[self.O_PROBS] = tf.nn.softmax(logits=logits)

        return output

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def _dropout(x, prob):
        return tf.nn.dropout(x, prob)

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')