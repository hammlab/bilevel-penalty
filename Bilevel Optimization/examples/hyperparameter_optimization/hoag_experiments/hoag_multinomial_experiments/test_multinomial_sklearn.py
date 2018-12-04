# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:52:05 2018

@author: Akshay
"""

import numpy as np
import keras
from sklearn import linear_model
from tensorflow.examples.tutorials.mnist import input_data

def resize_xs(xs):
    N,dim = xs.shape
    if dim != 28*28:
        print 'size wrong', dim
        return
    xs_new = np.zeros((N, 14*14),dtype='float32')
    for i in range(N):
        xi = xs[i].reshape(28,28)
        xs_new[i] = xi[::2,::2].reshape(14*14)
    return xs_new

mnist = input_data.read_data_sets("MNIST_data/")
X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

X_train = resize_xs(X_train)
X_test = resize_xs(X_test)

val = 1000
X_val = X_train[:val]
y_val = y_train[:val]
X_train = X_train[val:]
y_train = y_train[val:]


X_train = X_train#[:5000]
y_train = y_train#[:5000]
X_test = X_test#[:5000]
y_test = y_test#[:5000]

'''
clf = linear_model.LogisticRegression(solver='lbfgs', C=50. / X_train.shape[0], multi_class='multinomial', fit_intercept=False, tol=1e-22, max_iter=500)
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)
print clf.coef_.shape
y_test = keras.utils.to_categorical(y_test, 10) 
cost = linear_model.logistic._multinomial_loss(clf.coef_.ravel(), X_test, y_test, 0, np.ones(X_test.shape[0]))
print cost[0]
'''

# Turn up tolerance for faster convergence
clf = LogisticRegression(C=50. / train_samples,
                         multi_class='multinomial',
                         penalty='l1', solver='saga', tol=0.1)
clf.fit(X_train, y_train)
sparsity = np.mean(clf.coef_ == 0) * 100
score = clf.score(X_test, y_test)
# print('Best C % .4f' % clf.C_)
print("Sparsity with L1 penalty: %.2f%%" % sparsity)
print("Test score with L1 penalty: %.4f" % score)