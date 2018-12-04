# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:52:05 2018

@author: Akshay
"""

import numpy as np
from multilogistic import MultiLogisticRegressionCV
from tensorflow.examples.tutorials.mnist import input_data
#from scipy.misc import imresize

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
X_train_all = mnist.train.images
y_train_all = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

X_train_all = resize_xs(X_train_all)
X_test = resize_xs(X_test)


times = 5
loss_total = np.zeros(10)
loss_val_total = np.zeros(10)
time_total = np.zeros(10)
np.save("loss_hoag.npy", loss_total)
np.save("loss_val_hoag.npy", loss_val_total)
np.save("time_hoag.npy", time_total)

print("done")
for i in range(times):
    
    shuff_idx = np.arange(0, len(X_train_all))
    np.random.shuffle(shuff_idx)
    
    X_train = X_train_all[shuff_idx]
    y_train = y_train_all[shuff_idx]
    
    val = 10000
    X_val = X_train[-val:]
    y_val = y_train[-val:]
    X_train = X_train[:-val]
    y_train = y_train[:-val]
    
    #'''
    
    clf = MultiLogisticRegressionCV(verbose = 1)
    clf.fit(X_train, y_train, X_val, y_val, X_test, y_test)
    #print clf.score(X_test[:100], y_test[:100])
    #print clf.alpha_.shape
    #print('Regularization chosen by HOAG: alpha=%s' % (clf.alpha_[0]))
    #cost = linear_model.logistic._multinomial_loss(clf.coef_.ravel(), X_test, y_test, 0.)
    
    
    #clf = linear_model.LogisticRegression(solver='lbfgs', C=50. / X_train.shape[0], multi_class='multinomial', fit_intercept=False, tol=1e-22, max_iter=500)
    #clf.fit(X_train, y_train)
    #cost = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_test, y_test, 0.)
    #print "Final Cost ", cost

loss_total = np.load("loss_hoag.npy")
loss_val_total = np.load("loss_val_hoag.npy")
time_total = np.load("time_hoag.npy")
st_loss = ""
for idx in range(10):
    st_loss += str(float(loss_total[idx])/times) + ", "
print(st_loss, "\n\n")

st_loss = ""
for idx in range(10):
    st_loss += str(float(loss_val_total[idx])/times) + ", "
print(st_loss, "\n\n")

st_time = ""
for idx in range(10):
    st_time += str(float(time_total[idx])/times) + ", "
print(st_time)

print("HOAG")