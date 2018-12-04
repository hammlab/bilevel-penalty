# -*- coding: utf-8 -*-
"""
Created on Tue May 29 23:47:46 2018

@author: Akshay
"""

from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, vstack
from logistic import LogisticRegressionCV
from scipy.sparse import csr_matrix
import numpy as np
f = open("../../real-sim//real-sim", "r")
idx = 0
Y_all = []
row = []
col = []
data = []
max_col = 0
for line in f:
    entries = line.split(" ")
    Y_all.append(int(entries[0]))
    for ent in range(1, len(entries) - 1):
        entry = entries[ent]
        row.append(idx)
        item = entry.split(":")
        max_col = max(max_col, int(item[0]))
        col.append(item[0])
        data.append(item[1])
    idx+=1
    #print row
    #print col
    #print data
    #break
print "here"
row = np.array(row).astype(np.int)
col = np.array(col).astype(np.int)
data = np.array(data).astype(np.float)
X_all = csr_matrix((data,(row,col)), shape=(idx, max_col+1))
Y_all = np.array(Y_all)

shuff_idx = np.arange(X_all.shape[0])
np.random.shuffle(shuff_idx)
X_all = X_all[shuff_idx]
Y_all = Y_all[shuff_idx]

#####
times = 100
loss_total = np.zeros(101)
time_total = np.zeros(101)
np.save("loss_hoag.npy", loss_total)
np.save("time_hoag.npy", time_total)

print("done")
for i in range(times):
    print i
    shuff_idx = np.arange(0, X_all.shape[0])
    np.random.shuffle(shuff_idx)
    X_all_new = X_all[shuff_idx]
    Y_all_new = Y_all[shuff_idx]
    
    val = int(X_all_new.shape[0]/3)     
     
    X_train = X_all_new[:val]
    y_train = Y_all_new[:val]
          
    X_val = X_all_new[val: 2*val]
    y_val = Y_all_new[val: 2*val]
    
    X_test = X_all_new[2*val: ]
    y_test = Y_all_new[2*val: ]
    
    #y_test = y_test.reshape(7532, 1)      
    #y_train = y_train.reshape(11314, 1)  
    #print X_train.shape, X_val.shape, X_test.shape       
    #print y_train.shape, y_val.shape, y_test.shape  
    
    clf = LogisticRegressionCV(verbose = -1)
    clf.fit(X_train, y_train, X_val, y_val, X_test, y_test)
    #print clf.alpha_.shape
    #print('Regularization chosen by HOAG: alpha=%s' % (clf.alpha_[0]))


loss_total = np.load("loss_hoag.npy")
time_total = np.load("time_hoag.npy")
st_loss = ""
for idx in range(101):
    st_loss += str(float(loss_total[idx])/times) + ", "
print(st_loss, "\n\n")
st_time = ""
for idx in range(101):
    st_time += str(float(time_total[idx])/times) + ", "
print(st_time)

print("HOAG")
#clf = linear_model.LogisticRegression(solver='lbfgs', C=np.exp(-clf.alpha_[0]), fit_intercept=False, tol=1e-22, max_iter=500)
#clf.fit(X_train, y_train)
#cost = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_test, y_test, 0.)
#print "Final Cost ", cost