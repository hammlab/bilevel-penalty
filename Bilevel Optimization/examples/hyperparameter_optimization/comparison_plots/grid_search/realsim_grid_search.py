# -*- coding: utf-8 -*-
"""
Created on Tue May 29 19:05:08 2018

@author: Akshay
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
# load some data
from sklearn import linear_model
import time
from scipy.sparse import csr_matrix

f = open("real-sim//real-sim", "r")
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
X_all_old = csr_matrix((data,(row,col)), shape=(idx, max_col+1))
Y_all_old = np.array(Y_all)
#####
times = 100
hy_times = 10
total_scores = np.zeros(hy_times)
total_time = np.zeros(hy_times)
for ti in range(times):
    print ti
    shuff_idx = np.arange(X_all_old.shape[0])
    np.random.shuffle(shuff_idx)
    X_all = X_all_old[shuff_idx]
    Y_all = Y_all_old[shuff_idx]
    
    
    val = int(X_all.shape[0]/3)     
     
    X_train = X_all[:val]
    y_train = Y_all[:val]
          
    X_val = X_all[val: 2*val]
    y_val = Y_all[val: 2*val]
    
    X_test = X_all[2*val: ]
    y_test = Y_all[2*val: ]
    
    #y_test = y_test.reshape(7532, 1)      
    #y_train = y_train.reshape(11314, 1)  
    #print X_train.shape, X_val.shape, X_test.shape       
    #print y_train.shape, y_val.shape, y_test.shape  
        
       
    # range for regularization parameters
    alphas = np.linspace(-12, 12, hy_times)
    scores = np.zeros(hy_times)
    c_time = np.zeros(hy_times)
    it = 0
    cum_time = 0
    min_cost = 100000
    test_cost = 0
    for a in alphas:
        
        tick = time.time()
        clf = linear_model.LogisticRegression(solver='lbfgs', C=np.exp(-a), fit_intercept=False, tol=1e-22, max_iter=500)
        clf.fit(X_train, y_train)
        st_t = time.time() - tick
        cum_time += st_t
        
        val_cost = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_val, y_val, 0)
        
        if val_cost < min_cost:
            min_cost = val_cost
            test_cost = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_test, y_test, 0)
        
        print val_cost, test_cost, cum_time, a
        scores[it] = test_cost
        c_time[it] = cum_time
        it += 1
    
    total_scores += scores
    total_time += c_time

total_scores /= times
total_time /= times

print("\n\n")
st_loss = ""
for idx in range(hy_times):
    st_loss += str(float(total_scores[idx])) + ", "
print st_loss, "\n\n"
st_time = ""
for idx in range(hy_times):
    st_time += str(float(total_time[idx])) + ", "    
print st_time
print min_cost 
print("GRID SEARCH RealSim")