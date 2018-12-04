# -*- coding: utf-8 -*-
"""
Created on Tue May 29 17:33:53 2018

@author: Akshay
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:54:26 2018

@author: Akshay
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
# load some data
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, vstack
import time

data_train = fetch_20newsgroups(subset='train', categories=None, shuffle=False, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=None, shuffle=False, random_state=42)
print('data loaded')

target_names = data_train.target_names
y_train, y_test = data_train.target, data_test.target
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
print("n_samples: %d, n_features: %d" % X_train.shape)
X_test = vectorizer.transform(data_test.data)
print("n_samples: %d, n_features: %d" % X_test.shape)
X_all_old = vstack([X_train, X_test])

# binarize labels
y_train[data_train.target < 10] = -1
y_train[data_train.target >= 10] = 1
y_test[data_test.target < 10] = -1
y_test[data_test.target >= 10] = 1
  
Y_all_old = np.concatenate([y_train, y_test], axis = 0)
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
    
print("GRID SEARCH")