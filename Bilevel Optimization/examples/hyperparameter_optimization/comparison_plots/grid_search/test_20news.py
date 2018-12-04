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
from sklearn.preprocessing import StandardScaler

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
times = 2
pts = 500
total_scores = np.zeros(pts)
for ti in range(times):
    print ti
    shuff_idx = np.arange(X_all_old.shape[0])
    np.random.shuffle(shuff_idx)
    X_all = X_all_old[shuff_idx]
    Y_all = Y_all_old[shuff_idx]
    
    
    val = int(X_all.shape[0]/3)     
     
    X_train = X_all[:2*val]
    y_train = Y_all[:2*val]
          
    #X_val = X_all[val: 2*val]
    #y_val = Y_all[val: 2*val]
    
    X_test = X_all[2*val: ]
    y_test = Y_all[2*val: ]
    
    #y_test = y_test.reshape(7532, 1)      
    #y_train = y_train.reshape(11314, 1)  
    #print X_train.shape, X_val.shape, X_test.shape       
    #print y_train.shape, y_val.shape, y_test.shape  
        
       
    # range for regularization parameters
    alphas = np.linspace(-8, 0, pts)
    def cost_func(a):
        clf = linear_model.LogisticRegression(solver='lbfgs', C=np.exp(-a), fit_intercept=False, tol=1e-22, max_iter=500)
    
        clf.fit(X_train, y_train)
        
        #print "alpha", a
        cost_1 = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_test, y_test, 0)
        print "Test:", cost_1, a
        
        #cost_2 = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_val, y_val, 0)
        #print "Val:", cost_2
        return cost_1
    
    scores = [cost_func(a) for a in alphas]
    total_scores += scores
    print scores

total_scores /= times
# make the plot bigger than default
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.rcParams['font.size'] = 20

# plot the scores
plt.plot(alphas, total_scores, lw=3, label='cross-validation error')
plt.xlabel(r'$\alpha$', fontsize=40)
#plt.xlim(xmin = -7, xmax = 7)
#plt.xticks(np.arange(-7.0, 7.0, 1))
plt.legend(fontsize=20)
plt.grid()
plt.savefig('graph.pdf', bbox_inches='tight')
#plt.show()
