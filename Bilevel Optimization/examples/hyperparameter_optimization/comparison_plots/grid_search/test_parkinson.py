# -*- coding: utf-8 -*-
"""
Created on Fri May 25 08:08:37 2018

@author: Akshay
"""
import matplotlib
matplotlib.use('Agg')
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
# Data Formatting 
data = pd.read_csv("parkinson_data.csv",header=0)
mapping = {1 : 1, 0 : -1}
data['23'] = data['23'].map(mapping)
features = list(data.columns[:22])
X_all = data[features].values
print(X_all[0])

Y_all = data['23'].values 
#print(Y_all[:50])               
#Y_all = keras.utils.to_categorical(Y_all, 2)
#print(Y_all[0])             

shuff_idx = np.arange(X_all.shape[0])
np.random.shuffle(shuff_idx)
X_all = X_all[shuff_idx]
Y_all = Y_all[shuff_idx]

np.save("parkinson_shuff_idx.npy", shuff_idx)

val = int(X_all.shape[0]/3)     
 
X_train = X_all[:val]
y_train = Y_all[:val]
print(y_train[:50])
      
X_val = X_all[val: 2*val]
y_val = Y_all[val: 2*val]

X_test = X_all[2*val: ]
y_test = Y_all[2*val: ]

#y_test = y_test.reshape(7532, 1)      
#y_train = y_train.reshape(11314, 1)  
print X_train.shape, X_val.shape, X_test.shape       
print y_train.shape, y_val.shape, y_test.shape  

# range for regularization parameters
alphas = np.linspace(-5, 5, 200)

def cost_func(a):
    clf = linear_model.LogisticRegression(solver='lbfgs', C=np.exp(-a), fit_intercept=False, tol=1e-22, max_iter=500)

    clf.fit(X_train, y_train)
    
    print "alpha", a
    cost_1 = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_test, y_test, 0.)
    print "Test:", cost_1
    
    cost_2 = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_val, y_val, 0.)
    print "Val:", cost_2
    return cost_1

scores = [cost_func(a) for a in alphas]
print scores

# make the plot bigger than default
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.rcParams['font.size'] = 20

# plot the scores
plt.plot(alphas, scores, lw=3, label='cross-validation error')
plt.xlabel(r'$\alpha$', fontsize=40)
#plt.xlim(xmin = -7, xmax = 7)
#plt.xticks(np.arange(-7.0, 7.0, 1))
plt.legend(fontsize=20)
plt.grid()

plt.savefig('graph_parkinson.pdf', bbox_inches='tight')
#plt.show()