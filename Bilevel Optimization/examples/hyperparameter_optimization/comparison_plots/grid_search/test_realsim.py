# -*- coding: utf-8 -*-
"""
Created on Fri May 25 08:43:01 2018

@author: Akshay
"""
import matplotlib
matplotlib.use('Agg')
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import numpy as np
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
X_all = csr_matrix((data,(row,col)), shape=(idx, max_col+1))
Y_all = np.array(Y_all)

shuff_idx = np.arange(X_all.shape[0])
np.random.shuffle(shuff_idx)
X_all = X_all[shuff_idx]
Y_all = Y_all[shuff_idx]

np.save("realsim_shuff_idx.npy", shuff_idx)

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
alphas = np.linspace(-10, 2, 100)

def cost_func(a):
    clf = linear_model.LogisticRegression(solver='lbfgs', C=np.exp(-a), fit_intercept=False, tol=1e-22, max_iter=500)

    clf.fit(X_train, y_train)
    
    print "alpha", a
    cost_1 = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_test, y_test, 0.)
    print "Test:", cost_1
    
    #cost_2 = linear_model.logistic._logistic_loss(clf.coef_.ravel(), X_val, y_val, 0.)
    #print "Val:", cost_2
    
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

plt.savefig('graph_realsim.pdf', bbox_inches='tight')
#plt.show()