import numpy as np
# load some data
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix, vstack
from logistic import LogisticRegressionCV
categories = None
#remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=False, random_state=42)#, remove=remove)
data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=False, random_state=42)#, remove=remove)
print('data loaded')

target_names = data_train.target_names
y_train, y_test = data_train.target, data_test.target
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
#####
#X_train = X_train.todense()
print("n_samples: %d, n_features: %d" % X_train.shape)
X_test = vectorizer.transform(data_test.data)
#####
#X_test = X_test.todense()
print("n_samples: %d, n_features: %d" % X_test.shape)

#####
X_all = vstack([X_train, X_test])
#X_all = np.concatenate(([X_train, X_test]), axis = 0)

# binarize labels
y_train[data_train.target < 10] = -1
y_train[data_train.target >= 10] = 1
y_test[data_test.target < 10] = -1
y_test[data_test.target >= 10] = 1
  
Y_all = np.concatenate([y_train, y_test], axis = 0)
#####
times = 100
loss_total = np.zeros(101)
time_total = np.zeros(101)
np.save("loss_hoag.npy", loss_total)
np.save("time_hoag.npy", time_total)

print("done")
for i in range(times):
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