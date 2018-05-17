import pandas as pd 
import numpy as np 
import matplotlib
from matplotlib import pyplot as plt 
import sklearn as sl 
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import  train_test_split
import tensorflow as tf
import os
import keras
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
import logging
from cleverhans.utils import set_log_level
from bilevel_cs import bilevel_cs
import time

# Data Formatting 
data = pd.read_csv("breast_cancer_data.csv",header=0)
data = data.drop("a", 1)
mapping = {2.0 : 0, 4.0 : 1}
data['k'] = data['k'].map(mapping)
features = list(data.columns[:9]) # Appending all the columns in feature vector
data_X = data[features].values
print(data_X[0])
             
features_Y = list(data.columns[9])
data_Y = data[features_Y].values 
print(np.argwhere(data_Y < 0))             
old_y = data[features_Y].values 
data_Y = keras.utils.to_categorical(data_Y, 2)
print(data_Y[0])


# Model
k_model = Sequential()
k_model.add(Dense(input_dim=9, units=50))
#k_model.add(Dense(input_dim=9, units=50))
k_model.add(Dense(input_dim=9, units=2))
k_model.add(Activation("softmax"))
k_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

# Create TF session
sess = tf.Session()
print("Created TensorFlow session.")
#set_log_level(logging.DEBUG)
# Define input TF placeholder
x_tf = tf.placeholder(tf.float32, shape=(None, 9))
y_tf = tf.placeholder(tf.float32, shape=(None, 2))

# Define TF model graph
scope_model = 'breast_cancer_classifier'
with tf.variable_scope(scope_model):    
    model = k_model
preds = model(x_tf)
    
var_model = model.trainable_weights       
saver_model = tf.train.Saver(var_model,max_to_keep=None)
print("Defined TensorFlow model graph.")

###########################################################################
# Training the model using TensorFlow
###########################################################################

train_params = {
    'nb_epochs': 100,
    'batch_size': 512,
    'learning_rate': 1E-3,
    'train_dir': os.path.join(*os.path.split(os.path.join("models", "breast_cancer"))[:-1]),
    'filename': os.path.split( os.path.join("models", "breast_cancer"))[-1]
}

features_t = 9
times = 5
rng = np.random.RandomState([2017, 8, 30])
vals = np.zeros([features_t, 2])

for feature in range(features_t):
    vals_i = 0
    for inde in range(times):
        print(feature)
        print(inde)
        index = np.arange(len(data_X))
        np.random.shuffle(index)
        data_X = data_X[index]
        data_Y = data_Y[index]
        old_y = old_y[index]
        
        probs = np.zeros(data_X.shape[0])
        for it_x in range(data_X.shape[0]):
            if data_X[it_x][feature] > 5:
                probs[it_x] = 0.8
            else:
                probs[it_x] = 0.2
                
        probs = probs / np.sum(probs)
        
        tr = int(0.25 * len(data_X))
        ind = np.random.choice(data_X.shape[0], tr, p = probs)
        X_train = data_X[ind]
        Y_train = data_Y[ind]
        
        left_indices = np.delete(np.arange(data_X.shape[0]), ind)
        X_val = data_X[left_indices[:25]]
        Y_val = data_Y[left_indices[:25]]

        X_test = data_X[left_indices[25:125]]
        Y_test = data_Y[left_indices[25:125]]
              
        if True:
            sess.run(tf.global_variables_initializer())
            
            lr_outer = 1#0.1
            lr_inner = 1E-3
            rho = 0 # 1E1 for non L1 radius
            sig = lr_inner
            batch_size = 16
            nepochs = 100
            features = 9
            nb_classes = 2
            blmt = bilevel_cs(sess, model, var_model, batch_size, lr_outer, lr_inner, features, nb_classes, rho, sig)
            importance_atan = np.ones((X_train.shape[0]))
            
            for epoch in range(nepochs):
                tick = time.time()        
                #nb_batches = int(np.ceil(float(Ntrain) / FLAGS.batch_size))
                nb_batches = int(np.floor(float(X_train.shape[0]) / batch_size))
                index_shuf = np.arange(X_train.shape[0])
                np.random.shuffle(index_shuf)
                
                for batch in range(nb_batches):
                    ind = range(batch_size*batch, min(batch_size*(1+batch), X_train.shape[0]))
                    #if len(ind)<FLAGS.batch_size:
                    #    ind.extend([np.random.choice(Ntrain,FLAGS.batch_size-len(ind))])
                    ind_val = np.random.choice(X_val.shape[0], size=(batch_size), replace=False)
                    lin,lout1,lout2,lout3,timp_atan = blmt.train(X_train[index_shuf[ind],:], Y_train[index_shuf[ind],:], X_val[ind_val,:], Y_val[ind_val,:], importance_atan[index_shuf[ind]])
                    importance_atan[index_shuf[ind]] = timp_atan
            
                ## Should I renormalize importance_atan?
                if True:
                    importance = 0.5 * (np.tanh(importance_atan)+1.) # scale to beteen [0 1] from [-1 1]
                    importance = 0.5 * X_train.shape[0] * importance / sum(importance)
                    importance = np.maximum(.00001, np.minimum(.99999, importance))
                    importance_atan = np.arctanh(0.99999 * (2. * importance - 1.))
                    
                
                if epoch %1 == 0:
                    print('epoch %d: loss_inner=%f, loss_inner2=%f, loss_outer1=%f, loss_outer3=%f'%(epoch,lin,lout1,lout2,lout3))
                    print('mean ai=%f, mean I(ai>0.1)=%f'%(np.mean(importance),len(np.where(importance>0.1)[0])/np.float(X_train.shape[0])))
                    
                if epoch % 1 == 0:
                    print('acc = %f'%(model_eval(sess, x_tf, y_tf, preds, X_val, Y_val, args={'batch_size':512})))
                    #print('epoch %d: loss_inner=%f, loss_outer1=%f, loss_outer2=%f'%(epoch,lin,lout1,lout2))   
                        
            saver_model.save(sess,'./model_bilevel_breast_cancer.ckpt')
            #importance = 0.5*(np.tanh(importance_atan)+1.)
            np.save('./importance.npy',importance)
            vals[feature][0] += model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args={'batch_size':512})
            print('\n\nBilevel acc = %f\n\n'%(model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args={'batch_size':512})))
                
        if True:
            sess.run(tf.global_variables_initializer())
            ## Now, retrain temp model with the reduced set and evaluate accuracy
            model_path = os.path.join('models', 'temp_breast_cancer')
            random_params = {'nb_epochs':100, 'batch_size':512, 'learning_rate':1E-3, 
                'train_dir': os.path.join(*os.path.split(model_path)[:-1]),
                'filename': os.path.split(model_path)[-1]
            }
            
            X = np.concatenate((X_train, X_val), axis = 0)
            Y = np.concatenate((Y_train, Y_val), axis = 0)

            model_train(sess, x_tf, y_tf, preds, X, Y, args=random_params)
            vals[feature][1] += model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args={'batch_size':512})
            print('Random acc = %f'%(model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args={'batch_size':512})))
    print(vals/(times))    
print(vals/(times))