from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
import tensorflow as tf
from cleverhans.utils import AccuracyReport
import logging
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_train, model_eval, batch_eval, model_argmax
import time
from bilevel_penalty_mt import bilevel_mt
import cPickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import time

batch_size = 32
num_classes = 10
epochs = 25
data_augmentation = True
num_predictions = 20

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

def balanced_subsample(y, s):
    """Return a balanced subsample of the population"""
    sample = []
    # For every label in the dataset
    for label in np.unique(y):
        print(label)
        # Get the index of all images with a specific label
        images = np.where(y==label)[0]
        # Draw a random sample from the images
        random_sample = np.random.choice(images, size=s, replace=False)
        # Add the random sample to our subsample list
        sample += random_sample.tolist()
    return sample

# Pick 50 samples per class from the training samples
train_samples = balanced_subsample(y_train, 1000)
# Pick 50 samples per class from the extra dataset

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

X_all = x_train
Y_all = y_train

tr = 40000
X_val = x_train[train_samples]
Y_val = y_train[train_samples]
print( train_samples[0])
X_train = np.delete(x_train, train_samples, axis=0)
Y_train = np.delete(y_train, train_samples, axis=0)

X_test = x_test
Y_test = y_test

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

corrupt = 30000#12250#24500#36750
correct_points = {-1}
for i in range(corrupt):
    a = np.argmax(Y_train[i])
    Y_train[i] = np.zeros(10)
    #j = np.random.randint(0, 10)
    j = (a + 1)%10
    Y_train[i][j] = 1
    if a == j:
        correct_points.add(i)

for i in range(corrupt, X_train.shape[0]):
    correct_points.add(i)
    
correct_points.remove(-1)
print(len(correct_points))
    
print(len(correct_points))
np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_val.npy", X_val)
np.save("Y_val.npy", Y_val)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)

Keras_model = Sequential()
Keras_model.add(Conv2D(48, (3, 3), border_mode='same', input_shape=x_train.shape[1:]))
Keras_model.add(Activation('relu'))
Keras_model.add(Conv2D(48, (3, 3)))
Keras_model.add(Activation('relu'))
Keras_model.add(MaxPooling2D(pool_size=(2, 2)))
Keras_model.add(Dropout(0.25))
Keras_model.add(Conv2D(96, (3, 3), border_mode='same'))
Keras_model.add(Activation('relu'))
Keras_model.add(Conv2D(96, (3, 3)))
Keras_model.add(Activation('relu'))
Keras_model.add(MaxPooling2D(pool_size=(2, 2)))
Keras_model.add(Dropout(0.25))
Keras_model.add(Conv2D(192, (3, 3), border_mode='same'))
Keras_model.add(Activation('relu'))
Keras_model.add(Conv2D(192, (3, 3)))
Keras_model.add(Activation('relu'))
Keras_model.add(MaxPooling2D(pool_size=(2, 2)))
Keras_model.add(Dropout(0.25))
Keras_model.add(Flatten())
Keras_model.add(Dense(512))
Keras_model.add(Activation('relu'))
Keras_model.add(Dropout(0.5))
Keras_model.add(Dense(256))
Keras_model.add(Activation('relu'))
Keras_model.add(Dropout(0.5))
Keras_model.add(Dense(num_classes))
Keras_model.add(Activation('softmax'))
# Compile the model
Keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Object used to keep track of (and return) key accuracies
report = AccuracyReport()
# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

# Create TF session
sess = tf.Session()
print("Created TensorFlow session.")

set_log_level(logging.DEBUG)
# Define input TF placeholder
x_tf = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
y_tf = tf.placeholder(tf.float32, shape=(None, 10))

# Define TF model graph
scope_model = 'cifar_classifier'
with tf.variable_scope(scope_model):  
   model = Keras_model
preds = model(x_tf)
    
var_model = model.trainable_weights      
saver_model = tf.train.Saver(var_model, max_to_keep = None)
print("Defined TensorFlow model graph.")

######## Bilevel ########## 0.8627
if True:
    print("Bilevel")
    #'''
    train_params = {
        'nb_epochs': 100,
        'batch_size': 128,
        'learning_rate': 1E-3,
        'train_dir': os.path.join(*os.path.split(os.path.join("models", "cifar"))[:-1]),
        'filename': os.path.split( os.path.join("models", "cifar"))[-1]
    }
    rng = np.random.RandomState([2017, 8, 30])
    
    X = X_val
    Y = Y_val
    model_train(sess, x_tf, y_tf, preds, X, Y, args=train_params, save=os.path.exists("models"), rng=rng)
    
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args=eval_params)
    #assert X_test.shape[0] == 10000 - 0, X_test.shape
    print('Test accuracy of the ORACLE: {0}'.format(accuracy))
    accuracy = model_eval(sess, x_tf, y_tf, preds, X_train, Y_train, args=eval_params)
    #assert X_test.shape[0] == 10000 - 0, X_test.shape
    print('Train accuracy of the ORACLE: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy
    batch_size = 1024
    
    Y_train_init = []
    nb_batches = int(np.floor(float(X_train.shape[0]) / batch_size))
    left = X_train.shape[0] - nb_batches * batch_size
    for batch in range(nb_batches):
        ind = range(batch_size*batch, min(batch_size*(1+batch), X_train.shape[0]))
        if batch == 0:
            Y_train_init = model_argmax(sess, x_tf, preds, X_train[ind, :])
        else:
            Y_train_init = np.concatenate((Y_train_init, model_argmax(sess, x_tf, preds, X_train[ind, :])))
    Y_train_init = np.concatenate((Y_train_init, model_argmax(sess, x_tf, preds, X_train[-left:])))
    
    importance_atan = np.ones((X_train.shape[0]))
    c = 0
    d = 0
    for i in range(Y_train.shape[0]):
        a = np.argwhere(Y_train[i] == 1)
        b = Y_train_init[i]
        #print(b)
        if a == b:
            c += 1
            importance_atan[i] = np.arctanh(2.*0.8-1)
        else:
            d += 1
            importance_atan[i] = np.arctanh(2.*0.2-1)
    
    print(c)
    print(d)
    
    # Normalize
    importance = 0.5*(np.tanh(importance_atan)+1.)
    importance = 0.5*importance/np.mean(importance)
    importance = np.maximum(.00001,np.minimum(.99999,importance))
    importance_atan = np.arctanh(2.*importance-1.)
    np.save("importance_1.npy", importance)
    
    print(np.max(importance))
    print(np.min(importance))
    '''
    mimp_points = np.argwhere(importance >= np.max(importance)).flatten()
    print(mimp_points)
    recovered = 0
    extra_points = 0
    for imp_pts in range(len(mimp_points)):
        if mimp_points[imp_pts] in correct_points:
            recovered += 1
        else:
            extra_points += 1
    '''
    sort_pts = np.argsort(importance)
    till = X_train.shape[0] - corrupt
    recovered = 0
    extra_points = 0
    for imp_pts in range(till):
        if sort_pts[corrupt + imp_pts] in correct_points:
            recovered += 1
        else:
            extra_points += 1
    
    print(recovered)
    print(extra_points)
    print(len(correct_points))
    '''
    #'''
    lr_outer = 3
    lr_inner = 1E-3
    rho = 0
    sig = lr_inner
    batch_size = min(128, len(X_val))
    nepochs = 50#10#20#10#50#15#25
    height = 32
    width = 32
    nch = 3
    nb_classes = 10
    rho_t = 1E-2
    lamb_t = 1E0
    
    tick = time.time() 
    blmt = bilevel_mt(sess, model, var_model, batch_size, lr_outer, lr_inner, height, width, nch, nb_classes, rho, sig)
    print("--- %s seconds ---" % (time.time() - tick))
    '''
    importance_atan = np.ones((X_train.shape[0])) * np.arctanh(2.*0.8-1)
    # Normalize
    importance = 0.5*(np.tanh(importance_atan)+1.)
    importance = 0.5*importance/np.mean(importance)
    importance = np.maximum(.00001,np.minimum(.99999,importance))
    importance_atan = np.arctanh(2.*importance-1.)
    np.save("importance_1.npy", importance)
    sess.run(tf.global_variables_initializer())
    #'''
    for epoch in range(nepochs):
        print("epoch: ")
        print(epoch)
        
        tick = time.time()        
        #nb_batches = int(np.ceil(float(Ntrain) / FLAGS.batch_size))
        nb_batches = int(np.floor(float(X_train.shape[0]) / batch_size))
        index_shuf = np.arange(X_train.shape[0])
        np.random.shuffle(index_shuf)
        
        for batch in range(nb_batches):
            #print("batch" + batch)
            ind = range(batch_size*batch, min(batch_size*(1+batch), X_train.shape[0]))
            #if len(ind)<FLAGS.batch_size:
            #    ind.extend([np.random.choice(Ntrain,FLAGS.batch_size-len(ind))])
            ind_val = np.random.choice(X_val.shape[0], size=(batch_size), replace=False)
            l1, l2_1, l3, timp_atan = blmt.train(X_train[index_shuf[ind],:], Y_train[index_shuf[ind],:], X_val[ind_val,:], Y_val[ind_val,:], importance_atan[index_shuf[ind]],rho_t,lamb_t)
            importance_atan[index_shuf[ind]] = timp_atan
        print("--- %s seconds ---" % (time.time() - tick))
        
        rho_t *= 1.05
        lamb_t *= 0.99
        
        ## Should I renormalize importance_atan?
        if True:
            importance = 0.5 * (np.tanh(importance_atan)+1.) # scale to beteen [0 1] from [-1 1]
            importance = 0.5 * X_train.shape[0] * importance / sum(importance)
            importance = np.maximum(.00001, np.minimum(.99999, importance))
            importance_atan = np.arctanh(0.99999 * (2. * importance - 1.))
            
        
        if epoch %1 == 0:
            print('epoch %d: rho=%f, lamb=%f, f=%f, gvnorm=%f, lamb_g=%f, total=%f'%(epoch, rho_t, lamb_t, l1, l2_1, l3, l1+l2_1+l3))
            print('mean ai=%f, mean I(ai>0.1)=%f'%(np.mean(importance),len(np.where(importance>0.1)[0])/np.float(X_train.shape[0])))
            '''
            mimp_points = np.argwhere(importance >= 0.5).flatten()
            print(mimp_points)
            recovered = 0
            extra_points = 0
            for imp_pts in range(len(mimp_points)):
                if mimp_points[imp_pts] in correct_points:
                    recovered += 1
                else:
                    extra_points += 1
            '''
            sort_pts = np.argsort(importance)
            till = X_train.shape[0] - corrupt
            recovered = 0
            extra_points = 0
            for imp_pts in range(till):
                if sort_pts[corrupt + imp_pts] in correct_points:
                    recovered += 1
                else:
                    extra_points += 1
            
            print(recovered)
            print(extra_points)
            print(len(correct_points))
            
        if epoch % 1 == 0:
            print('acc = %f'%(model_eval(sess, x_tf, y_tf, preds, X_val, Y_val, args={'batch_size':batch_size})))
            #print('epoch %d: loss_inner=%f, loss_outer1=%f, loss_outer2=%f'%(epoch,lin,lout1,lout2))   
                
    saver_model.save(sess,'./model_bilevel_mt_cifar.ckpt')
    #importance = 0.5*(np.tanh(importance_atan)+1.)
    np.save('./importance.npy',importance)
    

######## Random ########## 0.8627
if False:
    ## Now, retrain temp model with the reduced set and evaluate accuracy
    importance = np.arange(0, X_train.shape[0])
    np.random.shuffle(importance)
    np.save('./importance.npy',importance)