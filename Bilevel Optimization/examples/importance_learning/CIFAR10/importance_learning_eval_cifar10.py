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
import cPickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import time
from keras.preprocessing.image import ImageDataGenerator

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

batch_size = 32
num_classes = 10
epochs = 25
data_augmentation = True
num_predictions = 20

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")
X_val = np.load("X_val.npy")
Y_val = np.load("Y_val.npy")
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")



Keras_model = Sequential()
Keras_model.add(Conv2D(48, (3, 3), border_mode='same', input_shape=X_train.shape[1:]))
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

train_params = {
    'nb_epochs': 100,
    'batch_size': 128,
    'learning_rate': 1E-3,
}
rng = np.random.RandomState([2017, 8, 30])
importance = np.load("importance.npy")
batch_size = min(1024, len(X_val))  

corrupt = 30000#12250#24500#36750
for pct in [4]:#3, 4, 0, 1, 2]: # percentile from top
    #noise = 1E-6*np.random.normal(size=importance.shape)
    #th = np.percentile(importance + noise,100.-pct) # percentile from bottom
    #ind = pct * X_train.shape[0] #np.where(importance + noise >=th)[0]
    ## Make sure the training starts with random initialization
    if pct == 0:
        #ind = np.argsort(importance)[-pct * 1000 :]
        #print(np.maximum(importance), np.minimum(importance))
        X = np.concatenate((X_train[corrupt:], X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train[corrupt:], Y_val), axis = 0)
        print(len(Y)) 
    
    elif pct == 1:
        X = X_val
        print(len(X))
        Y = Y_val
        print(len(Y))
        
    elif pct == 2:
        #ind = np.argsort(importance)[-pct * 1000 :]
        #print(np.maximum(importance), np.minimum(importance))
        X = np.concatenate((X_train, X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train, Y_val), axis = 0)
        print(len(Y)) 
    
    elif pct == 3:
        #ind = np.argsort(importance)[-pct * 1000 :]
        ind = np.argwhere(importance >= 0.1).flatten()
        #ind = np.argsort(importance)[corrupt:]
        print(len(ind))
        print(importance[ind])
        print(ind)
        #print(np.maximum(importance), np.minimum(importance))
        X = np.concatenate((X_train[ind], X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train[ind], Y_val), axis = 0)
        print(len(Y)) 
   
    elif pct == 4:
        
        importance = np.load("importance_1.npy")
        ind = np.argwhere(importance >= np.max(importance)).flatten()
        #ind = np.argsort(importance)[corrupt:]
        print(len(ind))
        print(importance[ind])
        print(ind)
        #print(np.maximum(importance), np.minimum(importance))
        X = np.concatenate((X_train[ind], X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train[ind], Y_val), axis = 0)
        print(len(Y)) 
    
    
    '''    
    sess.run(tf.global_variables_initializer())
    model_train(sess, x_tf, y_tf, preds, X, Y, args=train_params, save=os.path.exists("models"), rng=rng)
    eval_params = {'batch_size': 128}
    accuracy = model_eval(sess, x_tf, y_tf, preds, X_test, Y_test, args=eval_params)
    print('Test accuracy of the ORACLE: {0}'.format(accuracy))
    '''
    datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
    # train the model
    start = time.time()
    # Train the model
    model_info = Keras_model.fit_generator(datagen.flow(X, Y, batch_size = 128),
                                     samples_per_epoch = X.shape[0], nb_epoch = 100, 
                                     validation_data = (X_test, Y_test))
    end = time.time()
    print("Model took %0.2f seconds to train" % (end - start))
    print("Accuracy on test data is: %0.2f" % accuracy(X_test, Y_test, Keras_model))

