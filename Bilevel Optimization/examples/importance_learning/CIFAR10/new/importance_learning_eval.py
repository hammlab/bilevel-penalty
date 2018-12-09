from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import logging
from cleverhans.utils import set_log_level
import time
from cifar10_keras_model import CIFAR

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

num_classes = 10
data_augmentation = True

X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")
X_val = np.load("X_val.npy")
Y_val = np.load("Y_val.npy")
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")

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
   model = CIFAR(X_train, num_classes)
preds = model(x_tf)
    
var_model = model.trainable_weights      
saver_model = tf.train.Saver(var_model, max_to_keep = None)
print("Defined TensorFlow model graph.")

corrupt = int(0.5 * len(X_train))

for pct in [1]: 
    if pct == 0:
        # Train with Val + clean training data
        X = np.concatenate((X_train[corrupt:], X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train[corrupt:], Y_val), axis = 0)
        print(len(Y)) 
    
    elif pct == 1:
        # Train with Validation data
        X = X_val
        print(len(X))
        Y = Y_val
        print(len(Y))
        
    elif pct == 2:
        # Train with Val + all training data
        X = np.concatenate((X_train, X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train, Y_val), axis = 0)
        print(len(Y)) 
    
    elif pct == 3:
        # Train with Val + important training data
        importance = np.load("importance_noise.npy")
        ind = np.argwhere(importance >= 0.1).flatten()
        print(len(ind))
        print(importance[ind])
        print(ind)
        X = np.concatenate((X_train[ind], X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train[ind], Y_val), axis = 0)
        print(len(Y)) 
   
    elif pct == 4:
        # Train with Val + important training data (importance learnt without bilevel optimization i.e. just based on validation data)
        importance = np.load("importance_1.npy")
        ind = np.argwhere(importance >= np.max(importance)).flatten()
        print(len(ind))
        print(importance[ind])
        print(ind)
        X = np.concatenate((X_train[ind], X_val), axis = 0)
        print(len(X))
        Y = np.concatenate((Y_train[ind], Y_val), axis = 0)
        print(len(Y)) 
    
    
    datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
    start = time.time()

    #Train with data augmentation
    if data_augmentation:
        model_info = model.fit_generator(datagen.flow(X, Y, batch_size = 128),
                                         samples_per_epoch = X.shape[0], nb_epoch = 100, 
                                         validation_data = (X_test, Y_test))
    else:
        model_info = model.fit(X, Y, batch_size = 128,
                                         nb_epoch = 100, 
                                         validation_data = (X_test, Y_test))
    
    end = time.time()
    print("Model took %0.2f seconds to train" % (end - start))
    print("Accuracy on test data is: %0.2f" % accuracy(X_test, Y_test, model))

