from keras.applications.inception_v3 import InceptionV3
import numpy as np
from keras.models import Model
from keras.layers import  GlobalAveragePooling2D

dogsfishes = np.load('dataset_dog-fish_train-900_test-300.npz')

X_train = np.array(dogsfishes['X_train'])
Y_train = np.array(dogsfishes['Y_train'])
X_test = np.array(dogsfishes['X_test'])
Y_test = np.array(dogsfishes['Y_test'])

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
model_b = Model(inputs=base_model.input, outputs=x)
    
print "Extracting representation for training set"
train_features = np.zeros([len(X_train), 2048])
batch_size = 100
nb_batches = int(float(X_train.shape[0]) / batch_size)
for batch in range(nb_batches):
    ind = range(batch_size*batch, min(batch_size*(1+batch), X_train.shape[0]))

    feature = model_b.predict(X_train[ind])
    train_features[ind] = np.array(feature)
   
np.save("X_train_features_inception.npy", train_features)
    
print "Extracting representation for testing set"
test_features = np.zeros([len(X_test), 2048])
batch_size = 100
nb_batches = int(float(X_test.shape[0]) / batch_size)
for batch in range(nb_batches):
    ind = range(batch_size*batch, min(batch_size*(1+batch), X_test.shape[0]))

    feature = model_b.predict(X_test[ind])
    test_features[ind] = np.array(feature)
    
np.save("X_test_features_inception.npy", test_features)