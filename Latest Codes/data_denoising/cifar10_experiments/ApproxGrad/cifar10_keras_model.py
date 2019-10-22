from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def CIFAR(x, num_classes):
    Keras_model = Sequential()
    Keras_model.add(Conv2D(48, (3, 3), padding='same', input_shape=x.shape[1:]))
    Keras_model.add(Activation('relu'))
    Keras_model.add(Conv2D(48, (3, 3)))
    Keras_model.add(Activation('relu'))
    Keras_model.add(MaxPooling2D(pool_size=(2, 2)))
    Keras_model.add(Dropout(0.25))
    
    Keras_model.add(Conv2D(96, (3, 3), padding='same'))
    Keras_model.add(Activation('relu'))
    Keras_model.add(Conv2D(96, (3, 3)))
    Keras_model.add(Activation('relu'))
    Keras_model.add(MaxPooling2D(pool_size=(2, 2)))
    Keras_model.add(Dropout(0.25))
    
    Keras_model.add(Conv2D(192, (3, 3), padding='same'))
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
    return Keras_model