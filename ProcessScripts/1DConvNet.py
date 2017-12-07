import numpy as np
import os
import sys
from six.moves import cPickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import np_utils

import sys
from sklearn.metrics import classification_report,confusion_matrix
import csv
from keras.models import load_model

def load_data():

    nb_train_samples = 1950000

    X_train = np.zeros((nb_train_samples, 128), dtype="uint8")
    y_train = np.zeros((nb_train_samples,), dtype="uint8")

    action_labels = ["Walk", "Clap", "ArmSlowerTowards", "ArmFasterTowards", "PickUp", "SitAndStand", "CircleArm"]

    i = 0

    for j in range(1, 6):
        data = cPickle.load(open("train_batch_" + str(j) + ".p", "rb"))
        print "*** loaded train_batch_" + str(j) + " ***" 
        for label in action_labels:
            action_number = action_labels.index(label)
            for chirp in data[label]:
                X_train[i] = chirp
                y_train[i] = action_number
                i += 1
        print "*** processed train_batch_" + str(j) + " ***"

    X_train = np.reshape(X_train, (nb_train_samples,128,1))

    nb_test_samples = 390000
    X_test = np.zeros((nb_test_samples, 128), dtype="uint8")
    y_test = np.zeros((nb_test_samples,), dtype="uint8")

    data = cPickle.load(open("test_batch.p", "rb"))
    
    i = 0
    for label in action_labels:
        action_number = action_labels.index(label)
        for chirp in data[label]:
            X_test[i] = chirp
            y_test[i] = action_number
            i+=1
    
    X_test = np.reshape(X_test, (nb_test_samples,128,1))

    return (X_train, y_train), (X_test, y_test) 

batch_size = 1000
nb_classes = 7
nb_epoch = 1
(X_train, y_train), (X_test, y_test) = load_data()

print "X_train shape: " + str(X_train.shape)
print str(X_train.shape[0]) + " train samples"
print str(X_test.shape[0]) + " test samples"

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution1D(64, 3, padding='same', input_shape=(128, 1)))
model.add(Activation('relu'))
model.add(Convolution1D(64,3))
model.add(Activation('relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.25))

model.add(Convolution1D(128, 3, padding='same'))
model.add(Activation('relu'))
model.add(Convolution1D(128, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)

y_pred = model.predict_classes(X_test)
print str(y_pred)
target_names = ["Walk", "Clap", "ArmSlowerTowards", "ArmFasterTowards", "PickUp", "SitAndStand", "CircleArm"]
print classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names)
print confusion_matrix(np.argmax(Y_test,axis=1), y_pred)
