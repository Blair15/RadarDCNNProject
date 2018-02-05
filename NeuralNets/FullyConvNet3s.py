mport numpy as np
import os
import sys
from six.moves import cPickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.utils import np_utils

import sys
from sklearn.metrics import classification_report,confusion_matrix
import csv
from keras.models import load_model

def shuffleArrays(frames, labels):
    rng_state = np.random.get_state()
    np.random.shuffle(frames)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)

def load_data():

    ##We have 20 3 second recordings for each of the 39 files
    nb_train_samples = 650

    X_train = np.zeros((nb_train_samples, 384000, 2), dtype="uint8")
    y_train = np.zeros((nb_train_samples,), dtype="uint8")

    action_labels = ["Walk", "Clap", "ArmSlowerTowards", "ArmFasterTowards", "PickUp", "SitAndStand", "CircleArm"]

    i = 0

    ## Load data from pickled dicts
    for j in range(1, 6):
        data = cPickle.load(open("train_batch_" + str(j) + ".p", "rb"))
        print "*** loaded train_batch_" + str(j) + " ***"
        for label in action_labels:
            action_number = action_labels.index(label)
            if label in data.keys():
                for frame in data[label]:
                    X_train[i] = frame
                    y_train[i] = action_number
                    i += 1
        print "*** processed train_batch_" + str(j) + " ***"

    ## Shuffle the arrays so that we don't just get all the Clap data first etc.
    shuffleArrays(X_train, y_train)

    nb_test_samples = 130
    X_test = np.zeros((nb_test_samples, 384000, 2), dtype="uint16")
    y_test = np.zeros((nb_test_samples,), dtype="uint16")

    data = cPickle.load(open("test_batch.p", "rb"))

    i = 0
    for label in action_labels:
        action_number = action_labels.index(label)
        if label in data.keys():
            for frame in data[label]:
                X_test[i] = frame
                y_test[i] = action_number
                i+=1

    return (X_train, y_train), (X_test, y_test)

def FullyConv(weights_path=None):
    model = Sequential()
    model.add(Convolution1D(128, 800, padding='same', activation='relu', input_shape=(384000,2)))
    model.add(MaxPooling1D(4))


    model.add(Convolution1D(128, 400, padding='same', activation='relu'))
    model.add(MaxPooling1D((4)))

    model.add(Convolution1D(256, 200, padding='same', activation='relu'))
    model.add(MaxPooling1D(4))

    model.add(Convolution1D(512, 100, padding='same', activation='relu'))
    model.add(MaxPooling1D((4)))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(7, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

batch_size = 1
nb_classes = 7
nb_epoch = 10
(X_train, y_train), (X_test, y_test) = load_data()

print "X_train shape: " + str(X_train.shape)
print str(X_train.shape[0]) + " train samples"
print str(X_test.shape[0]) + " test samples"

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = FullyConv()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)

y_pred = model.predict_classes(X_test)
print str(y_pred)
target_names = ["Walk", "Clap", "ArmSlowerTowards", "ArmFasterTowards", "PickUp", "SitAndStand", "CircleArm"]
print classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names)
print confusion_matrix(np.argmax(Y_test,axis=1), y_pred)
