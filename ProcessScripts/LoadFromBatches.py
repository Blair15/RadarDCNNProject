import numpy as np
import os
import sys
from six.moves import cPickle
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD
#from keras.utils import np_utils

import sys
#from sklearn.metrics import classification_report,confusion_matrix
import csv
#from keras.models import load_model

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

    nb_test_samples = 390000
    X_test = np.zeros((nb_test_samples, 128), dtype="uint8")
    y_test = np.zeros((nb_test_samples,), dtype="uint8")

    data = cPickle.load(open("test_batch", "rb"))
    
    i = 0
    for label in action_labels:
        action_number = action_labels.index(label)
        for chirp in data[label]:
            X_test[i] = chirp
            y_test[i] = action_number
            i+=1

    return (X_train, y_train), (X_test, y_test) 

load_data()
      
