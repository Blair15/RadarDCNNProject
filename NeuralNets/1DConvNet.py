import numpy as np
import os
import sys
from six.moves import cPickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

import sys
from sklearn.metrics import classification_report,confusion_matrix
import csv
from keras.models import load_model

def load_data():

    nb_train_samples = 1950000

    X_train = np.zeros((nb_train_samples, 128))
    y_train = np.zeros((nb_train_samples,))

    action_labels = ["Walk", "Clap", "ArmSlowerTowards", "ArmFasterTowards", "PickUp", "SitAndStand", "CircleArm"]

    i = 0

    for i in range(1, 6):
        data = cPickle.load(open("train_batch_" + str(i) + ".p", "rb"))
        for label in action_labels:
            for chirp in data[label]:
                X_train[i] = chirp
                y_train[i] = label
                i += 1

    for chirp in X_train[0:10, :]:
        print chirp

load_data()
      
